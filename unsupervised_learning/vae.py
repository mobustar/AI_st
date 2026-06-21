"""
==============================================================
  vae.py | 変分オートエンコーダ (VAE)
==============================================================
PyTorch を使用。CUDA が使えれば自動的に利用する。

VAE (Variational Autoencoder) は生成モデルの一種。
通常のオートエンコーダと違い、潜在空間が確率分布になるため、
潜在変数をサンプリングして「新しいデータ」を生成できる。

数学:
  エンコーダ: q_φ(z|x) ≈ N(μ(x), σ²(x)I)
  デコーダ:   p_θ(x|z)
  目的関数 (ELBO: Evidence Lower BOund) を最大化:
      ELBO = E[log p_θ(x|z)] - KL(q_φ(z|x) || p(z))
           = 再構成損失 (Reconstruction Loss) - KL 正則化項

  再構成損失: BCE (Binary Cross Entropy) または MSE
      - クロスエントロピー: ピクセル値が [0,1] に正規化されたとき
  KL 正則化:
      KL(N(μ,σ²) || N(0,1)) = -0.5 * Σ(1 + log σ² - μ² - σ²)
      = 潜在空間を標準正規分布 N(0,I) に近づける

  再パラメータ化トリック:
      z = μ + σ * ε,  ε ~ N(0,I)
      これにより z に関する勾配が流せる (バックプロパゲーション可能)

クラス:
  VAE         - PyTorch モジュール (エンコーダ + デコーダ)
  VAETrainer  - 学習ループ + ELBO 計算
"""

import numpy as np

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    _TORCH_AVAILABLE = True
except ImportError:
    _TORCH_AVAILABLE = False


def _check_torch():
    if not _TORCH_AVAILABLE:
        raise ImportError(
            "PyTorch が見つかりません。'pip install torch' でインストールしてください。"
        )


# ─── VAE モジュール ─────────────────────────────────────────
class VAE(nn.Module if _TORCH_AVAILABLE else object):
    """
    変分オートエンコーダ。

    Encoder: Linear → ReLU → Linear → (μ, log σ²)
    Decoder: Linear → ReLU → Linear → Sigmoid

    Architecture:
        入力: input_dim 次元
        隠れ層: hidden_dim (エンコーダ・デコーダ共通)
        潜在空間: latent_dim

        Encoder:
            Linear(input_dim → hidden_dim) → ReLU
            → Linear(hidden_dim → latent_dim)  [μ]
            → Linear(hidden_dim → latent_dim)  [log σ²]

        Decoder:
            Linear(latent_dim → hidden_dim) → ReLU
            → Linear(hidden_dim → input_dim) → Sigmoid

    Args:
        input_dim  : 入力次元数
        hidden_dim : 隠れ層のユニット数
        latent_dim : 潜在変数の次元数 (2 にすると 2D 可視化できる)
    """

    def __init__(self, input_dim: int, hidden_dim: int = 128,
                 latent_dim: int = 2):
        _check_torch()
        super().__init__()

        # エンコーダ
        self.enc_fc1 = nn.Linear(input_dim, hidden_dim)
        self.enc_mu  = nn.Linear(hidden_dim, latent_dim)
        self.enc_lv  = nn.Linear(hidden_dim, latent_dim)  # log variance

        # デコーダ
        self.dec_fc1 = nn.Linear(latent_dim, hidden_dim)
        self.dec_out = nn.Linear(hidden_dim, input_dim)

    def encode(self, x):
        """x → (μ, log σ²)"""
        h = F.relu(self.enc_fc1(x))
        return self.enc_mu(h), self.enc_lv(h)

    def reparameterize(self, mu, log_var):
        """
        再パラメータ化トリック: z = μ + σ * ε, ε ~ N(0,I)

        log_var = log σ² なので:
            σ = exp(0.5 * log_var)
        """
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        return mu + std * eps

    def decode(self, z):
        """z → x̂ (再構成)"""
        h = F.relu(self.dec_fc1(z))
        return torch.sigmoid(self.dec_out(h))

    def forward(self, x):
        """x → (x̂, μ, log σ²)"""
        mu, log_var = self.encode(x)
        z = self.reparameterize(mu, log_var)
        return self.decode(z), mu, log_var

    def sample(self, n: int, device=None):
        """潜在空間から z ~ N(0,I) をサンプリングして新データを生成"""
        if device is None:
            device = next(self.parameters()).device
        z = torch.randn(n, self.enc_mu.out_features, device=device)
        with torch.no_grad():
            return self.decode(z).cpu().numpy()

    def encode_numpy(self, X: np.ndarray, device=None) -> np.ndarray:
        """NumPy 配列 → 潜在空間の μ を返す (可視化用)"""
        if device is None:
            device = next(self.parameters()).device
        self.eval()
        with torch.no_grad():
            x_t = torch.tensor(X, dtype=torch.float32, device=device)
            mu, _ = self.encode(x_t)
            return mu.cpu().numpy()


# ─── ELBO 損失関数 ──────────────────────────────────────────
def elbo_loss(x_recon, x, mu, log_var, beta: float = 1.0):
    """
    ELBO = 再構成損失 + β × KL 発散

    Args:
        x_recon : デコーダ出力  shape (batch, d)
        x       : 入力          shape (batch, d)
        mu      : エンコーダの μ  shape (batch, latent_dim)
        log_var : エンコーダの log σ²  shape (batch, latent_dim)
        beta    : KL 項の重み (β-VAE: β>1 で潜在変数の絡み合いを促進)

    Returns:
        loss : スカラー (負 ELBO / バッチサイズ)

    数式:
        再構成損失 = -Σ [x log x̂ + (1-x) log(1-x̂)]   (BCE の総和)
        KL 正則化  = -0.5 Σ (1 + log σ² - μ² - σ²)
        loss = (recon_loss + β * kl_loss) / batch_size
    """
    # BCE を要素ごとに計算し総和
    recon_loss = F.binary_cross_entropy(x_recon, x, reduction="sum")
    # KL ダイバージェンス
    kl_loss = -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp())
    return (recon_loss + beta * kl_loss) / x.size(0)


# ─── VAE トレーナー ─────────────────────────────────────────
class VAETrainer:
    """
    VAE の学習ループを管理する。

    学習フロー:
        1. MiniBatch (batch_size) で入力を取り出す
        2. Forward: x → (x̂, μ, log σ²)
        3. ELBO 損失を計算
        4. バックプロパゲーション & Adam で更新
        5. エポックごとに平均損失を記録

    Args:
        model      : VAE インスタンス
        lr         : Adam の学習率
        batch_size : ミニバッチサイズ
        device     : "cpu" または "cuda"
    """

    def __init__(self, model: "VAE", lr: float = 1e-3,
                 batch_size: int = 64, device: str = None):
        _check_torch()
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device     = torch.device(device)
        self.model      = model.to(self.device)
        self.opt        = torch.optim.Adam(model.parameters(), lr=lr)
        self.batch_size = batch_size

    def train(self, X: np.ndarray, n_epochs: int = 100,
              beta: float = 1.0, verbose_every: int = 10) -> list:
        """
        X で VAE を学習する。

        Args:
            X            : 入力データ shape (n, d)  値域 [0, 1] を前提
            n_epochs     : 学習エポック数
            beta         : β-VAE の KL 重み
            verbose_every: 何エポックごとに損失を表示するか (0 で無音)

        Returns:
            losses : エポックごとの平均 ELBO 損失のリスト
        """
        X_t = torch.tensor(X, dtype=torch.float32, device=self.device)
        n = len(X_t)
        losses = []

        self.model.train()
        for epoch in range(1, n_epochs + 1):
            # シャッフル
            perm = torch.randperm(n, device=self.device)
            epoch_loss = 0.0
            n_batches  = 0

            for start in range(0, n, self.batch_size if self.batch_size > 0 else n):
                end = min(start + (self.batch_size if self.batch_size > 0 else n), n)
                x_batch = X_t[perm[start:end]]

                self.opt.zero_grad()
                x_recon, mu, log_var = self.model(x_batch)
                loss = elbo_loss(x_recon, x_batch, mu, log_var, beta=beta)
                loss.backward()
                self.opt.step()

                epoch_loss += loss.item()
                n_batches  += 1

            avg_loss = epoch_loss / n_batches
            losses.append(avg_loss)
            if verbose_every > 0 and epoch % verbose_every == 0:
                print(f"    Epoch {epoch:>4}/{n_epochs}  loss={avg_loss:.4f}")

        self.model.eval()
        return losses
