"""
==============================================================
  main.py | 教師なし学習: 全手法の比較実行
==============================================================
4 種類の教師なし学習手法を順に実行・評価する。

フェーズ:
  1. クラスタリング  - K-means / DBSCAN / Agglomerative を比較
  2. 次元削減       - PCA / t-SNE で高次元データを 2D 可視化
  3. 異常検知       - Isolation Forest / LOF を比較
  4. 生成モデル (VAE) - 変分オートエンコーダで潜在空間を学習

実行:
    python main.py
"""

import numpy as np

from data      import make_blobs, make_moons, make_circles, make_anomaly, make_swiss_roll
from cluster   import get_cluster
from reduce    import get_reducer
from anomaly   import get_anomaly
from metrics   import (silhouette_score, davies_bouldin_score, adjusted_rand_index,
                       anomaly_metrics, reconstruction_error, trustworthiness)
from visualize import (scatter_2d, scatter_compare, loss_curve,
                       cluster_summary, anomaly_summary)


# ─── フェーズ 1: クラスタリング ──────────────────────────────
def run_clustering():
    print("=" * 66)
    print("  フェーズ 1/4 : クラスタリング")
    print("=" * 66)
    print("""  【処理内容】
    ラベルなしデータを「似たグループ」に自動分類します。
    3 種のアルゴリズムを 3 種のデータ形状で評価します。

  【評価指標】
    シルエットスコア: -1~+1 (高いほど良い)
      各点の「クラスタ内凝集度」と「クラスタ間分離度」のバランス
    Davies-Bouldin 指数: 小さいほど良い
      クラスタ間距離 / クラスタ内散布の比 (良いクラスタは小さい値)
    ARI (調整ランド指数): -1~+1 (1=真ラベルと完全一致)
      真のラベルと予測ラベルの一致率 (偶然の一致を補正)
""")

    # データセット: 形状ごとに比較
    datasets = [
        ("blobs (球状クラスタ)",   make_blobs(n_samples=300, n_clusters=3, seed=42)),
        ("moons (三日月形)",       make_moons(n_samples=300, noise=0.1, seed=42)),
        ("circles (同心円)",       make_circles(n_samples=300, noise=0.05, seed=42)),
    ]

    # K-means の k はデータに合わせて固定
    algo_configs = [
        ("K-means (k=3)",    "kmeans",        {"k": 3, "n_init": 10, "seed": 0}),
        ("DBSCAN (ε=0.4)",   "dbscan",        {"eps": 0.4, "min_samples": 5}),
        ("Agglomerative",    "agglomerative", {"n_clusters": 3}),
    ]

    for ds_name, ds in datasets:
        print(f"  ─── データ: {ds_name} ───")
        scatter_2d(ds.X, ds.y, title=f"{ds_name} (真ラベル)")

        for algo_name, algo_key, algo_kwargs in algo_configs:
            clf = get_cluster(algo_key, **algo_kwargs)
            clf.fit(ds.X)
            labels = clf.labels_

            sil = silhouette_score(ds.X, labels, sample_size=200, seed=0)
            db  = davies_bouldin_score(ds.X, labels)
            ari = adjusted_rand_index(ds.y, labels)

            cluster_summary(algo_name, labels, sil, db, ari)
            scatter_2d(ds.X, labels, title=f"{algo_name} 予測")
        print()


# ─── フェーズ 2: 次元削減 ────────────────────────────────────
def run_dimensionality_reduction():
    print("=" * 66)
    print("  フェーズ 2/4 : 次元削減")
    print("=" * 66)
    print("""  【処理内容】
    高次元データを 2 次元に圧縮して可視化します。
    スイスロール (3D) → 2D に削減し、元の構造が保たれるか確認します。

  【評価指標】
    PCA 分散説明率: 各主成分が全分散の何割を説明するか
    再構成誤差 (MSE): 低次元→元次元に再構成したときの誤差
    Trustworthiness: 近傍関係がどれだけ保存されているか (0~1)
""")

    ds = make_swiss_roll(n_samples=300, noise=0.1, seed=42)
    X, y = ds.X, ds.y
    print(f"  入力データ: {ds.name}  shape={X.shape}")

    # ── PCA ──
    print("\n  ─── PCA (主成分分析) ───")
    pca = get_reducer("pca", n_components=2)
    X_pca = pca.fit_transform(X)
    X_recon = pca.inverse_transform(X_pca)
    recon_err = reconstruction_error(X, X_recon)
    trust_pca = trustworthiness(X, X_pca, n_neighbors=10)

    ratio = pca.explained_variance_ratio_
    print(f"  分散説明率: PC1={ratio[0]:.3f}  PC2={ratio[1]:.3f}"
          f"  累積={ratio.sum():.3f}")
    print(f"  再構成誤差 (MSE): {recon_err:.4f}  (0 に近いほど情報損失が少ない)")
    print(f"  Trustworthiness: {trust_pca:.4f}  (1 に近いほど近傍構造を保存)")
    scatter_2d(X_pca, y, title="PCA 2次元投影 (色=スイスロールの位置)")

    # ── t-SNE ──
    print("\n  ─── t-SNE ───")
    print("  (O(n²) 実装のため少し時間がかかります...)")
    tsne = get_reducer("tsne", n_components=2, perplexity=30.0,
                       n_iter=500, lr=200.0, seed=42)
    X_tsne = tsne.fit_transform(X)
    trust_tsne = trustworthiness(X, X_tsne, n_neighbors=10)
    print(f"  Trustworthiness: {trust_tsne:.4f}")
    scatter_2d(X_tsne, y, title="t-SNE 2次元投影 (色=スイスロールの位置)")

    print("\n  PCA vs t-SNE 比較:")
    scatter_compare(
        np.column_stack([X_pca[:, 0], X_pca[:, 1]]),
        y, y,
        title_true="PCA", title_pred="t-SNE",
        width=28, height=14,
    )
    print(f"  PCA  Trustworthiness: {trust_pca:.4f}")
    print(f"  t-SNE Trustworthiness: {trust_tsne:.4f}")
    print()


# ─── フェーズ 3: 異常検知 ────────────────────────────────────
def run_anomaly_detection():
    print("=" * 66)
    print("  フェーズ 3/4 : 異常検知")
    print("=" * 66)
    print("""  【処理内容】
    正常データのみで学習し、「正常パターンから外れた点」を検出します。
    ラベルを一切使わずに学習 → ラベルで評価する半教師あり的な設定。

  【評価指標】
    Precision: 「異常と予測した点」のうち本当の異常の割合
    Recall   : 「本当の異常点」のうち正しく検出できた割合
    F1       : Precision と Recall の調和平均
""")

    ds = make_anomaly(n_normal=200, n_outliers=20, seed=42)
    X, y_true = ds.X, ds.y
    n_outliers_true = int((y_true == -1).sum())
    contamination = n_outliers_true / len(y_true)

    print(f"  データ: 正常={len(y_true)-n_outliers_true}点 / 外れ値={n_outliers_true}点")
    print(f"  汚染率: {contamination:.3f}")
    scatter_2d(X, y_true, title="異常検知データ (真ラベル: O=正常 .=外れ値)")

    detectors = [
        ("Isolation Forest", "isolation_forest",
         {"n_estimators": 100, "seed": 42}),
        ("LOF (k=20)",       "lof",
         {"n_neighbors": 20}),
    ]

    for det_name, det_key, det_kwargs in detectors:
        det = get_anomaly(det_key, **det_kwargs)
        det.fit(X)
        y_pred = det.predict(X, contamination=contamination)
        met = anomaly_metrics(y_true, y_pred)
        n_pred = int((y_pred == -1).sum())

        anomaly_summary(det_name, met, len(X), n_pred, n_outliers_true)
        scatter_compare(
            X, y_true, y_pred,
            title_true="真ラベル", title_pred=f"{det_name} 予測",
        )
        print()


# ─── フェーズ 4: VAE ─────────────────────────────────────────
def run_vae():
    print("=" * 66)
    print("  フェーズ 4/4 : 変分オートエンコーダ (VAE)")
    print("=" * 66)
    print("""  【処理内容】
    連続な潜在空間を学習し、新しいデータを生成します。
    blobs データ (2D) を入力とし、潜在次元=2 で学習します。

  【評価】
    ELBO 損失の推移: 下がるほど再構成精度と潜在空間の品質が向上
    再構成誤差: 入力と再構成の MSE
    潜在空間の可視化: 各クラスタが潜在空間で分離されるか
""")

    try:
        import torch
    except ImportError:
        print("  [スキップ] PyTorch が見つかりません。pip install torch で導入できます。")
        return

    from vae import VAE, VAETrainer

    # blobs データを [0,1] に正規化して使う
    ds = make_blobs(n_samples=400, n_clusters=3, seed=42)
    X, y = ds.X, ds.y
    X_min, X_max = X.min(axis=0), X.max(axis=0)
    X_norm = (X - X_min) / (X_max - X_min + 1e-8)

    input_dim  = X_norm.shape[1]
    hidden_dim = 32
    latent_dim = 2
    n_epochs   = 200
    beta       = 1.0

    print(f"  入力次元: {input_dim}  隠れ層: {hidden_dim}  潜在次元: {latent_dim}")
    print(f"  エポック: {n_epochs}  β={beta}  バッチサイズ: 64")
    print()

    model   = VAE(input_dim=input_dim, hidden_dim=hidden_dim, latent_dim=latent_dim)
    trainer = VAETrainer(model, lr=1e-3, batch_size=64)
    print("  学習中...")
    losses  = trainer.train(X_norm, n_epochs=n_epochs, beta=beta,
                            verbose_every=50)

    # 損失曲線
    loss_curve(losses, title="VAE ELBO 損失", width=50, height=10)

    # 再構成誤差
    import torch
    device = trainer.device
    with torch.no_grad():
        x_t     = torch.tensor(X_norm, dtype=torch.float32, device=device)
        x_recon, mu_t, _ = model(x_t)
        X_recon = x_recon.cpu().numpy()
    recon_err = reconstruction_error(X_norm, X_recon)
    print(f"\n  再構成誤差 (MSE): {recon_err:.6f}")

    # 潜在空間の可視化
    Z = model.encode_numpy(X_norm, device=device)
    scatter_2d(Z, y, title="潜在空間 z (クラスタが分離されていれば良い)")

    # 新しいデータの生成
    X_gen = model.sample(30, device=device)
    # 元のスケールに戻す
    X_gen_orig = X_gen * (X_max - X_min + 1e-8) + X_min
    print(f"\n  生成サンプル (30点):")
    print(f"    生成データの平均: {X_gen_orig.mean(axis=0)}")
    print(f"    元データの平均:   {X.mean(axis=0)}")
    print(f"    (値が近ければ潜在空間から正しく生成できている)")
    print()


# ─── メイン ─────────────────────────────────────────────────
if __name__ == "__main__":
    print()
    print("=" * 66)
    print("  教師なし学習 — 全手法比較")
    print("  (K-means / DBSCAN / Agglomerative / PCA / t-SNE /")
    print("   IsolationForest / LOF / VAE)")
    print("=" * 66)
    print()

    run_clustering()
    run_dimensionality_reduction()
    run_anomaly_detection()
    run_vae()

    print("=" * 66)
    print("  すべてのフェーズ完了")
    print("=" * 66)
