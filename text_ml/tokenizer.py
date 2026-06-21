"""
==============================================================
  tokenizer.py | トークナイザ (戦略パターン)
==============================================================
テキストを「単語」や「サブワード」の列に分解する処理。
日本語は単語境界が無いため、単純な空白分割は機能しない。

選択アルゴリズム:
  1. WhitespaceTokenizer  - 英語など単語境界が明示的な言語向け
  2. CharNGramTokenizer   - 言語依存が少なく頑健 (日本語に有効)
  3. RegexTokenizer       - 正規表現で柔軟に語を抽出
  4. CharTokenizer        - 1 文字ずつ (語彙圧縮、最も簡素)

精度の傾向 (日本語短文の場合):
  CharNGram (n=2~3) > Regex > Char > Whitespace
"""

from abc import ABC, abstractmethod
from typing import List
import re
import unicodedata


class BaseTokenizer(ABC):
    """全トークナイザの共通インタフェース"""

    @abstractmethod
    def tokenize(self, text: str) -> List[str]:
        ...

    def _normalize(self, text: str) -> str:
        """NFKC 正規化 + 小文字化 (全半角ゆれ・大文字小文字を吸収)"""
        text = unicodedata.normalize("NFKC", text)
        return text.lower()


# ─── 1. 空白区切り ──────────────────────────────────────────
class WhitespaceTokenizer(BaseTokenizer):
    """空白で区切るだけ。英語向け、日本語ではほぼ機能しない。"""

    def tokenize(self, text: str) -> List[str]:
        return self._normalize(text).split()


# ─── 2. 文字 N-gram ─────────────────────────────────────────
class CharNGramTokenizer(BaseTokenizer):
    """
    文字 N-gram に分解する。
    例: "美味しい" (n=2) → ["美味", "味し", "しい"]

    日本語のように形態素解析器が無い環境でも機能する強力な手法。
    n=2~3 が経験的に良いことが多い。
    複数の n を組み合わせると更に精度が向上する。
    """

    def __init__(self, ns=(2, 3)):
        # ns: 抽出する n の集合 (例: (2,3) なら 2-gram と 3-gram の両方)
        self.ns = tuple(ns)

    def tokenize(self, text: str) -> List[str]:
        text = self._normalize(text)
        # 空白・記号は削除して連続文字列にする (短文向けの簡易処理)
        text = re.sub(r"\s+", "", text)
        tokens = []
        for n in self.ns:
            if len(text) < n:
                continue
            for i in range(len(text) - n + 1):
                tokens.append(text[i : i + n])
        return tokens


# ─── 3. 正規表現ベース ──────────────────────────────────────
class RegexTokenizer(BaseTokenizer):
    """
    正規表現で「語らしきもの」を抽出する。
    日本語の場合、文字種の切り替わり (漢字↔ひらがな↔カタカナ) を
    境界として擬似的に分割する。
    """

    # 漢字 / ひらがな / カタカナ / 英数字 をそれぞれ連続したまとまりとして抽出
    _PATTERN = re.compile(
        r"[\u4E00-\u9FFF]+"     # 漢字
        r"|[\u3040-\u309F]+"    # ひらがな
        r"|[\u30A0-\u30FF]+"    # カタカナ
        r"|[a-zA-Z0-9]+"        # 英数字
    )

    def tokenize(self, text: str) -> List[str]:
        text = self._normalize(text)
        return self._PATTERN.findall(text)


# ─── 4. 文字単位 ────────────────────────────────────────────
class CharTokenizer(BaseTokenizer):
    """
    1 文字ずつトークン化する。
    語彙が小さくなりメモリ効率が良いが、単語の意味は捉えにくい。
    深層学習モデル (RNN, Transformer) のベースラインとして利用。
    """

    def tokenize(self, text: str) -> List[str]:
        text = self._normalize(text)
        return [c for c in text if not c.isspace()]


# ─── ファクトリ関数 (Strategy 切り替え) ─────────────────────
def get_tokenizer(name: str = "char_ngram", **kwargs) -> BaseTokenizer:
    """
    名前で Tokenizer を取得する
        name in {"whitespace", "char_ngram", "regex", "char"}
    """
    table = {
        "whitespace":  WhitespaceTokenizer,
        "char_ngram":  CharNGramTokenizer,
        "regex":       RegexTokenizer,
        "char":        CharTokenizer,
    }
    if name not in table:
        raise ValueError(f"unknown tokenizer: {name}")
    return table[name](**kwargs)
