"""
==============================================================
  pipeline.py | テキスト分類パイプライン
==============================================================
Tokenizer → Vectorizer → Classifier を統合する高レベルAPI。

設計:
  ユーザは TextClassificationPipeline に各戦略を渡すだけで
  fit / predict が利用できる。
"""

from typing import List
import numpy as np

from tokenizer  import BaseTokenizer
from vectorizer import BaseVectorizer
from classifier import BaseClassifier


class TextClassificationPipeline:
    """
    使い方:
        pipe = TextClassificationPipeline(
            tokenizer=CharNGramTokenizer(ns=(2,3)),
            vectorizer=TfIdfVectorizer(min_df=1),
            classifier=LinearSVM(reg=1e-3),
        )
        pipe.fit(train_texts, train_labels)
        preds = pipe.predict(test_texts)
    """

    def __init__(self,
                 tokenizer:  BaseTokenizer,
                 vectorizer: BaseVectorizer,
                 classifier: BaseClassifier):
        self.tokenizer  = tokenizer
        self.vectorizer = vectorizer
        self.classifier = classifier

    def _tokenize_all(self, texts: List[str]) -> List[List[str]]:
        return [self.tokenizer.tokenize(t) for t in texts]

    def fit(self, texts: List[str], labels) -> "TextClassificationPipeline":
        token_lists = self._tokenize_all(texts)
        X           = self.vectorizer.fit_transform(token_lists)
        self.classifier.fit(X, np.asarray(labels))
        return self

    def predict(self, texts: List[str]) -> np.ndarray:
        token_lists = self._tokenize_all(texts)
        X           = self.vectorizer.transform(token_lists)
        return self.classifier.predict(X)
