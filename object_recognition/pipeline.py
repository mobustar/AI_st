"""
==============================================================
  pipeline.py | 物体認識パイプライン
==============================================================
Preprocessor → FeatureExtractor → Classifier の統合 API。
"""

import numpy as np
from preprocess import BasePreprocessor
from feature    import BaseFeature
from classifier import BaseClassifier


class ObjectRecognitionPipeline:
    """
    使い方:
        pipe = ObjectRecognitionPipeline(
            preprocessor = StandardScaler(),
            feature      = HOG(),
            classifier   = SoftmaxRegression(),
        )
        pipe.fit(train_images, train_labels)
        preds = pipe.predict(test_images)
    """

    def __init__(self,
                 preprocessor: BasePreprocessor,
                 feature:      BaseFeature,
                 classifier:   BaseClassifier):
        self.preprocessor = preprocessor
        self.feature      = feature
        self.classifier   = classifier

    def fit(self, images: np.ndarray, labels: np.ndarray):
        imgs = self.preprocessor.fit_transform(images)
        X    = self.feature.extract(imgs)
        self.classifier.fit(X, labels)
        return self

    def predict(self, images: np.ndarray) -> np.ndarray:
        imgs = self.preprocessor.transform(images)
        X    = self.feature.extract(imgs)
        return self.classifier.predict(X)
