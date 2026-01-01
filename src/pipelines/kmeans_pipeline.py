from __future__ import annotations

from collections import Counter
from typing import Dict

import numpy as np
from sklearn.cluster import KMeans

from common.pipeline import ClusteringPipeline
from pipelines.config import KMeansPipelineConfig


class KMeansPipeline(ClusteringPipeline):
    """ClusteringPipeline implementation for scikit-learn's KMeans."""

    def __init__(
        self,
        config: KMeansPipelineConfig,
        *,
        metadata_encoder: type | None = None,
    ) -> None:
        super().__init__(
            dataset_path=config.dataset_path,
            feature_cols=config.feature_cols,
            target_col=config.target_col,
            classes=config.classes,
            results_dir=config.results_dir,
            experiment_name="kmeans",
            config_name=config.config_name,
            test_size=config.test_size,
            random_state=config.random_state,
            stratify=config.stratify,
            metadata_encoder=metadata_encoder,
        )
        self.config = config
        self.cluster_to_class: Dict[int, str] = {}

    def init_model(self):
        return KMeans(**self.config.kmeans_params)

    def fit(self, model, X_train, y_train) -> None:
        model.fit(X_train)
        train_clusters = model.labels_
        dominant_class = Counter(y_train).most_common(1)[0][0]

        for cluster_id in range(model.n_clusters):
            cluster_mask = train_clusters == cluster_id
            cluster_labels = y_train[cluster_mask]
            if cluster_labels.size == 0:
                mapped_label = dominant_class
            else:
                mapped_label = Counter(cluster_labels).most_common(1)[0][0]
            self.cluster_to_class[cluster_id] = mapped_label

    def predict(self, model, X_test):
        return model.predict(X_test)

    def postprocess_predictions(self, predictions):
        if not self.cluster_to_class:
            raise RuntimeError("Cluster-to-class mapping missing. Did you call fit()?")
        return np.array([self.cluster_to_class[cluster_id] for cluster_id in predictions])

    def metadata_extra(self):
        return {"cluster_to_class": self.cluster_to_class}
