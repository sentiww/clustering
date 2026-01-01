from __future__ import annotations

from collections import Counter
from typing import Dict

import numpy as np
from sklearn.cluster import DBSCAN
from sklearn.neighbors import NearestNeighbors

from common.pipeline import ClusteringPipeline
from pipelines.config import DBSCANPipelineConfig


class DBSCANPipeline(ClusteringPipeline):
    """ClusteringPipeline implementation for DBSCAN with nearest-neighbor labeling."""

    def __init__(
        self,
        config: DBSCANPipelineConfig,
        *,
        metadata_encoder: type | None = None,
    ) -> None:
        super().__init__(
            dataset_path=config.dataset_path,
            feature_cols=config.feature_cols,
            target_col=config.target_col,
            classes=config.classes,
            results_dir=config.results_dir,
            experiment_name="dbscan",
            config_name=config.config_name,
            test_size=config.test_size,
            random_state=config.random_state,
            stratify=config.stratify,
            metadata_encoder=metadata_encoder,
        )
        self.config = config
        self.cluster_to_class: Dict[int, str] = {}
        self._train_cluster_labels = None
        self._neighbor_model: NearestNeighbors | None = None
        self._fallback_class: str | None = None

    def init_model(self):
        return DBSCAN(**self.config.dbscan_params)

    def fit(self, model, X_train, y_train) -> None:
        model.fit(X_train)
        train_clusters = model.labels_
        self._train_cluster_labels = train_clusters

        self._neighbor_model = NearestNeighbors(n_neighbors=1)
        self._neighbor_model.fit(X_train)

        fallback = Counter(y_train).most_common(1)[0][0]
        self._fallback_class = fallback

        unique_clusters = np.unique(train_clusters)
        for cluster_id in unique_clusters:
            cluster_mask = train_clusters == cluster_id
            cluster_labels = y_train[cluster_mask]
            if cluster_labels.size == 0:
                mapped_label = fallback
            else:
                mapped_label = Counter(cluster_labels).most_common(1)[0][0]
            self.cluster_to_class[int(cluster_id)] = mapped_label

    def predict(self, model, X_test):
        if self._neighbor_model is None or self._train_cluster_labels is None:
            raise RuntimeError("DBSCAN pipeline not fitted before predict.")

        _, indices = self._neighbor_model.kneighbors(X_test)
        nearest_clusters = self._train_cluster_labels[indices[:, 0]]
        return nearest_clusters

    def postprocess_predictions(self, predictions):
        if self._fallback_class is None:
            raise RuntimeError("Fallback class unresolved. Did fit() run?")
        mapped = []
        for cluster_id in predictions:
            mapped.append(self.cluster_to_class.get(int(cluster_id), self._fallback_class))
        return np.array(mapped)

    def metadata_extra(self):
        return {
            "cluster_to_class": self.cluster_to_class,
            "fallback_class": self._fallback_class,
        }
        
