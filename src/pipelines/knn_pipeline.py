from __future__ import annotations

from sklearn.neighbors import KNeighborsClassifier

from common.pipeline import ClusteringPipeline
from pipelines.config import KNNPipelineConfig


class KNNPipeline(ClusteringPipeline):
    """Concrete ClusteringPipeline implementation using scikit-learn's KNN classifier."""

    def __init__(
        self,
        config: KNNPipelineConfig,
        *,
        metadata_encoder: type | None = None,
    ) -> None:
        super().__init__(
            dataset_path=config.dataset_path,
            feature_cols=config.feature_cols,
            target_col=config.target_col,
            classes=config.classes,
            results_dir=config.results_dir,
            experiment_name="knn",
            config_name=config.config_name,
            test_size=config.test_size,
            random_state=config.random_state,
            stratify=config.stratify,
            metadata_encoder=metadata_encoder,
        )
        self.config = config

    def init_model(self):
        return KNeighborsClassifier(**self.config.knn_params)

    def experiment_params(self, model):
        return model.get_params()
        