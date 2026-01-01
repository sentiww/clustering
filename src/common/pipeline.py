from __future__ import annotations

import pickle
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Dict, Sequence

import pandas as pd
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split

from common.metadata import Metadata


class ClusteringPipeline(ABC):
    """
    Generic clustering pipeline that orchestrates dataset loading, preprocessing,
    splitting, model training, evaluation, and metadata persistence.
    Subclasses override the model-specific hooks (e.g., init_model, postprocessing).
    """

    def __init__(
        self,
        *,
        dataset_path: str | Path,
        feature_cols: Sequence[str],
        target_col: str,
        classes: Sequence[str],
        results_dir: str | Path,
        experiment_name: str,
        config_name: str = "default",
        test_size: float = 0.2,
        random_state: int | None = 0,
        stratify: bool = True,
        metadata_encoder: type | None = None,
    ) -> None:
        self.dataset_path = Path(dataset_path)
        self.feature_cols = list(feature_cols)
        self.target_col = target_col
        self.classes = list(classes)
        self.results_root = Path(results_dir)
        self.experiment_name = experiment_name
        self.config_name = config_name
        self.test_size = test_size
        self.random_state = random_state
        self.stratify = stratify
        self.metadata_encoder = metadata_encoder

        self.results_path = self.results_root / self.experiment_name / self.config_name
        self.results_path.mkdir(parents=True, exist_ok=True)

        # Placeholders shared with subclasses.
        self.X_full = None
        self.y_full = None
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None

    def run(self) -> Dict[str, Any]:
        """Execute the full pipeline and return evaluation artifacts."""
        df = self.load_data()
        X, y = self.preprocess(df)

        self.X_full, self.y_full = X, y

        X_train, X_test, y_train, y_test = self.split_data(X, y)
        self.X_train, self.X_test = X_train, X_test
        self.y_train, self.y_test = y_train, y_test

        model = self.init_model()
        self.fit(model, X_train, y_train)

        raw_predictions = self.predict(model, X_test)
        y_pred = self.postprocess_predictions(raw_predictions)

        evaluation = self.evaluate(y_test, y_pred)
        metadata = self.build_metadata(model)
        self.save_artifacts(model, metadata)

        return {
            "evaluation": evaluation,
            "metadata_path": self.results_path / "metadata.json",
            "model_path": self.results_path / "model.pkl",
        }

    def load_data(self) -> pd.DataFrame:
        """Load the dataset from disk."""
        return pd.read_csv(self.dataset_path)

    def preprocess(self, df: pd.DataFrame) -> tuple:
        """Select feature columns, one-hot encode categoricals, and return numpy arrays."""
        feature_frame = df[self.feature_cols]
        categorical_cols = feature_frame.select_dtypes(include="object").columns
        encoded = pd.get_dummies(feature_frame, columns=categorical_cols)
        X = encoded.to_numpy()
        y = df[self.target_col].to_numpy()
        return X, y

    def split_data(self, X, y) -> tuple:
        """Split data into train/test sets."""
        stratify_labels = y if self.stratify else None
        return train_test_split(
            X,
            y,
            test_size=self.test_size,
            stratify=stratify_labels,
            random_state=self.random_state,
        )

    @abstractmethod
    def init_model(self):
        """Construct and return the estimator."""

    def fit(self, model, X_train, y_train) -> None:
        """Default fit implementation; override for unsupervised pipelines."""
        model.fit(X_train, y_train)

    def predict(self, model, X_test):
        """Default prediction step."""
        return model.predict(X_test)

    def postprocess_predictions(self, predictions):
        """Hook for subclasses that need to map cluster IDs to labels."""
        return predictions

    def evaluate(self, y_test, y_pred):
        """Compute confusion matrix for quick inspection."""
        cm = confusion_matrix(y_test, y_pred, labels=self.classes)
        print("Confusion matrix:")
        print(cm)

        return {"confusion_matrix": cm}

    def build_metadata(self, model) -> Metadata:
        """Package metadata for downstream utilities."""
        return Metadata(
            name=self.experiment_name,
            params=self.experiment_params(model),
            dataset_path=str(self.dataset_path),
            feature_cols=self.feature_cols,
            target_col=self.target_col,
            classes=self.classes,
            split_params={
                "test_size": self.test_size,
                "random_state": self.random_state,
                "stratify": self.stratify,
            },
            extra=self.metadata_extra(),
        )

    def experiment_params(self, model) -> Dict[str, Any]:
        """Structured parameters describing the experiment/model."""
        return getattr(model, "get_params", lambda: {})()

    def metadata_extra(self) -> Dict[str, Any]:
        """Override to add custom metadata fields."""
        return {}

    def save_artifacts(self, model, metadata: Metadata) -> None:
        """Persist the trained model and metadata JSON."""
        model_path = self.results_path / "model.pkl"
        with model_path.open("wb") as f:
            pickle.dump(model, f)
        metadata.save(self.results_path, encoder=self.metadata_encoder)
