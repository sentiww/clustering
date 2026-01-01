from __future__ import annotations

import argparse
import json
import pickle
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.neighbors import NearestNeighbors


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compute confusion matrix from saved experiment.")
    parser.add_argument("--input", required=True, help="Path to experiment result directory.")
    return parser.parse_args()


def load_metadata(input_dir: Path) -> dict:
    with (input_dir / "metadata.json").open() as f:
        return json.load(f)


def load_model(input_dir: Path):
    with (input_dir / "model.pkl").open("rb") as f:
        return pickle.load(f)


def preprocess_dataset(metadata: dict) -> tuple[np.ndarray, np.ndarray]:
    dataset_path = Path(metadata["dataset"]["path"])
    df = pd.read_csv(dataset_path)
    feature_cols = metadata["labels"]["features"]
    target_col = metadata["labels"]["target"]

    feature_df = df[feature_cols]
    categorical_cols = feature_df.select_dtypes(include="object").columns
    encoded = pd.get_dummies(feature_df, columns=categorical_cols)
    X = encoded.to_numpy()
    y = df[target_col].to_numpy()
    return X, y


def split_data(metadata: dict, X, y):
    split_params = metadata.get("split", {}) or {}
    test_size = split_params.get("test_size", 0.2)
    random_state = split_params.get("random_state", None)
    stratify = y if split_params.get("stratify") else None

    return train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=random_state,
        stratify=stratify,
    )


def predict(metadata: dict, model, X_train, X_test, y_train) -> np.ndarray:
    name = metadata["name"]
    if name == "knn":
        return model.predict(X_test)

    if name == "kmeans":
        cluster_map = metadata.get("cluster_to_class", {})
        clusters = model.predict(X_test)
        return np.array([cluster_map[str(int(c))] for c in clusters])

    if name == "dbscan":
        cluster_map = metadata.get("cluster_to_class", {})
        fallback = metadata.get("fallback_class")
        neighbor_model = NearestNeighbors(n_neighbors=1)
        neighbor_model.fit(X_train)
        _, indices = neighbor_model.kneighbors(X_test)
        train_clusters = model.labels_
        predicted_clusters = train_clusters[indices[:, 0]]
        mapped = []
        for cluster_id in predicted_clusters:
            key = str(int(cluster_id))
            if key in cluster_map:
                mapped.append(cluster_map[key])
            elif fallback is not None:
                mapped.append(fallback)
            else:
                raise ValueError(f"Unknown cluster id {cluster_id} with no fallback class.")
        return np.array(mapped)

    raise ValueError(f"Unsupported experiment type: {name}")


def main() -> None:
    args = parse_args()
    input_dir = Path(args.input)

    metadata = load_metadata(input_dir)
    model = load_model(input_dir)

    X, y = preprocess_dataset(metadata)
    X_train, X_test, y_train, y_test = split_data(metadata, X, y)
    y_pred = predict(metadata, model, X_train, X_test, y_train)

    classes = metadata["classes"]
    cm = confusion_matrix(y_test, y_pred, labels=classes)
    print("Confusion matrix:")
    print(cm)

    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=classes)
    disp.plot(cmap="Blues")
    plt.title(f"{metadata['name'].upper()} Confusion Matrix")
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
