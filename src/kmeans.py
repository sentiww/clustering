from __future__ import annotations

from pathlib import Path

from common.metadata import NumpyEncoder
from pipelines.config import KMeansPipelineConfig
from pipelines.kmeans_pipeline import KMeansPipeline


def main() -> None:
    work_dir = Path(__file__).resolve().parents[1]
    dataset_path = work_dir / "datasets" / "raw" / "bank-full.csv"
    results_dir = work_dir / "results"

    feature_cols = [
        "age",
        "job",
        "education",
        "default",
        "balance",
        "housing",
        "loan",
        "contact",
        "day",
        "month",
        "duration",
        "campaign",
        "pdays",
        "previous",
        "poutcome",
    ]

    config = KMeansPipelineConfig(
        dataset_path=dataset_path,
        results_dir=results_dir,
        feature_cols=feature_cols,
        target_col="y",
        classes=["no", "yes"],
        config_name="manual",
        test_size=0.2,
        random_state=0,
        stratify=True,
        kmeans_params={
            "n_clusters": 1000,
            "init": "k-means++",
            "n_init": "auto",
            "max_iter": 1000,
            "tol": 0.0001,
            "verbose": 0,
            "random_state": None,
            "copy_x": True,
            "algorithm": "lloyd",
        },
    )
    pipeline = KMeansPipeline(config, metadata_encoder=NumpyEncoder)
    result = pipeline.run()
    print(f"Metadata saved to: {result['metadata_path']}")
    print(f"Model saved to: {result['model_path']}")


if __name__ == "__main__":
    main()
