from __future__ import annotations

import argparse
from pathlib import Path

from common.metadata import NumpyEncoder
from pipelines.config import (
    DBSCANPipelineConfig,
    KMeansPipelineConfig,
    KNNPipelineConfig,
)
from pipelines.dbscan_pipeline import DBSCANPipeline
from pipelines.kmeans_pipeline import KMeansPipeline
from pipelines.knn_pipeline import KNNPipeline


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Driver script to run clustering pipelines.")
    parser.add_argument(
        "--pipelines",
        nargs="+",
        default=["knn", "kmeans", "dbscan"],
        choices=["knn", "kmeans", "dbscan"],
        help="Which pipelines to run.",
    )
    return parser.parse_args()


def feature_columns() -> list[str]:
    return [
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


def run_knn(work_dir: Path) -> None:
    dataset_path = work_dir / "datasets" / "raw" / "bank-full.csv"
    results_dir = work_dir / "results"
    config_path = work_dir / "config" / "knn" / "v1.json"

    config = KNNPipelineConfig.from_file(
        config_path,
        dataset_path=dataset_path,
        results_dir=results_dir,
        feature_cols=feature_columns(),
        target_col="y",
        classes=["no", "yes"],
        test_size=0.2,
        random_state=0,
        stratify=True,
    )
    pipeline = KNNPipeline(config, metadata_encoder=NumpyEncoder)
    result = pipeline.run()
    print(f"[knn] metadata: {result['metadata_path']} | model: {result['model_path']}")


def run_kmeans(work_dir: Path) -> None:
    dataset_path = work_dir / "datasets" / "raw" / "bank-full.csv"
    results_dir = work_dir / "results"
    config_path = work_dir / "config" / "kmeans" / "v1.json"

    config = KMeansPipelineConfig.from_file(
        config_path,
        dataset_path=dataset_path,
        results_dir=results_dir,
        feature_cols=feature_columns(),
        target_col="y",
        classes=["no", "yes"],
        test_size=0.2,
        random_state=0,
        stratify=True,
    )
    pipeline = KMeansPipeline(config, metadata_encoder=NumpyEncoder)
    result = pipeline.run()
    print(f"[kmeans] metadata: {result['metadata_path']} | model: {result['model_path']}")


def run_dbscan(work_dir: Path) -> None:
    dataset_path = work_dir / "datasets" / "raw" / "bank-full.csv"
    results_dir = work_dir / "results"
    config_path = work_dir / "config" / "dbscan" / "v1.json"

    config = DBSCANPipelineConfig.from_file(
        config_path,
        dataset_path=dataset_path,
        results_dir=results_dir,
        feature_cols=feature_columns(),
        target_col="y",
        classes=["no", "yes"],
        test_size=0.2,
        random_state=0,
        stratify=True,
    )
    pipeline = DBSCANPipeline(config, metadata_encoder=NumpyEncoder)
    result = pipeline.run()
    print(f"[dbscan] metadata: {result['metadata_path']} | model: {result['model_path']}")


def main() -> None:
    args = parse_args()
    work_dir = Path(__file__).resolve().parents[1]

    runners = {
        "knn": run_knn,
        "kmeans": run_kmeans,
        "dbscan": run_dbscan,
    }

    for name in args.pipelines:
        print(f"=== Running {name.upper()} ===")
        runners[name](work_dir)


if __name__ == "__main__":
    main()
