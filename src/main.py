from __future__ import annotations

import argparse
from pathlib import Path

from common.dataset_metadata import DatasetMetadata, load_dataset_metadata
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
    parser.add_argument(
        "--dataset-metadata",
        type=Path,
        default=Path("datasets/raw/bank-full.json"),
        help="Path to a dataset metadata JSON file that defines the dataset path, features, and target column.",
    )
    return parser.parse_args()


def run_knn(work_dir: Path, dataset_meta: DatasetMetadata) -> None:
    results_dir = work_dir / "results"
    config_path = work_dir / "config" / "knn" / "default.json"

    config = KNNPipelineConfig.from_file(
        config_path,
        dataset_path=dataset_meta.dataset_path,
        results_dir=results_dir,
        feature_cols=dataset_meta.feature_cols,
        target_col=dataset_meta.target_col,
        classes=["no", "yes"],
        test_size=0.2,
        random_state=0,
        stratify=True,
    )
    pipeline = KNNPipeline(config, metadata_encoder=NumpyEncoder)
    result = pipeline.run()
    print(f"[knn] metadata: {result['metadata_path']} | model: {result['model_path']}")


def run_kmeans(work_dir: Path, dataset_meta: DatasetMetadata) -> None:
    results_dir = work_dir / "results"
    config_path = work_dir / "config" / "kmeans" / "default.json"

    config = KMeansPipelineConfig.from_file(
        config_path,
        dataset_path=dataset_meta.dataset_path,
        results_dir=results_dir,
        feature_cols=dataset_meta.feature_cols,
        target_col=dataset_meta.target_col,
        classes=["no", "yes"],
        test_size=0.2,
        random_state=0,
        stratify=True,
    )
    pipeline = KMeansPipeline(config, metadata_encoder=NumpyEncoder)
    result = pipeline.run()
    print(f"[kmeans] metadata: {result['metadata_path']} | model: {result['model_path']}")


def run_dbscan(work_dir: Path, dataset_meta: DatasetMetadata) -> None:
    results_dir = work_dir / "results"
    config_path = work_dir / "config" / "dbscan" / "v1.json"

    config = DBSCANPipelineConfig.from_file(
        config_path,
        dataset_path=dataset_meta.dataset_path,
        results_dir=results_dir,
        feature_cols=dataset_meta.feature_cols,
        target_col=dataset_meta.target_col,
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
    metadata_path = args.dataset_metadata
    if not metadata_path.is_absolute():
        metadata_path = (work_dir / metadata_path).resolve()
    dataset_meta = load_dataset_metadata(metadata_path, base_dir=work_dir)

    runners = {
        "knn": run_knn,
        "kmeans": run_kmeans,
        "dbscan": run_dbscan,
    }

    for name in args.pipelines:
        print(f"=== Running {name.upper()} ===")
        runners[name](work_dir, dataset_meta)


if __name__ == "__main__":
    main()
