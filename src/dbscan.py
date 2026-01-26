from __future__ import annotations

import argparse
from pathlib import Path

from common.dataset_metadata import load_dataset_metadata
from common.metadata import NumpyEncoder
from pipelines.config import DBSCANPipelineConfig
from pipelines.dbscan_pipeline import DBSCANPipeline


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the DBSCAN clustering pipeline.")
    parser.add_argument(
        "--dataset-metadata",
        type=Path,
        default=Path("datasets/raw/bank-full.json"),
        help="Path to the dataset metadata JSON file.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    work_dir = Path(__file__).resolve().parents[1]
    metadata_path = args.dataset_metadata
    if not metadata_path.is_absolute():
        metadata_path = (work_dir / metadata_path).resolve()
    dataset_meta = load_dataset_metadata(metadata_path, base_dir=work_dir)

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
    print(f"Metadata saved to: {result['metadata_path']}")
    print(f"Model saved to: {result['model_path']}")


if __name__ == "__main__":
    main()
