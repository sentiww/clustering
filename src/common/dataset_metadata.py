from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence


@dataclass(frozen=True)
class DatasetMetadata:
    """Lightweight representation of dataset metadata used to configure pipelines."""

    name: str
    dataset_path: Path
    feature_cols: Sequence[str]
    target_col: str
    raw: dict[str, Any]


def _resolve_path(path: str | Path, *, base_dir: Path | None, default_dir: Path) -> Path:
    resolved = Path(path)
    if resolved.is_absolute():
        return resolved
    if base_dir is not None:
        return (base_dir / resolved).resolve()
    return (default_dir / resolved).resolve()


def _extract_target(target_config: Any) -> str:
    if isinstance(target_config, str):
        return target_config
    if isinstance(target_config, dict):
        for key in ("name", "column", "id"):
            if key in target_config:
                return target_config[key]
    raise ValueError("Dataset metadata missing 'target' definition")


def load_dataset_metadata(path: str | Path, *, base_dir: Path | None = None) -> DatasetMetadata:
    """
    Load dataset metadata consisting of dataset path, feature columns, and target column.

    Parameters
    ----------
    path:
        Path to the metadata JSON file.
    base_dir:
        Optional base directory used to resolve relative dataset paths (defaults to the
        metadata file's parent directory).
    """

    metadata_path = Path(path)
    metadata_path = metadata_path if metadata_path.is_absolute() else metadata_path.resolve()
    with metadata_path.open() as f:
        metadata = json.load(f)

    dataset_name = metadata.get("dataset", metadata_path.stem)
    dataset_file = metadata.get("file")
    if not dataset_file:
        raise ValueError(f"Dataset metadata '{metadata_path}' missing 'file' entry")

    dataset_path = _resolve_path(
        dataset_file,
        base_dir=base_dir,
        default_dir=metadata_path.parent,
    )
    features = metadata.get("features")
    if not features:
        raise ValueError(f"Dataset metadata '{metadata_path}' missing 'features' list")
    target = _extract_target(metadata.get("target"))

    return DatasetMetadata(
        name=dataset_name,
        dataset_path=dataset_path,
        feature_cols=list(features),
        target_col=target,
        raw=metadata,
    )
