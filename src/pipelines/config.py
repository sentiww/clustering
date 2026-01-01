from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Sequence


@dataclass
class BasePipelineConfig:
    dataset_path: Path
    results_dir: Path
    feature_cols: Sequence[str]
    target_col: str
    classes: Sequence[str]
    config_name: str = "default"
    test_size: float = 0.2
    random_state: int | None = 0
    stratify: bool = True


def _load_json(path: str | Path) -> Dict[str, Any]:
    with Path(path).open() as f:
        return json.load(f)


@dataclass
class KNNPipelineConfig(BasePipelineConfig):
    knn_params: Dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_file(cls, path: str | Path, **kwargs) -> "KNNPipelineConfig":
        config_data = _load_json(path)
        kwargs.setdefault("config_name", Path(path).stem)
        return cls(knn_params=config_data, **kwargs)


@dataclass
class KMeansPipelineConfig(BasePipelineConfig):
    kmeans_params: Dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_file(cls, path: str | Path, **kwargs) -> "KMeansPipelineConfig":
        config_data = _load_json(path)
        kwargs.setdefault("config_name", Path(path).stem)
        return cls(kmeans_params=config_data, **kwargs)


@dataclass
class DBSCANPipelineConfig(BasePipelineConfig):
    dbscan_params: Dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_file(cls, path: str | Path, **kwargs) -> "DBSCANPipelineConfig":
        config_data = _load_json(path)
        kwargs.setdefault("config_name", Path(path).stem)
        return cls(dbscan_params=config_data, **kwargs)
