import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Sequence

import numpy as np


@dataclass
class Metadata:
    """
    Common helper to encapsulate experiment metadata and persist it to disk.
    """

    name: str
    params: Dict[str, Any]
    dataset_path: str
    feature_cols: Sequence[str]
    target_col: str
    classes: Sequence[str]
    split_params: Dict[str, Any] | None = None
    extra: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        metadata = {
            "name": self.name,
            "params": self.params,
            "dataset": {
                "path": self.dataset_path,
            },
            "labels": {
                "features": list(self.feature_cols),
                "target": self.target_col,
            },
            "classes": list(self.classes),
        }
        if self.split_params:
            metadata["split"] = self.split_params
        metadata.update(self.extra)
        return metadata

    def save(self, output_dir: str | Path, encoder: type[json.JSONEncoder] | None = None) -> None:
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        metadata_path = output_path / "metadata.json"
        with metadata_path.open("w") as file:
            json.dump(self.to_dict(), file, cls=encoder)


class NumpyEncoder(json.JSONEncoder):
    """JSON encoder that safely serializes numpy data structures."""

    def default(self, obj):  # type: ignore[override]
        if isinstance(obj, (np.integer, np.bool_)):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super().default(obj)
