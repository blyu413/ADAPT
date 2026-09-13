"""Default assets and released models in the source-checkout layout."""

import json
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
ASSET_DIR = PROJECT_ROOT / "models" / "g1"
CHECKPOINT_DIR = PROJECT_ROOT / "logs" / "rsl_rl"


def checkpoint_path(name: str, suffix: str = ".onnx") -> Path:
    """Resolve a released policy name to its local model path."""
    manifest = json.loads((CHECKPOINT_DIR / "manifest.json").read_text())
    for entry in manifest["models"]:
        if entry["name"] == name:
            return (CHECKPOINT_DIR / entry["path"]).with_suffix(suffix)
    raise ValueError(f"Unknown released policy: {name}")
