import json
from pathlib import Path
from typing import Any, Dict


class MetaLoadError(Exception):
    pass


def load_meta(meta_path: str) -> Dict[str, Any]:
    path = Path(meta_path)
    if not path.exists():
        raise MetaLoadError(f"meta file not found: {meta_path}")
    try:
        return json.loads(path.read_text())
    except json.JSONDecodeError as exc:
        raise MetaLoadError(f"invalid meta json: {meta_path}") from exc
