"""Repository-relative artifacts and reproducible seeds."""

import hashlib
import json
from pathlib import Path

from astra.runtime import ROOT


def stable_seed(*parts):
    value = json.dumps(parts, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
    return int.from_bytes(hashlib.sha256(value).digest()[:8], "little") % (2**63 - 1)


def artifact_path(value):
    path = Path(value)
    if path.is_absolute():
        raise ValueError("artifact paths must be relative to the repository")
    result = (ROOT / path).resolve()
    if not result.is_relative_to(ROOT):
        raise ValueError("artifact outside the repository")
    return result


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def use_workspace(path):
    """Select one explicit user training task; preserve the source model assets."""
    import sys
    global ROOT
    root = Path(path).resolve()
    if not (root / "training/config.json").is_file() or not (root / "resource.json").is_file():
        raise ValueError("training workspace needs training/config.json and resource.json")
    ROOT = root
    for name in ("astra.training.cache", "astra.training.preparation", "astra.training.train"):
        module = sys.modules.get(name)
        if module is not None:
            module.ROOT = root
    return root
