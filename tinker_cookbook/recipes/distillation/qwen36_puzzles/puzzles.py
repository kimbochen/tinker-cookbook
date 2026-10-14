"""Read the packaged puzzle datasets and check their recorded checksums.

The checkout stores each dataset as `data/<name>.gz` next to `data/manifest.json`.
"""

import gzip
import hashlib
import json
from pathlib import Path
from typing import TypedDict

from tinker_cookbook import renderers


class PuzzleRow(TypedDict):
    prompt: list[renderers.Message]
    label: str
    metadata: dict[str, str]
    _index: int


class ManifestEntry(TypedDict):
    bytes: int
    rows: int
    sha256: str


DATA_DIR = Path(__file__).resolve().parent / "data"


def manifest(data_dir: Path = DATA_DIR) -> dict[str, ManifestEntry]:
    return json.loads((Path(data_dir) / "manifest.json").read_text())


def packaged_bytes(name: str, data_dir: Path = DATA_DIR) -> bytes:
    """Uncompressed bytes of one packaged dataset, after checking its size and SHA-256."""
    record = manifest(data_dir)[name]
    data = gzip.decompress((Path(data_dir) / f"{name}.gz").read_bytes())
    if len(data) != record["bytes"] or hashlib.sha256(data).hexdigest() != record["sha256"]:
        raise ValueError(f"Dataset checksum mismatch: {name}")
    return data


def load_split(domain: str, split: str, data_dir: Path = DATA_DIR) -> list[PuzzleRow]:
    """Rows of one packaged split. Each row gains `_index`, its position in the file."""
    name = f"{domain}-{split}.jsonl"
    rows: list[PuzzleRow] = [
        json.loads(line) for line in packaged_bytes(name, data_dir).splitlines()
    ]
    if len(rows) != manifest(data_dir)[name]["rows"]:
        raise ValueError(f"Dataset row count mismatch: {name}")
    for index, row in enumerate(rows):
        if row["metadata"]["domain"] != domain:
            raise ValueError(f"{name} row {index} belongs to {row['metadata']['domain']}")
        row["_index"] = index
    return rows
