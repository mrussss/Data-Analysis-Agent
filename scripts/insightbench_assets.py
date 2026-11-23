"""Immutable public input provenance shared by the runner and the scorer."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from urllib.request import urlopen

PROJECT = Path(__file__).resolve().parents[1]
CACHE = PROJECT / "data" / "downloads" / "insightbench"
MANIFEST_PATH = PROJECT / "evaluation" / "insightbench_v1_manifest.json"
MANIFEST = json.loads(MANIFEST_PATH.read_text())


def asset(name: str, *, download: bool = False) -> Path:
    expected = MANIFEST["files"][name]
    path = CACHE / name
    if not path.exists() and download:
        url = f"https://raw.githubusercontent.com/ServiceNow/insight-bench/{MANIFEST['revision']}/{name}"
        with urlopen(url, timeout=30) as response:
            content = response.read(2_000_000)
        if hashlib.sha256(content).hexdigest() != expected:
            raise ValueError(f"Download hash mismatch for {name}")
        path.parent.mkdir(parents=True, exist_ok=True)
        try:
            with path.open("xb") as output:
                output.write(content)
        except FileExistsError:
            pass
    if not path.is_file():
        raise ValueError(f"Missing {name}; run with --download")
    if hashlib.sha256(path.read_bytes()).hexdigest() != expected:
        raise ValueError(f"Cached file hash mismatch for {name}")
    return path
