"""Ordinary finite-beta initialization; a vacuum guess is never an accepted root."""
import hashlib
import json
from pathlib import Path
import time


def verify_runtime():
    import vmex
    case = Path(__file__).resolve().parent
    manifest = json.loads((case / 'runtime_manifest.json').read_text())
    runtime = Path(vmex.__file__).resolve().parent
    for name, expected in manifest['files'].items():
        path = runtime / name
        if not path.is_file() or hashlib.sha256(path.read_bytes()).hexdigest() != expected:
            raise RuntimeError('VMEX runtime differs from pinned commit ' + manifest['commit'] + ': ' + name)
    return runtime

