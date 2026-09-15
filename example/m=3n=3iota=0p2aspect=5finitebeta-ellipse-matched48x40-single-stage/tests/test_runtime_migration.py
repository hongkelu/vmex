"""Check migrated runtime identity and fail-closed source verification."""
import hashlib
import importlib.util
import json
from pathlib import Path

import pytest

CASE = Path(__file__).resolve().parents[1]

def initialization():
    """Load this case without launching its runner."""
    spec = importlib.util.spec_from_file_location("case_initialization", CASE / "initialization.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module

def test_active_runtime_matches_complete_manifest():
    """Verify that every runtime module matches the active pin."""
    runtime = initialization().verify_runtime()
    manifest = json.loads((CASE / "runtime_manifest.json").read_text())
    actual = {str(p.relative_to(runtime)): hashlib.sha256(p.read_bytes()).hexdigest()
              for p in runtime.rglob("*.py")}
    assert manifest["files"] == actual

def test_modified_runtime_is_rejected(monkeypatch, tmp_path):
    """Reject modified source before any case computation."""
    import vmex
    fake = tmp_path / "__init__.py"
    fake.write_text("# modified runtime\n")
    monkeypatch.setattr(vmex, "__file__", str(fake))
    with pytest.raises(RuntimeError, match="differs from pinned commit"):
        initialization().verify_runtime()

def test_historical_manifest_is_preserved():
    """Retain the previous pin without transferring qualification."""
    migration = json.loads((CASE / "runtime_migration.json").read_text())
    archive = CASE / migration["previous_manifest"]
    assert hashlib.sha256(archive.read_bytes()).hexdigest() == migration["previous_manifest_sha256"]
    assert json.loads(archive.read_text())["commit"] == migration["previous_commit"]
    assert not migration["physical_qualification"]
