import importlib.util  # noqa: F401
import sys
from pathlib import Path
import pytest

sys.path.insert(0, str(Path(__file__).parent))
import workflow as w
import campaign as c


def status():
    return dict(
        decision="complete",
        last_recorded_step=10,
        owned_run_processes=[],
        exit={"returncode": 0},
        summary={"status": "step_budget_reached", "target_step": 10, "final_step": 10},
        numerical_failure_events=[],
        missing_periodic_snapshots=[],
        metrics={"checkpoint_sha256": "verified"},
    )


def test_unpinned_source_cannot_launch(tmp_path, monkeypatch):
    monkeypatch.setattr(w, "ROOT", tmp_path)
    monkeypatch.setattr(c, "tick", lambda *a: pytest.fail("unconfirmed code cannot launch"))
    assert w.tick(True)["decision"] == "main_revision_pending"


def test_only_verified_validation_exit_is_restartable_for_production():
    s = status()
    s.update(
        segment_count=1,
        active_seconds=100,
        wall_seconds=100,
        nonprogress_interruptions=0,
        disk_free_bytes=10000,
        last_event={},
    )
    p = dict(
        target_step=3000,
        max_active_seconds=86400,
        max_wall_seconds=172800,
        max_segments=16,
        max_nonprogress_interruptions=3,
        min_free_bytes=100,
    )
    assert c.decision(s, p) == "unexpected_exit_needs_review"
    p["validation_checkpoint_sha256"] = "wrong"
    assert c.decision(s, p) == "unexpected_exit_needs_review"
    p["validation_checkpoint_sha256"] = "verified"
    assert c.decision(s, p) == "restart_ready"
    s["numerical_failure_events"] = [{"error": "failed"}]
    assert c.decision(s, p) == "numerical_failure_needs_review"


@pytest.mark.parametrize("bad", ["wrong_count", "failed_exit", "live_process", "numerical_failure"])
def test_invalid_validation_cannot_promote(bad):
    s = status()
    if bad == "wrong_count":
        s["last_recorded_step"] = 9
    if bad == "failed_exit":
        s["exit"]["returncode"] = -9
    if bad == "live_process":
        s["owned_run_processes"] = ["runner"]
    if bad == "numerical_failure":
        s["numerical_failure_events"] = [{"error": "failed"}]
    with pytest.raises(AssertionError):
        w.validate_ten(s)


def test_no_production_transition_in_read_only_tick(tmp_path, monkeypatch):
    monkeypatch.setattr(w, "ROOT", tmp_path)
    c.write(tmp_path / "policy.json", {"target_step": 10})
    monkeypatch.setattr(w, "readiness", lambda: None)
    calls = []
    monkeypatch.setattr(c, "tick", lambda restart: calls.append(restart) or status())
    monkeypatch.setattr(w, "validate_ten", lambda s: {"checkpoint_sha256": "verified"})
    result = w.tick(False)  # noqa: F841
    assert calls == [False] and c.read(tmp_path / "policy.json")["target_step"] == 10
    assert not (tmp_path / "validation10_passed.json").exists()


def test_verified_ten_steps_enable_exactly_one_production_launch(tmp_path, monkeypatch):
    monkeypatch.setattr(w, "ROOT", tmp_path)
    c.write(tmp_path / "policy.json", {"target_step": 10})
    monkeypatch.setattr(w, "readiness", lambda: None)
    calls = []

    def tick(restart):
        calls.append(restart)
        return status() if len(calls) == 1 else {"decision": "initializing", "action": "restarted"}

    monkeypatch.setattr(c, "tick", tick)
    monkeypatch.setattr(w, "validate_ten", lambda s: {"checkpoint_sha256": "verified"})
    result = w.tick(True)
    assert calls == [True, True] and result["phase"] == "production3000"
    assert c.read(tmp_path / "policy.json")["target_step"] == 3000
    assert c.read(tmp_path / "policy.json")["validation_checkpoint_sha256"] == "verified"
