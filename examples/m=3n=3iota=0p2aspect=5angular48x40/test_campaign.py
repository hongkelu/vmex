import copy  # noqa: F401
import fcntl
import importlib.util
import json  # noqa: F401
import time  # noqa: F401
from pathlib import Path
import pytest

spec = importlib.util.spec_from_file_location("campaign", Path(__file__).with_name("campaign.py"))
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)
POLICY = dict(
    target_step=3000,
    max_active_seconds=86400,
    max_wall_seconds=172800,
    max_segments=16,
    max_nonprogress_interruptions=3,
    min_free_bytes=100,
)


def interrupted():
    return dict(
        owned_run_processes=[],
        last_recorded_step=1144,
        summary=None,
        exit={"returncode": -9},
        segment_count=1,
        active_seconds=7200,
        wall_seconds=8000,
        nonprogress_interruptions=0,
        disk_free_bytes=1000,
        numerical_failure_events=[],
        last_event={},
    )


def test_sigkill_with_progress_is_restartable_and_preserves_total_target():
    s = interrupted()
    assert m.decision(s, POLICY) == "restart_ready"
    assert POLICY["target_step"] == 3000


@pytest.mark.parametrize("status", ["converged", "stagnated"])
def test_normal_scientific_stop_is_not_restarted(status):
    s = interrupted()
    s.update(summary={"status": status}, exit={"returncode": 0})
    assert m.decision(s, POLICY) == status


@pytest.mark.parametrize("error", ["VmecError", "ValueError", "RuntimeError"])
def test_signal_after_numerical_failure_is_not_restarted(error):
    s = interrupted()
    s["numerical_failure_events"] = [{"error_type": error}]
    assert m.decision(s, POLICY) == "numerical_failure_needs_review"


def test_live_process_prevents_duplicate_even_if_stale_exit_exists():
    s = interrupted()
    s["owned_run_processes"] = ["runner"]
    assert m.decision(s, POLICY) == "running"


@pytest.mark.parametrize(
    "field,value,expected",
    [
        ("segment_count", 16, "restart_budget_reached"),
        ("nonprogress_interruptions", 3, "no_progress_needs_review"),
        ("active_seconds", 86400, "campaign_time_budget_reached"),
        ("wall_seconds", 172800, "campaign_time_budget_reached"),
        ("disk_free_bytes", 99, "insufficient_disk_space"),
    ],
)
def test_restart_circuit_breakers(field, value, expected):
    s = interrupted()
    s[field] = value
    assert m.decision(s, POLICY) == expected


def test_no_restart_at_3000_even_if_terminal_summary_missing():
    s = interrupted()
    s["last_recorded_step"] = 3000
    assert m.decision(s, POLICY) == "target_reached_needs_terminal_review"
    s["summary"] = {"final_step": 3000, "status": "step_budget_reached"}
    s["exit"] = {"returncode": 0}
    assert m.decision(s, POLICY) == "complete"


def row(step, elapsed):
    return dict(
        step=step,
        elapsed_s=elapsed,
        mean_iota=0.2,
        aspect_ratio=5.0,
        b0_T=-0.175,
        qs_error_raw=0.1,
        qs_objective_normalized=0.5,
        checkpoint_path=f"checkpoint_step_{step:04d}.npz",
    )


def test_join_two_restarts_preserves_contiguous_history_and_one_anchor(tmp_path):
    h = [row(0, 0), row(1, 10)]
    m.append_rows(h, [row(1, 2), row(2, 20)], tmp_path / "a")
    m.append_rows(h, [row(2, 3), row(3, 30)], tmp_path / "b")
    assert [x["step"] for x in h] == [0, 1, 2, 3]
    assert h[-1]["elapsed_s"] == 60 and h[-1]["checkpoint_path"] == str(tmp_path / "b/checkpoint_step_0003.npz")


def test_mismatched_anchor_rejected(tmp_path):
    r = row(1, 0)
    r["mean_iota"] = 0.201
    with pytest.raises(AssertionError):
        m.append_rows([row(0, 0), row(1, 10)], [r], tmp_path)


def test_simultaneous_controller_cannot_launch(tmp_path, monkeypatch):
    monkeypatch.setattr(m, "ROOT", tmp_path)
    with (tmp_path / "controller.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        assert m.tick(True)["action"] == "controller_busy"


def test_gpu_occupied_never_starts_process(tmp_path, monkeypatch):
    monkeypatch.setattr(m, "ROOT", tmp_path)
    monkeypatch.setattr(m, "verify_source", lambda: None)
    monkeypatch.setattr(m, "processes", lambda: [])
    monkeypatch.setattr(m.subprocess, "check_output", lambda *a, **k: m.GPU)
    assert m.launch({"segments": []}, interrupted(), POLICY) == "waiting_for_gpu"
    assert not (tmp_path / "segments").exists()


def test_two_ticks_launch_only_once(tmp_path, monkeypatch):
    monkeypatch.setattr(m, "ROOT", tmp_path)
    m.write(tmp_path / "registry.json", {"segments": []})
    m.write(tmp_path / "policy.json", POLICY)

    def collect(reg):
        s = interrupted()
        s["segment_count"] = len(reg["segments"])
        if reg["segments"]:
            s["owned_run_processes"] = ["runner"]
        return s

    calls = []

    def launch(reg, s, p):
        calls.append(s["last_recorded_step"])
        reg["segments"].append({"directory": "segments/segment_0001"})
        return "started"

    monkeypatch.setattr(m, "collect", collect)
    monkeypatch.setattr(m, "launch", launch)
    assert m.tick(True)["action"] == "started"
    assert m.tick(True)["decision"] == "running"
    assert calls == [1144]
