"""Guard the refined campaign's ten-step validation and 3000-total-step handoff."""

from pathlib import Path
import argparse
import fcntl
import hashlib  # noqa: F401
import json
import time  # noqa: F401
import campaign as c

ROOT = Path(__file__).resolve().parent


def readiness():
    pin = c.read(ROOT / "source_ready.json")
    env = c.read(ROOT / "setup/install_complete.json")
    error = c.read(ROOT / "setup/install_error.json")
    if not pin:
        return dict(
            decision="main_revision_pending",
            action="none",
            environment_ready=bool(env),
            environment_error=error,
            validation_steps=10,
            target_step=3000,
        )
    if error and not env:
        return dict(decision="environment_setup_failed", action="none", error=error)
    if not env:
        return dict(decision="environment_setup_pending", action="none")
    assert pin["source_manifest_sha256"] == c.sha(ROOT / "source_manifest.json"), "pinned source changed"
    assert pin["edge_residual_agreement_verified"] is True, "new-main edge agreement not verified"
    assert pin["main_commit"] == c.read(ROOT / "policy.json")["main_commit"], "main provenance mismatch"
    c.verify_source()
    if not (ROOT / "registry.json").exists():
        return dict(decision="validated_initial_checkpoint_pending", action="none")
    return None


def validate_ten(status):
    assert status["decision"] == "complete" and status["last_recorded_step"] == 10
    assert not status["owned_run_processes"] and status["exit"]["returncode"] == 0
    assert status["summary"]["status"] == "step_budget_reached" and status["summary"]["target_step"] == 10
    assert status["summary"]["final_step"] == 10 and not status["numerical_failure_events"]
    assert not status["missing_periodic_snapshots"]
    history = c.lines(ROOT / "campaign/metrics.jsonl")
    assert [row["step"] for row in history] == list(range(11))
    for row in history:
        c.authenticate_checkpoint(row)
    accepted = []
    for seg in c.read(ROOT / "registry.json")["segments"]:
        events = c.lines(ROOT / seg["directory"] / "output/progress.jsonl")
        for e in events:
            if e["phase"] == "failure" and e.get("error_type") != "TimeoutError":
                raise AssertionError("numerical failure in validation")
            if e["phase"] == "adjoint":
                for row in e["rows"]:
                    assert row["accepted"] and row["relative_residual"] <= 2e-5
            if e["phase"] == "promoted":
                assert e["maximum_coil_bound_m"] <= 0.001 * (1 + 1e-12)
                assert e["maximum_current_fraction"] <= 0.01 * (1 + 1e-12)
                assert e["acceptance"]["accepted"] is True
                accepted.append(e["absolute_step"])
    assert accepted == list(range(1, 11)), "ten accepted promotions required"
    return dict(
        verified_at_utc=c.now(),
        accepted_steps=10,
        checkpoint_sha256=status["metrics"]["checkpoint_sha256"],
        source_manifest_sha256=c.sha(ROOT / "source_manifest.json"),
        all_gates_passed=True,
    )


def tick(restart=False):
    with (ROOT / "workflow.lock").open("a") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            return dict(decision="workflow_busy", action="none")
        try:
            waiting = readiness()
            if waiting:
                result = waiting
            else:
                result = c.tick(restart)
                policy = c.read(ROOT / "policy.json")
                if policy["target_step"] == 10 and result.get("decision") == "complete":
                    proof = validate_ten(result)
                    if restart:
                        c.write(ROOT / "validation10_passed.json", proof)
                        policy.update(target_step=3000, validation_checkpoint_sha256=proof["checkpoint_sha256"])
                        c.write(ROOT / "policy.json", policy)
                        result = c.tick(True)
                        result["transition"] = "ten_step_validation_passed_to_3000_total"
                    else:
                        result["validation10_verified"] = proof
                result["phase"] = (
                    "validation10" if c.read(ROOT / "policy.json")["target_step"] == 10 else "production3000"
                )
        except Exception as exc:
            result = dict(
                decision="integrity_or_controller_error", action="none", error_type=type(exc).__name__, error=str(exc)
            )
        result["checked_at_utc"] = c.now()
        c.write(ROOT / "workflow_status.json", result)
        return result


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("mode", choices=["tick"])
    p.add_argument("--restart", action="store_true")
    a = p.parse_args()
    print(json.dumps(tick(a.restart)))
