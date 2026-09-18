"""Verify matched periodic exports and complete accepted-step histories."""

from pathlib import Path
import csv
import hashlib
import importlib.util
import json
import sys
import pytest

spec = importlib.util.spec_from_file_location(
    "case_diagnostics", Path(__file__).resolve().parents[1] / "diagnostics.py"
)
m = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = m
spec.loader.exec_module(m)


def values(step):
    return dict(
        qs_error_raw=0.1 / (step + 1),
        qs_objective_normalized=0.5 / (step + 1),
        physical=dict(mean_iota=0.2, aspect_ratio=5.0, b0=-0.175),
        target_errors=dict(mean_iota=0.0, aspect_ratio=0.0, b0=0.0),
    )


def checkpoint(root, step):
    p = root / f"checkpoint_step_{step:04d}.npz"
    p.write_bytes(str(step).encode())
    return dict(path=str(p), sha256=hashlib.sha256(p.read_bytes()).hexdigest(), accepted_step=step)


def recorder(root):
    return m.Diagnostics(root, targets=[0.2, 5.0, -0.175], tolerances=[0.002, 0.05, 0.00175])


def record(d, root, step, export, **changes):
    args = dict(
        step=step,
        elapsed_s=float(step),
        metrics=values(step),
        root_residual=1e-7,
        forces=dict(fsqr=1e-11, fsqz=1e-11, fsql=1e-12, fedge=1e-11),
        checkpoint=checkpoint(root, step),
        export=export,
    )
    args.update(changes)
    d.record(**args)
    return args


def test_periodic_exports_and_roundtrip_history(tmp_path):
    d = recorder(tmp_path)
    calls = []
    for step in range(41):

        def export(coils, wout, step=step):
            calls.append(step)
            coils.write_text(json.dumps({"step": step}))
            wout.write_bytes(f"wout-{step}".encode())

        record(d, tmp_path, step, export)
    assert calls == [0, 20, 40]
    rows = list(csv.DictReader((tmp_path / "metrics.csv").open()))
    native = [json.loads(s) for s in (tmp_path / "metrics.jsonl").read_text().splitlines()]
    assert [int(r["step"]) for r in rows] == list(range(41))
    for step, (row, item) in enumerate(zip(rows, native)):
        assert float(row["qs_error_raw"]) == item["qs_error_raw"] == values(step)["qs_error_raw"]
        assert float(row["b0_T"]) == -0.175 and item["constraints_feasible"]
    snapshots = [json.loads(s) for s in (tmp_path / "diagnostics/snapshots.jsonl").read_text().splitlines()]
    for entry in snapshots:
        step = entry["step"]
        assert entry["metrics"] == values(step)
        for artifact in entry["files"].values():
            assert m.sha256(tmp_path / artifact["path"]) == artifact["sha256"]
    cp = checkpoint(tmp_path, 40)
    d.snapshot(step=40, checkpoint=cp, metrics=values(40), export=lambda *args: pytest.fail("duplicate export"))
    assert calls == [0, 20, 40]


def test_gap_or_duplicate_cannot_enter_history(tmp_path):
    d = recorder(tmp_path)

    def export(c, w):
        c.write_text("{}")
        w.write_bytes(b"wout")

    record(d, tmp_path, 0, export)
    for step in (0, 2):
        with pytest.raises(ValueError, match="contiguous"):
            record(d, tmp_path, step, export)
    assert len((tmp_path / "metrics.jsonl").read_text().splitlines()) == 1


def test_unmatched_checkpoint_and_nonfinite_values_rejected(tmp_path):
    d = recorder(tmp_path)
    cp = checkpoint(tmp_path, 0)
    cp["accepted_step"] = 1
    with pytest.raises(ValueError, match="checkpoint"):
        record(d, tmp_path, 0, lambda *args: None, checkpoint=cp)
    bad = values(0)
    bad["physical"]["b0"] = float("nan")
    with pytest.raises(ValueError, match="nonfinite"):
        record(d, tmp_path, 0, lambda *args: None, metrics=bad)
    assert not (tmp_path / "metrics.jsonl").exists()


def test_partial_export_not_published(tmp_path):
    d = recorder(tmp_path)

    def export(c, w):
        w.write_bytes(b"partial")
        raise OSError("coil writer failed")

    with pytest.raises(OSError):
        record(d, tmp_path, 0, export)
    assert not (tmp_path / "diagnostics/snapshots.jsonl").exists()
    assert not (tmp_path / "diagnostics/step_0000/metadata.json").exists()


def test_final_noninterval_snapshot(tmp_path):
    d = recorder(tmp_path)
    calls = []

    def export(c, w):
        calls.append(c.parent.name)
        c.write_text("{}")
        w.write_bytes(b"wout")

    for step in range(4):
        last = record(d, tmp_path, step, export)
    d.snapshot(step=3, checkpoint=last["checkpoint"], metrics=last["metrics"], export=export)
    assert calls == ["step_0000", "step_0003"]
