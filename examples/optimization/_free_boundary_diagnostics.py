"""Durable accepted-step history and matched coil/free-boundary snapshots."""
import csv
import hashlib
import json
import math
import os
import signal
import time
import sys
from pathlib import Path


FIELDS = (
    'step', 'elapsed_s', 'mean_iota', 'qs_error_raw',
    'qs_objective_normalized', 'aspect_ratio', 'b0_T',
    'iota_error', 'aspect_error', 'b0_error_T', 'constraints_feasible',
    'root_residual', 'fsqr', 'fsqz', 'fsql', 'fedge',
    'checkpoint_path', 'checkpoint_sha256',
)


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def append_line(path, text):
    with path.open('a') as stream:
        stream.write(text + '\n')
        stream.flush()
        os.fsync(stream.fileno())


class AcceptedHistory:
    """Only accepted states enter history; both exports share that exact state."""

    def __init__(self, output, *, targets, tolerances, export_interval=20):
        if export_interval < 1:
            raise ValueError('export interval must be positive')
        self.output = Path(output).resolve()
        self.root = self.output / 'diagnostics'
        self.root.mkdir(exist_ok=False)
        self.targets = list(map(float, targets))
        self.tolerances = list(map(float, tolerances))
        self.interval = export_interval
        self.last_step = None
        self.exports = {}
        schema = dict(
            schema='vmex.accepted-step-diagnostics/v1',
            row_semantics='Initial certified state, then one row per accepted optimizer step; rejected trials remain in progress.jsonl.',
            targets=self.targets, absolute_tolerances=self.tolerances,
            columns=list(FIELDS),
            units={'b0_T': 'T, signed', 'b0_error_T': 'T', 'elapsed_s': 's',
                   'coil_coordinates': 'm', 'coil_currents': 'A'},
            qs_error_raw='Sum of squared campaign QA ratio residuals at normalized flux s=[0.25,0.5,0.75,1], helicity (1,0), evaluated on the accepted state. Lower is better; this is not a separately computed Boozer or WOUT QS diagnostic.',
            qs_objective_normalized='0.5 * qs_error_raw / initial_loss_scale**2',
            mean_iota='Canonical state mean iota; WOUT comparison uses mean(iotas[1:]).',
            b0_T='Canonical signed on-axis magnetic field, not its absolute value.',
            snapshots='Step zero, each multiple of 20, and the final accepted state. Each metadata file binds coils, WOUT, metrics and checkpoint by step and SHA256.',
        )
        (self.root / 'schema.json').write_text(json.dumps(schema, indent=2) + '\n')
        with (self.output / 'metrics.csv').open('x', newline='') as stream:
            csv.DictWriter(stream, fieldnames=FIELDS).writeheader()

    def record(self, *, step, elapsed_s, metrics, root_residual, forces,
               checkpoint, export):
        if self.last_step is not None and step != self.last_step + 1:
            raise ValueError('accepted-step history must be contiguous and unique')
        physical, errors = metrics['physical'], metrics['target_errors']
        checkpoint_path = Path(checkpoint['path']).resolve()
        if not checkpoint_path.is_relative_to(self.output):
            raise ValueError('checkpoint must belong to this run')
        if checkpoint['accepted_step'] != step or sha256(checkpoint_path) != checkpoint['sha256']:
            raise ValueError('checkpoint does not match the recorded accepted step')
        row = dict(
            step=int(step), elapsed_s=float(elapsed_s),
            mean_iota=float(physical['mean_iota']),
            qs_error_raw=float(metrics['qs_error_raw']),
            qs_objective_normalized=float(metrics['qs_objective_normalized']),
            aspect_ratio=float(physical['aspect_ratio']), b0_T=float(physical['b0']),
            iota_error=float(errors['mean_iota']), aspect_error=float(errors['aspect_ratio']),
            b0_error_T=float(errors['b0']),
            constraints_feasible=all(abs(errors[n]) <= tol for n, tol in
                                     zip(('mean_iota', 'aspect_ratio', 'b0'), self.tolerances)),
            root_residual=float(root_residual),
            **{name: float(forces[name]) for name in ('fsqr', 'fsqz', 'fsql', 'fedge')},
            checkpoint_path=str(checkpoint_path.relative_to(self.output)),
            checkpoint_sha256=checkpoint['sha256'],
        )
        if any(not math.isfinite(v) for v in row.values() if isinstance(v, float)):
            raise ValueError('nonfinite history value')
        with (self.output / 'metrics.csv').open('a', newline='') as stream:
            csv.DictWriter(stream, fieldnames=FIELDS).writerow(row)
            stream.flush()
            os.fsync(stream.fileno())
        append_line(self.output / 'metrics.jsonl', json.dumps(row, allow_nan=False))
        self.last_step = step
        if step == 0 or step % self.interval == 0:
            self.snapshot(step=step, checkpoint=checkpoint, metrics=metrics, export=export)

    def snapshot(self, *, step, checkpoint, metrics, export):
        if step in self.exports:
            if self.exports[step]['checkpoint_sha256'] != checkpoint['sha256']:
                raise ValueError('cannot reuse a snapshot for a different checkpoint')
            return self.exports[step]
        directory = self.root / f'step_{step:04d}'
        directory.mkdir(exist_ok=False)
        # A snapshot is published in the index only once BOTH serializers succeed.
        export(directory / 'coils.json', directory / 'wout.nc')
        files = {name: dict(path=str((directory / name).relative_to(self.output)),
                           sha256=sha256(directory / name))
                 for name in ('coils.json', 'wout.nc')}
        record = dict(step=step, checkpoint_sha256=checkpoint['sha256'],
                      checkpoint_path=str(Path(checkpoint['path']).relative_to(self.output)),
                      metrics=metrics, files=files,
                      coil_semantics='Four base coils and their NFP=2 stellarator-symmetric copies; WOUT EXTCUR stores the four actual base currents in A.')
        text = json.dumps(record, indent=2, allow_nan=False) + '\n'
        (directory / 'metadata.json').write_text(text)
        append_line(self.root / 'snapshots.jsonl', json.dumps(record, allow_nan=False))
        self.exports[step] = record
        return record


class Diagnostics:
    """Observe public problem events; persist accepted results without changing them."""

    def __init__(self, output, *, max_wall_hours):
        self.output = Path(output).resolve()
        self.output.mkdir(parents=True, exist_ok=False)
        for key, name in (("JAX_COMPILATION_CACHE_DIR", "jax"), ("MPLCONFIGDIR", "mpl"),
                          ("XDG_CACHE_HOME", "xdg"), ("TMPDIR", "tmp"), ("CUDA_CACHE_PATH", "cuda")):
            directory = self.output / "cache" / name
            directory.mkdir(parents=True)
            os.environ[key] = str(directory)
        os.environ.update(JAX_ENABLE_X64="1", XLA_PYTHON_CLIENT_PREALLOCATE="false", PYTHONDONTWRITEBYTECODE="1")
        self.started = time.monotonic()
        self.deadline = self.started + max_wall_hours * 3600
        self.problem = self.history = self.monitor = None
        self.status = "initializing"
        self.last_checkpoint = None
        self.last_candidate = None

    def __enter__(self):
        self.handlers = {sig: signal.signal(sig, self._stop) for sig in (signal.SIGTERM, signal.SIGINT)}
        return self

    @staticmethod
    def _stop(signum, frame):
        raise TimeoutError(f"signal {signum}")

    def save_json(self, name, data):
        path = self.output / name
        temporary = path.with_suffix(path.suffix + ".tmp")
        temporary.write_text(json.dumps(data, indent=2, allow_nan=False) + "\n")
        temporary.replace(path)

    def event(self, phase, **data):
        append_line(self.output / "progress.jsonl", json.dumps(
            dict(phase=phase, elapsed_s=time.monotonic()-self.started, **data), allow_nan=False))

    def bind(self, problem, *, inputs):
        from vmex import optimize as opt
        import vmex
        import jax
        self.problem = problem
        self.initial_step = problem.accepted_step
        self.monitor = opt.OptimizationMonitor(problem)
        self.history = AcceptedHistory(self.output, targets=problem.targets,
                                       tolerances=problem.constraint_tolerances)
        self.save_json("manifest.json", dict(
            vmex=vmex.__version__, python=sys.version, jax=jax.__version__,
            inputs={p.name: sha256(p) for p in sorted(Path(inputs).iterdir()) if p.is_file()},
            checkpoint_identity=json.loads(problem.checkpoint_identity),
            source_sha256={str(p.relative_to(Path(vmex.__file__).parent.parent)): sha256(p)
                           for p in Path(vmex.__file__).parent.rglob("*.py")},
            example_sha256={p.name: sha256(p) for p in (Path(__file__), Path(__file__).with_name(
                "single_stage_free_boundary_optimization.py"))}))
        problem.inp.to_indata(self.output / "input.effective")
        self.record_accepted()

    def metrics(self):
        import numpy as np
        problem = self.problem
        rows = problem.optimizer_rows(problem.accepted)
        physical = problem.constraint_values(problem.accepted.parameters)
        names = ("mean_iota", "aspect_ratio", "b0")
        return dict(qs_error_raw=float((rows[0]*problem.loss_scale)**2),
                    qs_objective_normalized=float(0.5*rows[0]**2),
                    physical=dict(zip(names, map(float, physical))),
                    target_errors=dict(zip(names, map(float, physical-np.asarray(problem.targets)))))

    def export(self, coil_path, wout_path):
        import vmex
        point = self.problem.accepted.parameters
        self.problem.coils_from_x(point).to_json(str(coil_path))
        equilibrium = self.problem.equilibrium_from_x(point)
        vmex.write_wout(wout_path, equilibrium.wout, overwrite=False)

    def record_accepted(self):
        p = self.problem
        self.last_checkpoint = p.save_checkpoint(self.output / f"checkpoint_step_{p.accepted_step:04d}.npz")
        self.save_json("latest_checkpoint.json", self.last_checkpoint)
        self.history.record(step=p.accepted_step, elapsed_s=time.monotonic()-self.started,
                            metrics=self.metrics(), root_residual=p.accepted.root_residual_norm,
                            forces={n: float(getattr(p.accepted.result, n)) for n in ("fsqr", "fsqz", "fsql", "fedge")},
                            checkpoint=self.last_checkpoint, export=self.export)

    def optimizer_event(self, phase, **data):
        if phase == "direction":
            direction = data["direction"]
            self.event(phase, absolute_step=self.problem.accepted_step, mode=direction.mode,
                       projected_gradient_norm=float(direction.projected_gradient_norm),
                       gradient_threshold=float(data["threshold"]), constraints=direction.constraint_info,
                       restoring_weight=float(direction.restoring_weight))
        elif phase == "trial":
            self.event("trial_decision", absolute_step=self.problem.accepted_step+1, **data["record"])
        elif phase == "accepted":
            self.last_candidate = None
            self.record_accepted()
            self.event("promoted", absolute_step=self.problem.accepted_step, metrics=self.metrics(),
                       checkpoint=self.last_checkpoint)
        elif phase == "stagnated":
            self.status = "stagnated"
            self.event(phase, trials=len(data["trials"]))

    def solver_event(self, phase, **data):
        import numpy as np
        if phase == "proposal":
            self.trial_name = f"trial_step_{self.problem.accepted_step+1:04d}_trial_{data['trial']:02d}"
            np.savez_compressed(self.output / (self.trial_name+"_proposal.npz"),
                                parameters=self.problem.accepted.parameters, delta=data["delta"],
                                rows=self.problem.optimizer_rows(self.problem.accepted), jacobian=data["jacobian"])
            self.event(phase, trial=data["trial"], points=data["points"])
        elif phase == "correction":
            result = data["stage"].result
            self.event(phase, trial=data["trial"], index=data["index"], seconds=data["seconds"],
                       converged=bool(result.converged), iterations=int(result.iterations),
                       forces={n: float(getattr(result, n)) for n in ("fsqr", "fsqz", "fsql", "fedge")})
        elif phase == "certification":
            self.event(phase, trial=data["trial"], root_residual=data["candidate"].root_residual_norm)
        elif phase == "candidate":
            stage = data["stage"]
            if stage is not None:
                fields = ("R_cos", "R_sin", "Z_cos", "Z_sin", "L_cos", "L_sin")
                path = self.output / (self.trial_name+"_candidate.npz")
                np.savez_compressed(path, parameters=np.asarray(data["parameters"]),
                                    rcon0=np.asarray(stage.rcon0), zcon0=np.asarray(stage.zcon0), eligible_for_resume=False,
                                    **{n: np.asarray(getattr(stage.result.state, n)) for n in fields})
                self.last_candidate = dict(path=str(path), sha256=sha256(path), eligible_for_resume=False)
        else:
            self.event(phase, **data)

    def __exit__(self, exc_type, exc, traceback):
        summary = dict(status=self.status, elapsed_s=time.monotonic()-self.started)
        try:
            if exc is not None:
                summary["status"] = "failed"
                self.event("failure", error_type=type(exc).__name__, error=str(exc))
            try:
                if self.problem is not None and self.last_checkpoint is not None:
                    p = self.problem
                    # Save the stopping reference even when no new step was accepted.
                    final_checkpoint = p.save_checkpoint(self.output / f"checkpoint_final_{p.accepted_step:04d}.npz")
                    self.save_json("latest_checkpoint.json", final_checkpoint)
                    summary.update(initial_step=self.initial_step, final_step=p.accepted_step,
                                   final_checkpoint=final_checkpoint,
                                   initial_projected_gradient_norm=p.initial_gradient_norm,
                                   unaccepted_diagnostic_state=self.last_candidate)
                    self.monitor.save(self.output / "objectives.csv")
                    summary["final_metrics"] = self.metrics()
                    snapshot = self.history.snapshot(step=p.accepted_step, checkpoint=self.last_checkpoint,
                                                     metrics=summary["final_metrics"], export=self.export)
                    summary.update(final_wout=snapshot["files"]["wout.nc"],
                                   final_coils=snapshot["files"]["coils.json"])
            except Exception as error:
                summary.update(status="failed", finalization_error=str(error))
                if exc is None:
                    raise
            finally:
                self.save_json("summary.json", summary)
        finally:
            if self.problem is not None:
                self.problem.close()
            for sig, handler in self.handlers.items():
                signal.signal(sig, handler)
        return False
