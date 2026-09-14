"""Durable accepted-step history and matched coil/free-boundary snapshots."""
import csv
import hashlib
import json
import math
import os
from pathlib import Path


FIELDS = (
    'step', 'elapsed_s', 'mean_iota', 'qs_error_raw',
    'qs_objective_normalized', 'aspect_ratio', 'b0_T',
    'iota_error', 'aspect_error', 'b0_error_T', 'major_radius_m', 'major_radius_error_m', 'constraints_feasible',
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


class Diagnostics:
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
            b0_error_T=float(errors['b0']), major_radius_m=float(physical['major_radius']), major_radius_error_m=float(errors['major_radius']),
            constraints_feasible=all(abs(errors[n]) <= tol for n, tol in
                                     zip(('mean_iota', 'aspect_ratio', 'b0', 'major_radius'), self.tolerances)),
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
