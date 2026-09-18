"""Bounded, per-run tuning of the fast dense adjoint's column batch size."""
import statistics
import time

BATCH_SIZES = (32, 64)


def tune_adjoint_batch(evaluate, *, deadline, event, clock=time.monotonic):
    """Return (batch, owned linearization, report) at one unchanged root.

    ``evaluate(batch)`` must return a certified native linearization. Compile
    both shapes, then measure three pairs in alternating order. Choose 64 only
    if it beats 32 by at least 5% in every pair. The first 32 gradient is the
    numerical reference; no candidate may change any gradient row by 1e-8.
    All objects except the returned winner are closed, including on failure.
    No timings or numerical factors are shared across runs/devices.
    """
    import numpy as np

    roots, records = {}, []
    reference = None
    started = clock()
    baseline, candidate = BATCH_SIZES
    try:
        # Discard the first call for each static batch shape (includes JIT).
        schedule = [(batch, None) for batch in BATCH_SIZES]
        schedule += [(batch, pair) for pair, order in enumerate(
            (BATCH_SIZES, BATCH_SIZES[::-1], BATCH_SIZES)) for batch in order]
        for batch, pair in schedule:
            if clock() >= deadline:
                raise TimeoutError('walltime during adjoint batch tuning')
            if batch in roots:
                roots.pop(batch).close()
            event('adjoint_batch_probe_start', batch=batch, pair=pair)
            start = clock()
            root = evaluate(batch)
            roots[batch] = root
            jac = np.asarray(root.field_jacobian)  # synchronize before timing
            seconds = clock() - start
            if clock() >= deadline:
                raise TimeoutError('walltime during adjoint batch tuning')
            if not np.isfinite(seconds) or seconds <= 0:
                raise ValueError('invalid adjoint batch timing')
            if not np.all(np.isfinite(jac)):
                raise ValueError('nonfinite adjoint batch gradient')
            if reference is None:
                reference = jac.copy()
            if jac.shape != reference.shape:
                raise ValueError('adjoint batch gradient shape changed')
            norm = np.linalg.norm(reference, axis=1)
            difference = np.linalg.norm(jac - reference, axis=1)
            if not np.all(np.isfinite(norm)) or not np.all(np.isfinite(difference)):
                raise ValueError('nonfinite adjoint batch gradient norm')
            # Zero reference rows require exact agreement; never hide them
            # behind an absolute tolerance or a division-by-zero workaround.
            relative = np.divide(difference, norm, out=np.zeros_like(norm), where=norm > 0)
            if np.any((norm == 0) & (difference != 0)) or np.any(relative > 1e-8):
                raise ValueError('adjoint batch gradient agreement failed')
            record = dict(batch=batch, pair=pair, warmup=pair is None,
                          seconds=seconds, maximum_gradient_row_relative_difference=float(max(relative)))
            records.append(record)
            event('adjoint_batch_probe', **record)
        timings = {batch: [r['seconds'] for r in records
                          if r['batch'] == batch and not r['warmup']] for batch in BATCH_SIZES}
        ratios = [b / a for a, b in zip(timings[baseline], timings[candidate])]
        selected = candidate if all(r < 0.95 for r in ratios) else baseline
        report = dict(requested='auto', selected_batch_size=selected,
                      candidates=list(BATCH_SIZES), records=records, paired_64_over_32=ratios,
                      warm_median_seconds={str(b): statistics.median(t) for b, t in timings.items()},
                      minimum_consistent_speedup_fraction=0.05,
                      gradient_agreement_rtol=1e-8, elapsed_seconds=clock() - started,
                      reason='consistent_gain' if selected == candidate else 'no_consistent_gain')
        event('adjoint_batch_selected', **report)
        winner = roots.pop(selected)
        return selected, winner, report
    finally:
        for root in roots.values():
            root.close()
