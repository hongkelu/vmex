"""Compare completed, matched pilot artifacts; never stop or launch a job."""
from pathlib import Path
import argparse
import json
import numpy as np


def read(root, name):
    return json.loads((root / name).read_text())


def compare(root):
    result = dict(arms={}, eligible_to_replace=False)
    for arm in ('fixed', 'free'):
        old, new = root / f'{arm}-baseline', root / f'{arm}-candidate'
        results = [read(p, 'result.json') for p in (old, new)]
        contracts = [read(p, 'contract.json') for p in (old, new)]
        assert all(r['status'] == 'pilot_and_fd_passed' and r['accepted_steps'] == 2 for r in results)
        assert all(read(p, 'gradient_verification.json')['passed'] for p in (old, new))
        for key in ('schema', 'optimization_targets', 'shared_main', 'core_sources'):
            assert contracts[0][key] == contracts[1][key], (arm, key)
        numerical = {'TRIAL_DENSE_REBUILDS', 'NEWTON_STAGNATION_STEPS'}
        assert ({k: v for k, v in contracts[0]['physics'].items() if k not in numerical}
                == {k: v for k, v in contracts[1]['physics'].items() if k not in numerical})
        starts = [read(p, 'restart.json') for p in (old, new)]
        for key in ('checkpoint_sha256', 'prepared_manifest_sha256'):
            assert starts[0][key] == starts[1][key], (arm, key)
        runtime = [read(p, 'runtime.json') for p in (old, new)]
        assert runtime[0]['versions'] == runtime[1]['versions']
        assert runtime[0]['hardware'] == runtime[1]['hardware']
        with np.load(old / 'initial-derivative.npz', allow_pickle=False) as a, \
             np.load(new / 'initial-derivative.npz', allow_pickle=False) as b:
            np.testing.assert_array_equal(a['x'], b['x'])
            np.testing.assert_allclose(a['rows'], b['rows'], rtol=1e-7, atol=1e-9)
            gradient_errors = np.linalg.norm(a['gradient']-b['gradient'], axis=1) / np.maximum(
                np.linalg.norm(a['gradient'], axis=1), 1e-10)
            assert np.all(gradient_errors <= 1e-5), (arm, gradient_errors)
        histories = [[json.loads(s) for s in (p / 'accepted_steps.jsonl').read_text().splitlines()]
                     for p in (old, new)]
        assert all([r['step'] for r in h] == [0, 1, 2] for h in histories)
        endpoints = [h[-1] for h in histories]
        optimization = [read(p, 'optimization.json') for p in (old, new)]
        objective_ok = endpoints[1]['objective'] <= endpoints[0]['objective'] * (1+1e-5) + 1e-8
        slack_ok = (optimization[1]['minimum_optimizer_slack'] >=
                    optimization[0]['minimum_optimizer_slack'] - 1e-4)
        elapsed = [r['seconds_to_accepted_budget'] for r in results]
        assert all(np.isfinite(t) and t > 0 for t in elapsed)
        speedup = elapsed[0] / elapsed[1]
        # A marginal change needs a repeat to distinguish it from timing noise.
        needs_repeat = 1 < speedup < 1.1
        eligible = bool(speedup >= 1.1 and objective_ok and slack_ok)
        result['arms'][arm] = dict(
            speedup=speedup, baseline_seconds=elapsed[0], candidate_seconds=elapsed[1],
            gradient_relative_errors=gradient_errors.tolist(),
            baseline_endpoint=endpoints[0], candidate_endpoint=endpoints[1],
            objective_progress_comparable=bool(objective_ok), constraint_progress_comparable=bool(slack_ok),
            needs_timing_repeat=needs_repeat, eligible_to_replace=eligible,
            note='Two accepted steps and FD checks qualify this bounded comparison, not convergence.')
    result['eligible_to_replace'] = all(r['eligible_to_replace'] for r in result['arms'].values())
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('root', type=Path)
    a = parser.parse_args()
    result = compare(a.root)
    (a.root / 'comparison.json').write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result, indent=2))
