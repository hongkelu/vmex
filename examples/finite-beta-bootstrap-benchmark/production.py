"""Build -> optimize -> independently verify, matching the main entry points.

No derivative experiments or implicit scientific jobs at import time.
"""

import json
from case import contract, setup, write
from main_reference import REFERENCE, VACUUM_SCRIPTS


def run(arm, args, *, build_problem, run_optimizer, verify_endpoint):
    if args.dry_run:
        print(
            json.dumps(
                dict(
                    arm=arm,
                    vacuum_reference=str(REFERENCE / VACUUM_SCRIPTS[arm]),
                    workflow=["build_problem", "run_optimizer", "verify_endpoint"],
                    **contract(),
                ),
                indent=2,
            )
        )
        return 0
    out = setup(args)
    try:
        from optimizer_driver import report_result

        stage = build_problem(args)
        result = run_optimizer(stage, args)
        optimization = report_result(stage, result)
        write(out / "optimization.json", optimization)
        verification = verify_endpoint(stage, args, result)
        return 0 if optimization["optimizer_success"] and verification["all_reported_checks_pass"] else 2
    except Exception as error:
        write(out / "failure.json", dict(error=f"{type(error).__name__}: {error}"))
        raise
