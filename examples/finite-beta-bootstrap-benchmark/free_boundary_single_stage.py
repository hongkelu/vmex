#!/usr/bin/env python
"""Free-boundary scalar single stage: main's workflow plus live Redl current.

Shared physics/coil settings are in bootstrap_settings.py and main parameters.py.
The vacuum reference script remains unchanged. Run verify_gradients.py separately.
"""

from case import arguments, CaseRun

ARM = "free"


def parse_args(argv=None):
    return arguments(ARM, argv)


def build_problem(args):
    """Build a coupled equilibrium/current problem from authenticated inputs."""
    from optimizer_driver import build_stage

    return build_stage(CaseRun(args, ARM, args.output.resolve()))


def run_optimizer(stage, args):
    """Call the optimizer shared by both finite-beta arms."""
    from optimizer_driver import run_optimizer as run_reference

    return run_reference(stage, args, arm=ARM)


def verify_endpoint(stage, args, result):
    """Re-solve the accepted endpoint at NS201 with self-consistent current."""
    return stage.campaign.verify()


def main(argv=None):
    from production import run

    return run(
        ARM, parse_args(argv), build_problem=build_problem, run_optimizer=run_optimizer, verify_endpoint=verify_endpoint
    )


if __name__ == "__main__":
    raise SystemExit(main())
