#!/usr/bin/env python3
"""Optimize rotating-ellipse coils with a free-boundary QA objective."""
import argparse
from dataclasses import asdict
from pathlib import Path

from _free_boundary_diagnostics import Diagnostics

DATA = Path(__file__).resolve().parents[1] / "data/free_boundary_qa"
SURFACES = (0.25, 0.5, 0.75, 1.0)
IOTA_TARGET, ASPECT_TARGET, B0_TARGET = 0.2, 5.0, -0.17506474574437714
FORCE_TOLERANCE, ROOT_TOLERANCE, ADJOINT_TOLERANCE = 1e-11, 2e-6, 2e-5
TARGET_STEP, MAX_WALL_HOURS, DEVICE = 3000, 1.0, "cuda:0"
ADJOINT_BATCH_SIZE = 32


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--target-step", type=int, default=TARGET_STEP)
    parser.add_argument("--max-wall-hours", type=float, default=MAX_WALL_HOURS)
    parser.add_argument("--device", default=DEVICE)
    parser.add_argument("--initialize-only", action="store_true")
    parser.add_argument("--resume-checkpoint", type=Path)
    parser.add_argument("--checkpoint-sha256")
    parser.add_argument("--adjoint-dense-batch-size", choices=("32", "64", "auto"), default=str(ADJOINT_BATCH_SIZE))
    args = parser.parse_args(argv)
    if not 1 <= args.target_step <= 3000 or not 0 < args.max_wall_hours <= 24:
        parser.error("target step must be in 1..3000 and walltime in (0, 24] hours")
    if bool(args.resume_checkpoint) != bool(args.checkpoint_sha256):
        parser.error("resume requires both checkpoint and SHA256")
    if args.initialize_only and args.resume_checkpoint:
        parser.error("initialize-only cannot resume")
    return args


def main(argv=None):
    args = parse_args(argv)
    with Diagnostics(args.output_dir, max_wall_hours=args.max_wall_hours) as output:
        import jax
        import numpy as np
        import vmex
        from vmex import optimize as opt
        from vmex.core.statephysics import on_axis_magnetic_field
        from essos.coils import Coils

        # Inputs and design variables: three relative currents plus 108 coefficients.
        inp = vmex.VmecInput.from_file(DATA / "input.rotating_ellipse")
        coils = Coils.from_json(str(DATA / "coils.json"))
        coils.n_segments = 75  # Preserve the equilibrium field quadrature.
        scales = np.r_[np.full(3, 0.06), np.tile([0.002] + [0.002/k**2 for k in range(1, 5) for _ in range(2)], 12)]
        parameters = opt.CoilParameters.from_coils(coils, current_dofs=(1, 2, 3), max_coil_mode=4, scales=scales)

        # Initial-normalized QA, with the original physical target bands.
        qs = opt.QuasisymmetryRatioResidual(SURFACES, helicity_m=1, helicity_n=0)
        objective_terms = [(qs, 0.0, 1.0)]
        constraints = [
            opt.TargetBand(opt.mean_iota, IOTA_TARGET, rtol=0.01, scale=0.005),
            opt.TargetBand(opt.aspect_ratio, ASPECT_TARGET, rtol=0.01, scale=0.05),
            opt.TargetBand(on_axis_magnetic_field, B0_TARGET, rtol=0.01, scale=0.01),
        ]
        policy = opt.ProjectedOptions(maximum_coil_step_m=0.001, maximum_current_fraction_step=0.01, max_trials=6)
        platform, _, ordinal = args.device.partition(":")
        if platform not in ("cpu", "cuda", "gpu") or (ordinal and not ordinal.isdecimal()):
            raise ValueError("device must be cpu, cpu:N, cuda:N or gpu:N")
        solver = dict(device=jax.devices("gpu" if platform in ("cuda", "gpu") else "cpu")[int(ordinal or 0)],
                      ftol=FORCE_TOLERANCE, edge_force_tolerance=FORCE_TOLERANCE, max_iterations=12000,
                      include_edge_in_convergence=True, adjoint_solver="forward_dense_jax", adjoint_fail="error",
                      adjoint_tol=ADJOINT_TOLERANCE, adjoint_residual_rtol=ADJOINT_TOLERANCE,
                      adjoint_dense_batch_size=32 if args.adjoint_dense_batch_size == "auto" else int(args.adjoint_dense_batch_size),
                      adjoint_dense_max_dofs=4096, adjoint_gcrot_m=30, adjoint_gcrot_k=10, adjoint_maxiter=100)

        # Fresh starts solve once; resumes certify the exact saved accepted root.
        problem = opt.FreeBoundaryProblem.from_tuples(
            inp, objective_terms, parameterization=parameters, constraints=constraints,
            restart_from=None if args.resume_checkpoint else DATA / "initial_state.npz",
            checkpoint=args.resume_checkpoint, checkpoint_sha256=args.checkpoint_sha256,
            checkpoint_identity=dict(qa_surfaces=SURFACES, helicity=[1, 0], optimizer=asdict(policy)),
            solver_options=solver, root_residual_atol=ROOT_TOLERANCE,
            continuation_step=0.1, max_continuation_steps=64, event=output.solver_event, deadline=output.deadline)
        output.bind(problem, inputs=DATA)
        if args.target_step <= problem.accepted_step:
            raise ValueError("target step must exceed the saved accepted step")
        if args.initialize_only:
            output.status = "initialized_only"
            return
        if args.adjoint_dense_batch_size == "auto":
            output.save_json("adjoint_batch_tuning.json", problem.tune_adjoint_batch())

        # One optimizer implementation; only accepted steps enter the history.
        result = opt.minimize_projected(
            problem, maxiter=args.target_step - problem.accepted_step, options=policy,
            initial_gradient_norm=problem.initial_gradient_norm, event=output.optimizer_event,
            callback=output.monitor)
        output.status = result.status
        opt.EquilibriumReporter(("QA", qs.total_state, ".6e"), ("aspect", opt.aspect_ratio, ".4f"),
                                ("mean iota", opt.mean_iota, ".4f"))("final", problem.equilibrium_from_x(result.x))
    print(f"Accepted history and matching coil/WOUT files: {args.output_dir}")


if __name__ == "__main__":
    main()
