#!/usr/bin/env python3
"""QA optimization with coil currents and Fourier coefficients as variables.

The plasma boundary is determined by the free-boundary equilibrium.
Run with --output-dir NEW_DIRECTORY; --help lists bounded run/resume options.
"""

from example_support import CaseRun, parse_args

SURFACES = (0.25, 0.5, 0.75, 1.0)
IOTA_TARGET, ASPECT_TARGET, B0_TARGET = 0.2, 5.0, -0.17506474574437714
TARGET_STEP, MAX_WALL_HOURS, DEVICE = 3000, 1.0, "cuda:0"
ADJOINT_BATCH_SIZE = 32


def main(argv=None):
    args = parse_args(
        argv, target_step=TARGET_STEP, device=DEVICE, max_wall_hours=MAX_WALL_HOURS, batch_size=ADJOINT_BATCH_SIZE
    )

    # Load authenticated coils/input or a checkpoint; certify the initial root.
    # CaseRun also saves accepted checkpoints and matching coil/WOUT snapshots.
    with CaseRun(args) as run:
        from vmex import optimize as opt
        from vmex.core.statephysics import on_axis_magnetic_field

        # Objective: initial-normalized QA. Constraints retain their 1% bands.
        qs = opt.QuasisymmetryRatioResidual(SURFACES, helicity_m=1, helicity_n=0)
        objective_function_terms = [(qs, 0.0, 1.0)]
        constraints = [
            opt.TargetBand(opt.mean_iota, IOTA_TARGET, rtol=0.01, scale=0.005),
            opt.TargetBand(opt.aspect_ratio, ASPECT_TARGET, rtol=0.01, scale=0.05),
            opt.TargetBand(on_axis_magnetic_field, B0_TARGET, rtol=0.01, scale=0.01),
        ]

        # 3 relative currents + 108 Cartesian coil Fourier coefficients.
        # The chart preserves the first current, symmetry, and original scales.
        problem = opt.FreeBoundaryProblem.from_tuples(
            run.input,
            objective_function_terms,
            parameterization=run.coil_parameters,
            continuation=run.continuation,
            constraints=constraints,
            objective_normalization=run.loss_scale if run.resume is not None else "initial",
            deadline=run.deadline,
        )
        run.bind(problem)
        print(problem.dof_names)
        monitor = opt.OptimizationMonitor(problem)

        if args.initialize_only:
            run.status = "initialized_only"
        else:
            result = opt.minimize_projected(
                problem,
                maxiter=args.target_step - run.step,
                options=vars(run.policy),
                callback=monitor,
                initial_gradient_norm=run.initial_gradient_norm,
                event=run.optimizer_event,
            )
            run.status = result.status
            report = opt.EquilibriumReporter(
                ("QA", qs.total_state, ".6e"),
                ("aspect", opt.aspect_ratio, ".4f"),
                ("mean iota", opt.mean_iota, ".4f"),
            )
            report("final", problem.equilibrium_from_x(result.x))
            monitor.save(run.output / "objectives.csv")
    print(f"Accepted history and final coil/WOUT paths: {run.output}")




if __name__ == "__main__":
    main()
