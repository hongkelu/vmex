# Validation scope

The initial similarity check failed when only coil currents and flux were reversed: leaving internal lambda unchanged gave B0=5.100010346327627 T. Inspection of the pinned VMEX field equations showed that internal lambda multiplies nonnegative lamscale. Reversing both L_cos and L_sin restored signed similarity; the corrected test checks B0=5.1 T and unchanged iota/aspect to relative tolerance 1e-12. R/Z geometry is checked bitwise. The original checkpoint remains unchanged.

Other checks compare the copied pressure/current profiles pointwise through VMEX's independent namelist parser, verify that unrelated input and coil geometry fields are unchanged, preserve explicit targets during initialization/resume, and mock ordinary-solve success/failure to check the force gate and retry prohibition.

No finite-beta equilibrium, coil-optimization step, dense adjoint or GPU job was executed. A local JAX persistent-cache warning reported missing optional filelock; the checks completed without changing the environment.
