# The three-terms free boundary by least squares

This page covers ``method="least_squares"`` of
{func}`~vmex.core.freeboundary_vc.solve_free_boundary_three_terms`
({mod}`vmex.core.freeboundary_lsq`; the default, ``method="galerkin"``, solves a
square system for the deck's normal displacement, and is faster): what it
minimizes, why the poloidal labelling of the boundary needs a gauge, how the
exact Hessian is built through the equilibrium, and how the design gradients
of a single-stage problem follow from it.

## What is minimized

The unknowns are the boundary Fourier coefficients $b$ (with $R_{00}$).
Every iterate is a converged fixed-boundary equilibrium on that boundary.
The rows $r(b)$ are the interface conditions on an $n_\phi \times n_\theta$
grid over one field period, each weighted by the square root of its area
element:

- in vacuum, $B_\mathrm{coil}\cdot n / B_0$. $B_\mathrm{out}$ is the coil
  field exactly, so this is the field-line condition, and the interior field
  does not enter;
- at finite beta, $B_\mathrm{out}\cdot n$, the pressure-balance jump and the
  three components of $n \times (B_\mathrm{out} - B_\mathrm{in})$, with
  $B_\mathrm{out} = B_\mathrm{coil} + B_\mathrm{plasma}$ from virtual casing.
  $B_\mathrm{in}$ is first restricted to VMEC's own Galerkin band
  ($m \le$ `mpol` $-1$, $|n| \le$ `ntor`); outside it the edge field is not
  determined by the equilibrium.

One more row, $(G_\mathrm{plasma} - G_\mathrm{coil})/G_\mathrm{coil}$, fixes
the net poloidal current. That direction is one that $B\cdot n$ cannot see.
The solver minimizes $f(b) = \tfrac12 |r(b)|^2$.

## Why the labelling needs a gauge

A set of boundary coefficients describes a surface *and* a poloidal angle on
it. The relabelling $\theta \to \theta + \omega(\theta, \phi)$ leaves the
surface unchanged. With truncated series, though, it changes the coefficients
and the shapes they can represent, and with them the residual at the
truncation level.

The exact Hessian of $f$ makes this visible. On P30A coils at 8 × 8 modes:

| subspace | Hessian eigenvalues |
|---|---|
| relabellings | about $-10^{-8}$ to $10^{-9}$: flat and indefinite |
| shape (complement) | $2\cdot 10^{-7}$ to $5\cdot 10^{2}$ |

So a free labelling has no minimum near the start: the cost keeps falling as
the angle drifts. Anchoring the labelling to the start with weighted rows makes
every start solve a different problem. Run to full convergence from three
starts, that gives quasi-symmetry residuals of $4.5$, $3.4$ and
$3.8 \cdot 10^{-6}$, on surfaces 0.1 mm apart.

The solver therefore fixes the labelling by exact linear constraints $c(b) = 0$,
one per relabelling mode $\sin(m\theta - n N_{fp}\phi)$, including the $m = 0$
shifts of the angle origin. As in the Galerkin method, the deck's boundary
$b_\mathrm{deck}$ fixes the labelling: the displacement $b - b_\mathrm{deck}$
has no component along the deck's relabellings (ESS-scaled metric). Only the
labelling is fixed; the shape stays free. The start (``initial_boundary``) is
projected onto $c = 0$, so every start reaches the same boundary.

Newton steps stay in the null space of $\nabla c$, and each trial is
projected back onto $c = 0$ (exact, since $c$ is linear).

### Why not a reference-free gauge

A gauge defined by the boundary alone is possible: stationarity of the spectral
width $F_p = \sum m^p (R_{mn}^2 + Z_{mn}^2)$ under the $m > 0$ relabellings,
and of the twist $\sum (n N_{fp})^2 (R_{mn}^2 + Z_{mn}^2)$ under the shifts. It
is unique, but at low resolution its labelling costs the truncated series real
shape. On an optimized, strongly shaped 8-mode boundary, relabelling to the
$F_6$ minimum and truncating moves the surface by 4-6 mm (maximum normal
distance). Satisfying the constraints exactly moves it by 15-33 mm. For the
vacuum benchmark the field-line distance of the last closed surface at 12 modes
is 0.114 mm in that gauge, against 0.048 mm with the deck's labelling.

### Constraint rows: the bootstrap current

With the bootstrap current in the solve, the current's rows are square in the
current coordinates. They are equality constraints, not least-squares rows: any
finite weight would trade the current against the interface conditions. The
steps live in the null space of the gauge and the current rows' Jacobian. Each
trial is restored onto the current rows by chord Newton steps in the current
coordinates. The Hessian is the Lagrangian's, with the current rows' multipliers.

## The exact Hessian through the equilibrium

Let the equilibrium be $F(z, p) = 0$, with $z$ the evolved state and $p$ the
parameters (the boundary among them), and write $\Phi(z, p) = \tfrac12 |r|^2$.

- The multiplier solves $F_z^\top \lambda = \Phi_z$. That is one transposed
  solve with the block factorization the implicit derivatives already use.
- The Lagrangian is $L = \Phi - \lambda\cdot F$, and the reduced gradient is
  $L_p$.
- Along the state responses $F_z\,\delta z_j = -F_p\,\delta p_j$, the Hessian is

$$
H_{ij} = [\delta p_i;\ \delta z_i]^\top\, \nabla^2 L\, [\delta p_j;\ \delta z_j].
$$

Each column costs one forward-over-reverse pass of $L$. Unlike Gauss-Newton's
$J^\top J$, $H$ contains $\sum_k r_k \nabla^2 r_k$. That term is not small,
because the residual does not vanish at finite resolution.

Checked against central differences of the exact gradient, the error falls as
$\epsilon^2$. In vacuum it is $4\cdot10^{-6}$ to $4\cdot10^{-5}$ at
$\epsilon = 10^{-4}$. At $\beta = 2.5\,\%$ it is $10^{-1}$ at
$\epsilon = 10^{-3}$ and $10^{-3}$ at $10^{-4}$.

The trust-region step is solved exactly in the eigenbasis of the reduced,
ESS-scaled model, so indefinite models are handled too.  The exact Hessian
enters as products: a preconditioned block-Krylov subspace (preconditioner
the reduced $J^\top J$, Eisenstat-Walker forcing), used where a Gauss-Newton
model fails and to certify convergence.

- The Hessian is recomputed every few steps. In between, BFGS updates use exact
  adjoint gradients.
- When a step is not very successful by the radius update's own test, a
  second-order correction $-(JN)^{+} r(x_\mathrm{trial})$ follows it. At finite
  beta this takes a weak, strongly non-quadratic low-order shape mode from
  29 Hessians to 10.
- The solve stops when the Newton step's predicted gain, or the boundary
  displacement it implies, falls below its tolerance. A stall is reported as a
  stall.

## Design gradients

In single-stage optimization the boundary $b^*(x)$ minimizes $f(b, x)$ on the
gauge manifold (and the current rows). A function $h$ of the equilibrium then has

$$
\frac{dh}{dx} = h_x - \mu^\top f_{bx}, \qquad
N^\top W N\, \mu_N = N^\top h_b, \quad \mu = N \mu_N,
$$

with $W$ the Lagrangian Hessian and $N$ a null-space basis of the
constraints' Jacobian. The current rows depend on the plasma parameters too;
their response off the null space adds one more term for those parameters.

- The coil part of $f_{bx}\,\mu$ is one forward-over-reverse pass through the
  field.
- The plasma-parameter part comes from Hessian columns along those parameters.

At $\beta = 2.5\,\%$ (6 × 6) the error against re-solved central differences is
below $2\cdot10^{-3}$ for $\iota$ and aspect ratio, and at most $2\,\%$ for the
quasi-symmetry residual. Gauss-Newton is off by up to 21 % on the same
directions.
