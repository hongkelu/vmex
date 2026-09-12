"""Case-local equality-tangent direction and sampled physical step."""
import numpy as np


def displacement(delta):
    delta = np.asarray(delta)
    if delta.shape != (111,) or not np.all(np.isfinite(delta)):
        raise ValueError('expected finite 111-vector')
    t = np.arange(75)/75
    basis = np.ones((75,9))
    for k in range(1,5):
        basis[:,2*k-1] = np.sin(2*np.pi*k*t)
        basis[:,2*k] = np.cos(2*np.pi*k*t)
    return np.einsum('cdk,sk->csd',delta[3:].reshape(4,3,9),basis)


def proposal(values, jacobian, scales):
    jac = np.asarray(jacobian)*scales[None,:]
    if jac.shape != (5,111) or not np.all(np.isfinite(jac)):
        raise ValueError('expected finite 5 x 111 Jacobian')
    constraints = jac[1:]
    _, singular, vt = np.linalg.svd(constraints,full_matrices=False)
    if singular[-1] <= 1e-12*max(1.,singular[0]):
        raise RuntimeError('equality Jacobian is rank deficient')
    gradient = float(values[0])*jac[0]
    projected = gradient-vt.T@(vt@gradient)
    direction = -projected*scales
    motion = float(np.max(np.linalg.norm(displacement(direction),axis=-1)))
    if not np.isfinite(motion) or motion <= 0:
        raise RuntimeError('no finite coil-motion direction')
    step = direction*(.001/motion)
    return step
