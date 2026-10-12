import ast, sys, time, json, numpy as np, jax, jax.numpy as jnp
jax.config.update("jax_enable_x64", True)
D="/pscratch/sd/h/hongkelu/freeboundary-single-stage/vmex/examples/optimization/"
sys.path.insert(0, D); import coil_rows_local as L
src=open(D+"single_stage_free_boundary_optimization_three_term.py").read()
keep=[n for n in ast.parse(src).body if isinstance(n, ast.FunctionDef) and n.name in ("resampled","segment_distances","separations","coil_forces")]
ns={}; exec(compile(ast.Module(body=keep, type_ignores=[]),"drv","exec"), ns)
from essos.coils import Coils
from essos.objective_functions import regularization_circ
coils=Coils.from_json("coils/gilpar_L30_F11.json"); reg=regularization_circ(0.05); nb=coils.dofs_curves.shape[0]
for N in (256,):
    c=ns["resampled"](coils,N); P=c.gamma
    f_full=lambda P: jnp.stack(ns["separations"](P)); f_loc=lambda P: jnp.stack(L.separations_local(P))
    print("sep values full", np.asarray(f_full(P)), "local", np.asarray(f_loc(P)))
    gf=jax.jacrev(f_full)(P); gl=jax.jacrev(f_loc)(P); print("sep jac max|diff|", float(jnp.max(jnp.abs(gf-gl))), "max|jac|", float(jnp.max(jnp.abs(gf))))
    g,gd,gdd,I,q=c.gamma,c.gamma_dash,c.gamma_dashdash,jnp.asarray(coils.currents),c.quadpoints
    ns["FORCE_POINTS"]=N; ns["CONDUCTOR_RADIUS"]=0.05
    ff=lambda g,gd,gdd,I: L.coil_forces_local(g,gd,gdd,I,q,nb,reg)
    from essos.fields import BiotSavart_from_gamma
    from essos.objective_functions import B_regularized_pure
    def full(g,gd,gdd,I):
        others=jnp.asarray([[j for j in range(g.shape[0]) if j!=i] for i in range(nb)])
        def one(i,o):
            m=BiotSavart_from_gamma(g[o],gd[o],gdd[o],I[o]); b=jax.vmap(m.B)(g[i])+B_regularized_pure(g[i],gd[i],gdd[i],q,I[i],reg)
            t=gd[i]/jnp.linalg.norm(gd[i],axis=1)[:,None]; return jnp.max(jnp.linalg.norm(jnp.cross(I[i]*t,b),axis=1))
        return jax.vmap(one)(jnp.arange(nb),others)
    print("force full", np.asarray(full(g,gd,gdd,I))[:3], "local", np.asarray(ff(g,gd,gdd,I))[:3])
    jf=jax.jacrev(full,argnums=(0,3))(g,gd,gdd,I); jl=jax.jacrev(ff,argnums=(0,3))(g,gd,gdd,I)
    print("force jac rel diff g", float(jnp.max(jnp.abs(jf[0]-jl[0]))/jnp.max(jnp.abs(jf[0]))), "I", float(jnp.max(jnp.abs(jf[1]-jl[1]))/jnp.max(jnp.abs(jf[1]))))
N=1024; c=ns["resampled"](coils,N); P=c.gamma; g,gd,gdd,I,q=c.gamma,c.gamma_dash,c.gamma_dashdash,jnp.asarray(coils.currents),c.quadpoints
t=time.time(); j=jax.jit(jax.jacrev(lambda P: jnp.stack(L.separations_local(P))))(P); j.block_until_ready(); t1=time.time()-t
t=time.time(); j=jax.jit(jax.jacrev(lambda P: jnp.stack(L.separations_local(P))))(P); j.block_until_ready(); print(f"N=1024 sep jacrev ok, first {t1:.1f}s")
t=time.time(); j=jax.jit(jax.jacrev(lambda g,gd,gdd,I: L.coil_forces_local(g,gd,gdd,I,q,nb,reg),argnums=(0,1,2,3)))(g,gd,gdd,I); jax.block_until_ready(j); print(f"N=1024 force jacrev ok {time.time()-t:.1f}s")
print("values 1024: sep", np.asarray(L.separations_local(P)), "max force", float(jnp.max(L.coil_forces_local(g,gd,gdd,I,q,nb,reg))))
print("peak GPU MiB", jax.devices()[0].memory_stats().get("peak_bytes_in_use",0)/2**20)
