"""Centered rotating ellipse with fixed area-equivalent aspect ratio."""
import dataclasses
import math
import numpy as np


def rotating_ellipse(reference, *, major_radius, aspect, t, handedness, phiedge,
                     ns=50, ftol=1e-13):
    if handedness not in (-1,1) or not (0<t<1 and major_radius>0 and aspect>1 and phiedge>0):
        raise ValueError('invalid rotating-ellipse geometry or flux')
    a=major_radius/aspect
    c,d=a*math.cosh(t),a*math.sinh(t)
    rbc=np.zeros_like(reference.rbc);zbs=np.zeros_like(reference.zbs)
    n=int(reference.ntor)
    rbc[n,0]=major_radius
    rbc[n,1]=zbs[n,1]=c
    rbc[n+handedness,1]=d;zbs[n+handedness,1]=-d
    axis=np.zeros_like(reference.raxis_c);axis[0]=major_radius
    return dataclasses.replace(reference,rbc=rbc,zbs=zbs,
        rbs=np.zeros_like(reference.rbs),zbc=np.zeros_like(reference.zbc),
        raxis_c=axis,raxis_s=np.zeros_like(reference.raxis_s),
        zaxis_c=np.zeros_like(reference.zaxis_c),zaxis_s=np.zeros_like(reference.zaxis_s),
        ntheta=48,nzeta=40,lfreeb=False,mgrid_file='NONE',phiedge=phiedge,ns_array=(16,ns),
        ftol_array=(1e-8,ftol),niter_array=(4000,12000),ncurr=1)


def geometry_metadata(major_radius,aspect,t,handedness):
    a=major_radius/aspect
    return dict(major_radius_m=major_radius,minor_radius_m=a,aspect=aspect,
        t=t,handedness=handedness,semimajor_m=a*math.exp(t),
        semiminor_m=a*math.exp(-t),semiaxis_ratio=math.exp(2*t))
