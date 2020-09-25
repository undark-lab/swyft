#!/usr/bin/env python
from __future__ import absolute_import, unicode_literals, print_function
import numpy as np
from numpy import pi, cos
from pymultinest.solve import solve
import os
try: os.mkdir('chains')
except OSError: pass

np.random.seed(25)
SIGMA = 0.03
def model(z):
    grid = np.linspace(0, 1, 50, 50)
    X, Y = np.meshgrid(grid, grid)
    x01, y01, r1, w1,fx,fy,Amp = z[0], z[1], z[2]*0.4+0.2, z[3]*0.1+0.05,\
        z[4]*6*np.pi,z[5]*6*np.pi,z[6]
    
    
    R1 = ((X-x01)**2 + (Y-y01)**2)**0.5+0.3*Amp*np.cos(fx*X)*np.sin(fy*Y)
    
    mu = np.exp(-(R1-r1)**2/w1**2/2)
    x = mu
    return x
def noisemodel(x, z = None, noiselevel = 1.):
    n = np.random.randn(*x.shape)*SIGMA  #*noiselevel
    return x + n


z0 = np.array([0.1, 0.3, 0.2, 0.8,0.8,0.6,0.5])
xx0 = noisemodel(model(z0))

def myprior(cube):
    return cube
#maybe should use exact xx0 from SWYFT, with noise? with the same noise instance
def myloglike(cube):
    testDat=model(cube)
    chi= -0.5*((xx0 - testDat)**2).sum()/SIGMA**2
    return chi

# number of dimensions our problem has
parameters = ["x01", "y01", "r1", "w1","fx","fy","Amp"]
n_params = len(parameters)
# name of the output files
prefix = "chains/ring0-"

# run MultiNest
result = solve(LogLikelihood=myloglike, Prior=myprior, 
    n_dims=n_params, outputfiles_basename=prefix, verbose=True)

print()
print('evidence: %(logZ).1f +- %(logZerr).1f' % result)
print()
print('parameter values:')
for name, col in zip(parameters, result['samples'].transpose()):
    print('%15s : %.3f +- %.3f' % (name, col.mean(), col.std()))

# make marginal plots by running:
# $ python multinest_marginals.py chains/ring0-
# For that, we need to store the parameter names:
import json
with open('%sparams.json' % prefix, 'w') as f:
    json.dump(parameters, f, indent=2)