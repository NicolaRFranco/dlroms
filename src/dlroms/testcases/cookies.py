from dlroms import*
import numpy as np
from fenics import inner, grad, dx
from IPython.display import clear_output as clc
from scipy.sparse.linalg import spsolve

parameters = {'Parameter':['sqrt(s0)', 'sqrt(s1)', 'sqrt(s2)', 'sqrt(s3)'],
              'Min': [0, 0, 0, 0],
              'Max': [3, 3, 3, 3],
              'Meaning': ['Conductivity jth cookie']*4}

# Problem data and discretization
domain = fe.rectangle((0, 0), (1, 1))
mesh = fe.mesh(domain, stepsize = 0.01, structured = True)

Vh = fe.space(mesh, 'CG', 1)
Dh = fe.space(mesh, 'DG', 0)

def diskIndicator(xcenter, ycenter, radius):
  return lambda x: ((x[0]-xcenter)**2 + (x[1]-ycenter)**2)**0.5 < radius

centers = np.array([[0.25, 0.25],
                    [0.25, 0.75],
                    [0.75, 0.25],
                    [0.75, 0.75]])

# Pre-assembling the matrices A_out, A0, A1, A2, A3
a_out = lambda u, v: inner(0.01*grad(u), grad(v))*dx
A_out = fe.assemble(a_out, Vh)

Ajs = []
for j in range(4):
  sigmaj = diskIndicator(centers[j, 0], centers[j, 1], 0.15)
  sigmaj_h = fe.interpolate(sigmaj, Dh)

  aj = lambda u, v: inner(sigmaj_h*grad(u), grad(v))*dx
  Aj = fe.assemble(aj, Vh)

  Ajs.append(Aj)

A0, A1, A2, A3 = Ajs

# Pre-assembling the RHS
f = fe.interpolate(1.0, Vh)
F = lambda v: f*v*dx
fh = fe.assemble(F, Vh)

# Boundary conditions
boundary = lambda x: True
bc = fe.DirichletBC(boundary, 0.0)
fe.applyBCs(fh, Vh, bc)

clc()

def FOMsolver(mu):
  xi = mu**2
  Ah = A_out + xi[0]*A0 + xi[1]*A1 + xi[2]*A2 + xi[3]*A3
  Ah = fe.applyBCs(Ah, Vh, bc)

  uh = spsolve(Ah, fh)

  return uh
