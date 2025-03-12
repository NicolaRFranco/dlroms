import matplotlib.pyplot as plt
from IPython.display import clear_output as clc
from fenics import*
from dlroms import*
import numpy as np

mesh = fe.unitsquaremesh(20, 20)
Vh = fe.space(mesh, 'CG', 1, vector_valued = True)
clc()

parameters = {'Parameter':['rho', 'lambda', 'nu', 'm', 'delta', 'x0', 'theta'],
              'Min': [0.5, 0.9, 0.5, 1.0, 0.05, 0.3, np.pi/4.0],
              'Max': [2.0, 1.1, 1.0, 1.5, 0.10, 0.7, 3*np.pi/4.0],
              'Meaning': ['Body force (shield)', 'First Lamé parameter (shield)', 
                          'Second Lamé parameter (shield)', 'Mass (hail grain)', 'Diameter (hail grain)',
                         'Impact location', 'Angle of impact']}

def FOMsolver(**kwargs):
  mu = kwargs['mu']
  rho, lambda_, nu, mass, delta, x0, theta = mu

  from ufl_legacy import nabla_div
  from scipy.sparse.linalg import spsolve
  from scipy.sparse import csr_matrix

  # Boundary conditions
  tol = 1e-14
  def clamped_boundary(x, on_boundary):
      return on_boundary and x[1]<tol
  bc = DirichletBC(Vh, Constant((0, 0)), clamped_boundary)

  # Auxiliary definitions
  def epsilon(u):
      return 0.5*(nabla_grad(u) + nabla_grad(u).T)

  def sigma(u):
      return lambda_*nabla_div(u)*Identity(2) + 2*nu*epsilon(u)

  # Variational problem
  u = TrialFunction(Vh)
  v = TestFunction(Vh)
  f = Constant((0, -rho))
  T = fe.interpolate(lambda x: [mass*np.cos(theta)*(x[1] > 0.99)*(np.abs(x[0]-x0)<delta),
                                   -mass*np.sin(theta)*(x[1] > 0.99)*(np.abs(x[0]-x0)<delta)], Vh)

  a = inner(sigma(u), epsilon(v))*dx
  L = dot(f, v)*dx + dot(T, v)*ds

  # Assembling and adjusting
  A = assemble(a)
  F = assemble(L)
  bc.apply(A)
  bc.apply(F)

  A = csr_matrix(A.array())
  F = F[:]

  # Solving
  u = spsolve(A, F)
  clc()
  return u
