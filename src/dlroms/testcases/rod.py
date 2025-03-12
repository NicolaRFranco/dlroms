import matplotlib.pyplot as plt
from IPython.display import clear_output as clc
from dlroms import*
import numpy as np
import gdown

gdown.download(id = "1-YmekOC5d1ACFsCAO0IEg5AuWji03SxF", output = "rod.xml", quiet = True)
mesh = fe.loadmesh("rod.xml")
Vh = fe.space(mesh, 'CG', 1)
clc()

parameters = {'Parameter':['T1', 'T2', 'T3', 'log10(f)', 'log(10)(alpha)'],
              'Min': [80, 80, 80, 1.5, -3.5],
              'Max': [120, 120, 120, 5, -1.5],
              'Meaning': ['Contact temperature']*3 + ['External heat source', 'Thermal radiation coefficient']}

def FOMsolver(**kwargs):
    mu = kwargs['mu']
    from fenics import Function, TestFunction, Constant, DirichletBC, dx, inner, grad, exp, solve
    
    # Portions of domain boundary subject to Dirichlet b.c.
    def leftCircle(x, on):
        return on and ((x[0]+2)**2 + x[1]**2)**0.5 < 0.5
    def rightCircle(x, on):
        return on and ((x[0]-2)**2 + x[1]**2)**0.5 < 0.8
    def rag(x, on):
        return on and (x[0]>-0.3) and (x[0]<0.3) and (x[1]>0)
    
    dbc1 = DirichletBC(Vh, Constant(mu[0]), leftCircle)
    dbc2 = DirichletBC(Vh, Constant(mu[1]), rightCircle)
    dbc3 = DirichletBC(Vh, Constant(mu[2]), rag)
    
    u, v = Function(Vh), TestFunction(Vh)
    L = inner((1000+exp(u/8.0))*grad(u), grad(v))*dx - Constant(10**mu[3])*v*dx + (10**mu[4])*(u**4)*v*dx
    clc()
    
    u.vector()[:] = (mu[0]+mu[1]+mu[2])/3.0 # Initial guess for nonlinear solver
    solve(L == 0, u, [dbc1, dbc2, dbc3])
    return u.vector()[:]
