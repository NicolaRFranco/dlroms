from fenics import *
import gdown
import numpy as np
from IPython.display import clear_output as clc
from dlroms import num2p
import dlroms.fespaces as fe

parameters = {'Parameter':['eps', 'rho'],
              'Min': [10.0**(-3.5), 0.5],
              'Max': [10.0**(-2.5), 1.0],
              'Meaning': ['Viscosity', 'Density']}

# Time discretization
T = 3.5                  
num_steps = int(1000*T)   
dt = T / num_steps        

# Load mesh and define function spaces
gdown.download(id = "1pkKBI4qGB2C6IGiXgAiY5CvZx0DEm88O", output = "nstokes_mesh.xml")
mesh = fe.loadmesh("nstokes_mesh.xml")
V = VectorFunctionSpace(mesh, 'P', 2)
Q = FunctionSpace(mesh, 'P', 1)
Vm = fe.space(mesh, 'CG', 2)
clc()

# Problem data
inflow   = 'near(x[0], 0)'
outflow  = 'near(x[0], 2.2)'
walls    = 'near(x[1], 0) || near(x[1], 0.41)'
cylinder = 'on_boundary && x[0]>0.1 && x[0]<0.3 && x[1]>0.1 && x[1]<0.3'

inflow_profile = ('4.0*1.5*x[1]*(0.41 - x[1]) / pow(0.41, 2)', '0')

bcu_inflow = DirichletBC(V, Expression(inflow_profile, degree=2), inflow)
bcu_walls = DirichletBC(V, Constant((0, 0)), walls)
bcu_cylinder = DirichletBC(V, Constant((0, 0)), cylinder)
bcp_outflow = DirichletBC(Q, Constant(0), outflow)
bcu = [bcu_inflow, bcu_walls, bcu_cylinder]
bcp = [bcp_outflow]

clc()

def FOMsolver(mu):
    eps, rho = mu
  
    # Trial and test functions
    u = TrialFunction(V)
    v = TestFunction(V)
    p = TrialFunction(Q)
    q = TestFunction(Q)

    # Define functions for solutions at previous and current time steps
    u_n = Function(V)
    u_  = Function(V)
    p_n = Function(Q)
    p_  = Function(Q)

    # Define expressions used in variational forms
    U  = 0.5*(u_n + u)
    n  = FacetNormal(mesh)
    f  = Constant((0, 0))
    k  = Constant(dt)
    eps = Constant(eps)
    rho = Constant(rho)

    # Define symmetric gradient
    def epsilon(u):
        return sym(nabla_grad(u))

    # Define stress tensor
    def sigma(u, p):
        return 2*eps*epsilon(u) - p*Identity(len(u))

    # Define variational problem for step 1
    F1 = (  rho*dot((u - u_n) / k, v)*dx 
          + rho*dot(dot(u_n, nabla_grad(u_n)), v)*dx 
          + inner(sigma(U, p_n), epsilon(v))*dx
          + dot(p_n*n, v)*ds - dot(eps*nabla_grad(U)*n, v)*ds
          - dot(f, v)*dx)
    a1 = lhs(F1)
    L1 = rhs(F1)

    # Define variational problem for step 2
    a2 = dot(nabla_grad(p), nabla_grad(q))*dx
    L2 = dot(nabla_grad(p_n), nabla_grad(q))*dx - (1/k)*div(u_)*q*dx

    # Define variational problem for step 3
    a3 = dot(u, v)*dx
    L3 = dot(u_, v)*dx - k*dot(nabla_grad(p_ - p_n), v)*dx

    # Assemble matrices
    A1 = assemble(a1)
    A2 = assemble(a2)
    A3 = assemble(a3)

    # Apply boundary conditions to matrices
    [bc.apply(A1) for bc in bcu]
    [bc.apply(A2) for bc in bcp]

    solution = []

    # Time-stepping
    t = 0
    solution.append(u_.vector()[:]+0.0)
    for n in range(num_steps):
        if(n%10==0 and verbose):
            clc(wait = True)
            print("Progress: %s." % num2p(n/num_steps))
        # Update current time
        t += dt

        # Step 1: Tentative velocity step
        b1 = assemble(L1)
        [bc.apply(b1) for bc in bcu]
        solve(A1, u_.vector(), b1, 'bicgstab', 'hypre_amg')

        # Step 2: Pressure correction step
        b2 = assemble(L2)
        [bc.apply(b2) for bc in bcp]
        solve(A2, p_.vector(), b2, 'bicgstab', 'hypre_amg')

        # Step 3: Velocity correction step
        b3 = assemble(L3)
        solve(A3, u_.vector(), b3, 'cg', 'sor')


        # Update previous solution
        u_n.assign(u_)
        p_n.assign(p_)
        solution.append(u_.vector()[:]+0.0)      
    if(verbose):
        clc()
    sol2 = [solution[i] for i in range(0, num_steps+1, 25)]
    umod = np.linalg.norm(np.stack(sol2).reshape(len(sol2), -1, 2), axis = -1)
    return umod
