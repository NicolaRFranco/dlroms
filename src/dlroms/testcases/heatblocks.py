from dlroms import fespaces as fe
from fenics import grad, inner, dx
from scipy.sparse.linalg import spsolve
from IPython.display import clear_output as clc
import numpy as np

### Problem parameters
parameters = {'Parameter':['log10(s1)', 'log10(s2)', 'log10(s3)'],
              'Min': [-1, -1, -1],
              'Max': [1,   1,  1],
              'Meaning': ['Conductivity jth block']*3}

### Problem geometry
domain = fe.rectangle((0, 0), (3, 1))
mesh = fe.mesh(domain, stepsize = 0.025*np.sqrt(5.0/4.0), structured = True)

### Finite Element spaces
Vh = fe.space(mesh, 'CG', 1) # solution space
Dh = fe.space(mesh, 'DG', 0) # auxiliary space for conductivity fields
        
indicator1 = lambda x: 0.0 + (x[0] < 1.0) + (x[0]>=1.0)*(x[1]<=0.65)*(x[1]>=0.35)*(x[0]<=1.25)
indicator3 = lambda x: 0.0 + (x[0] > 2.0) + (x[0]<=2.0)*(x[1]<=0.65)*(x[1]>=0.35)*(x[0]>1.75)
indicator2 = lambda x: 1.0 - indicator1(x) - indicator3(x)

s1 = fe.interpolate(indicator1, Dh)
s2 = fe.interpolate(indicator2, Dh)
s3 = fe.interpolate(indicator3, Dh)


M = fe.assemble(lambda u, v: u*v*dx, Vh)
A1 = fe.assemble(lambda u, v: inner(s1*grad(u), grad(v))*dx, Vh)
A2 = fe.assemble(lambda u, v: inner(s2*grad(u), grad(v))*dx, Vh)
A3 = fe.assemble(lambda u, v: inner(s3*grad(u), grad(v))*dx, Vh)
clc()

# Initial condition
u0f = fe.interpolate(lambda x: 30*indicator2(x), Vh) 
u0 = fe.dofs(u0f) # vector representation

### Definition of the FOM solver 
def FOMstep(u_now, dt, mu):
    """Evolves the system in time by 1 dt (via Backward Euler scheme).
    
    Input:
        u0     (numpy.ndarray)     solution at current time
        dt     (float)             time-step
        mu     (numpy.ndarray)     conductivity values in the 3 blocks (log10 scale)
                             
    Output:
       (numpy.ndarray) dof representation of the evolved solution.
    """
    th1, th2, th3 = 10.0**mu
    
    A = M + dt*th1*A1 + dt*th2*A2 + dt*th3*A3
    F = M @ u_now
    
    u_next = spsolve(A, F)
    return u_next


def FOMsolver(mu, steps = 50, dt = 0.001):
    """Given the conductivity coefficients, evolves the system in time, producing a full trajectory.
    
    Input:
        mu     (numpy.ndarray)     1d-array listing the three conductivity coefficients (log10 scale)
        steps  (float)             number of time steps for the rollout. Defaults to 50
        dt     (float)             time-step. Defaults to 0.001
                             
    Output:
       (numpy.ndarray) 2d-array listing solution values at all time-steps (each row contains the
                       dof representation of the solution at a given time instant).
    """
    
    # Initialization
    u = [u0] # list of states in time
    
    # Time loop
    for n in range(steps):
        u_now = u[-1]
        u_next = FOMstep(u_now, dt, mu)
        u.append(u_next)
        
    return np.stack(u)


### Data loader (allows to skip data collection, if necessary)
def loadData():
    import gdown
    gdown.download(id = "1YB9JzZ4sYLKUJd2qwDFdYB7YQv0GQAvq", output = "FOMdata.npz", quiet=False)
    data = np.load("FOMdata.npz")
    clc()
    return data['mu'], data['u']


### Auxiliary stuff (domain visualization)
def showdomain():
    import matplotlib.pyplot as plt
    plt.figure(figsize = (5, 1.75))
    plt.plot([0, 3, 3, 0, 0], [0, 0, 1, 1, 0], 'k')
    plt.fill([1.0, 1.0, 1.25, 1.25, 1.0, 1.0, 0, 0, 1], [0, 0.35, 0.35, 0.65, 0.65, 1.0, 1, 0, 0], 'k', alpha = 0.1)
    plt.fill([2.0, 2.0, 1.75, 1.75, 2.0, 2.0, 3, 3, 2], [0, 0.35, 0.35, 0.65, 0.65, 1.0, 1, 0, 0], 'k', alpha = 0.2)
    plt.fill([1.0, 1.0,  1.25, 1.25, 1.0,  1.0, 2.0, 2.0, 1.75, 1.75, 2.0,  2.0,  1.0],
             [0.0, 0.35, 0.35, 0.65, 0.65, 1.0, 1.0, 0.65, 0.65, 0.35, 0.35, 0.0, 0.0],
             'k', alpha = 0.05)
    plt.text(0, 1.08, "$\Omega$", fontsize = 14)
    plt.text(0.43, 0.45, "$\Omega_1$", fontsize = 11)
    plt.text(1.43, 0.45, "$\Omega_2$", fontsize = 11)
    plt.text(2.43, 0.45, "$\Omega_3$", fontsize = 11)
    plt.axis("off")
    plt.show()
