from dlroms import fespaces as fe
    
def NavierStokes(mesh, nu, flow_bcs, pressure_bcs = dict(), pressure_space = None, flow_space = None):
    from fenics import split, dot, FiniteElement, VectorElement, NodalEnrichedElement, DirichletBC, FunctionSpace
    from fenics import inner, split, TestFunctions, Function, dot, grad, dx, div, solve
    import numpy as np
    
    pspace  = FiniteElement("CG", mesh.ufl_cell(), 1)
    vspace = VectorElement(NodalEnrichedElement(FiniteElement("CG",       mesh.ufl_cell(), 1),
                                                FiniteElement("Bubble",   mesh.ufl_cell(), mesh.topology().dim() + 1)))
    W = FunctionSpace(mesh, vspace*pspace)
    bcs = ([DirichletBC(W.sub(0), flow_bcs[where], where) for where in flow_bcs.keys()] +
           [DirichletBC(W.sub(1), pressure_bcs[where], where) for where in pressure_bcs.keys()]
          )

    v, q = TestFunctions(W)
    w = Function(W)
    u, p = split(w)
    F = nu*inner(grad(u), grad(v))*dx + dot(dot(grad(u), u), v)*dx \
        - p*div(v)*dx - q*div(u)*dx

    solve(F == 0, w, bcs)
    
    Vp = fe.space(mesh, 'CG', 1) if pressure_space is None else pressure_space
    Vu = fe.space(mesh, 'CG', 1, scalar = False, bubble = True) if flow_space is None else flow_space

    w.set_allow_extrapolation(True)
    p = np.array([w(x)[2] for x in fe.coordinates(Vp)])
    u = np.array([w(x)[:2] for x in fe.coordinates(Vu)[::2]]).reshape(-1)

    return u, p

def AdvectionDiffusionReaction(a, b, c, f = None, bcs = dict(), flux = dict(), space = None):
    from fenics import split, dot, FiniteElement, VectorElement, NodalEnrichedElement, DirichletBC, FunctionSpace
    from fenics import inner, TestFunction, TrialFunction, Function, dot, grad, dx, ds, div, solve, Constant
    import numpy as np
    
    V = space
    bcs = [DirichletBC(V, bcs[where], where) for where in bcs.keys()]

    
    ff = f if (not (f is None)) else Constant(0.0)
    phi = np.zeros(V.dim())
    boundary = DirichletBC(V, 0.0, lambda x, on: on).get_boundary_values().keys()
    x = fe.coordinates(V)
    for i in boundary:
        for where in flux.keys():
            if(where(x[i], True)):
                phi[i] += flux[where][i]    
    phi = fe.asvector(phi, V)
    
    u = TrialFunction(V)
    v = TestFunction(V)
    
    A = inner(a*grad(u), grad(v))*dx + inner(grad(u), b)*v*dx + c*u*v*dx
    F = ff*v*dx + phi*v*ds

    u = Function(V)
    solve(A == F, u, bcs)
    
    return u.vector()[:]
