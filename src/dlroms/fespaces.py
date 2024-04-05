# Written by: Nicola Rares Franco, Ph.D. (MOX, Department of Mathematics, Politecnico di Milano)
# 
# Scientific articles based on this Python package:
#
# [1] Franco et al., Mathematics of Computation (2023).
#     A deep learning approach to reduced order modelling of parameter dependent partial differential equations.
#     DOI: https://doi.org/10.1090/mcom/3781.
#
# [2] Franco et al., Neural Networks (2023).
#     Approximation bounds for convolutional neural networks in operator learning.
#     DOI: https://doi.org/10.1016/j.neunet.2023.01.029
#
# [3] Franco et al., Journal of Scientific Computing (2023). 
#     Mesh-Informed Neural Networks for Operator Learning in Finite Element Spaces.
#     DOI: https://doi.org/10.1007/s10915-023-02331-1
#
# [4] Vitullo, Colombo, Franco et al., Finite Elements in Analysis and Design (2024).
#     Nonlinear model order reduction for problems with microstructure using mesh informed neural networks.
#     DOI: https://doi.org/10.1016/j.finel.2023.104068
#
# Please cite the Author if you use this code for your work/research.

import matplotlib.pyplot as plt
import numpy
import torch
import warnings
from dlroms import gifs
from dlroms.cores import coreof
try:
    import dolfin
    try:
        from ufl.finiteelement.mixedelement import VectorElement, FiniteElement
        from ufl.finiteelement.enrichedelement import NodalEnrichedElement
    except:
        from ufl_legacy.finiteelement.mixedelement import VectorElement, FiniteElement
        from ufl_legacy.finiteelement.enrichedelement import NodalEnrichedElement
    from fenics import FunctionSpace
    from fenics import Function
    from fenics import set_log_active    
    set_log_active(False)

    dx = dolfin.dx
    ds = dolfin.ds
    grad = dolfin.grad
    inner = dolfin.inner
    div = dolfin.div
except:
    warnings.warn("Either dolfin or fenics are not available. Some functions might not be available or work as expected.")


def space(mesh, obj, deg, scalar = True, bubble = False):
    """Returns the Finite Element (FE) space of specified type (e.g. continuous/discontinuous galerkin) and degree.
    Note: only constructs FE spaces of scalar-valued functions.
    
    Input
        mesh    (dolfin.cpp.mesh.Mesh)  Underlying mesh of reference
        obj     (str)                   Type of space. 'CG' = Continuous Galerkin, 'DG' = Discontinuous Galerkin
        deg     (int)                   Polynomial degree at each element
        scalar  (bool)                  Whether the space consists of scalar or vector-valued functions (in which
                                        case scalar == True and scalar == False respectively). Defaults to True.
        bubble  (bool)                  If True, enriches each element with bubble polynomials. Defaults to False.
        
    Output
        (dolfin.function.functionspace.FunctionSpace).
    """
    if(scalar):
        if(bubble):
            element = FiniteElement(obj, mesh.ufl_cell(), deg) +  FiniteElement("Bubble", mesh.ufl_cell(), mesh.topology().dim() + 1)
        else:
            element = FiniteElement(obj, mesh.ufl_cell(), deg)
    else:
        if(bubble):
            element = VectorElement(NodalEnrichedElement(FiniteElement(obj,       mesh.ufl_cell(), deg),
                                                         FiniteElement("Bubble",   mesh.ufl_cell(), mesh.topology().dim() + 1)))
        else:
            element = VectorElement(obj, mesh.ufl_cell(), deg)
    
    return FunctionSpace(mesh, element)

def embedd(u, oldspace, newspace):
    """Returns a new dof representation of a given functional object (NB: works only for FE spaces written in terms of a nodal basis)
    
    Input
        u          (numpy.ndarray or torch.Tensor)                     Vector containing the dofs of the functional object (w.r.t. the nodal basis of "oldspace").
                                                                       If u is 2-dimensional, the algorithm is applied batchwise.
        oldspace   (dolfin.function.functionspace.FunctionSpace).      Functional space of reference
        newspace   (dolfin.function.functionspace.FunctionSpace).      New functional space where to embedd u
        
    Output
        (numpy.ndarray) dofs of u in the new ambient space.
    """
    if(len(u.shape)==1):
        uu = asvector(u, oldspace)
        uu.set_allow_extrapolation(True)
        unew = [uu(z) for z in coordinates(newspace)]
        return numpy.array(unew) if(not isinstance(u, torch.Tensor)) else coreof(u).tensor(unew)
    else:
        unew = [embedd(a, oldspace, newspace) for a in u]
        return numpy.stack(unew) if(not isinstance(u, torch.Tensor)) else torch.stack(unew)

def coordinates(space):
    """Returns the coordinates of the degrees of freedom for the given functional space.
    
    Input
        space   (dolfin.function.functionspace.FunctionSpace).      Functional space for which the dofs have to be located.
        
    Output
        (numpy.ndarray).
    """
    return space.tabulate_dof_coordinates().astype("float32")

def boundary(mesh = None, V = None):
    """Returns the indexes of those nodes that lie on the boundary of a given domain.
    
    Input
        mesh    (dolfin.cpp.mesh.Mesh)   Underlying mesh.
    
    Output
        (numpy.ndarray). Ex: calling mesh.coordinates()[boundary(mesh)] returns the coordinates of the nodes along the boundary.
    """
    space = V if(not (V is None)) else dolfin.function.functionspace.FunctionSpace(mesh, 'CG', 1)
    indexes = list(dolfin.fem.dirichletbc.DirichletBC(space, 0.0, lambda x,y:y).get_boundary_values().keys())
    return indexes if(not (V is None)) else dolfin.cpp.fem.dof_to_vertex_map(space)[indexes]
    
def closest(mesh, x):
    """Given a mesh and a point, returns the closest mesh vertix to that point.
    
    Input
        mesh    (dolfin.cpp.mesh.Mesh)   Underlying mesh.
        x       (tuple)                  Coordinates of the point.
        
    Output
        (int) index of the closest vertix. Its coordinates can obtained as mesh.coordinates()[closest(mesh, x)]
    """
    return numpy.argmin(numpy.sum((numpy.array(x)-mesh.coordinates())**2, axis = 1))

def loadmesh(path):
    """Loads a mesh in .xml format from the given path.
    
    Input
        path    (str)   Path where the .xml file is located.
        
    Output
        (dolfin.cpp.mesh.Mesh).
    """
    return dolfin.cpp.mesh.Mesh(path)
    
def savemesh(path, mesh):
    """Saves a given mesh into .xml format.
    
    Input
        path    (str)   Path where to save the mesh. The string should end with '.xml' for correct formatting.
        
    Output
        None.
    """
    if(".xml" not in path):
        raise RuntimeError("Wrong extension. Meshes can only be saved in .xml format")
    dolfin.cpp.io.File(path).write(mesh)
    
def point(p):
    """Creates a dolfin.Point object given its coordinates. Should be regarded as a private method.
    
    Input
        p   (tuple)  Coordinates of the point.
    
    Output
        (dolfin.cpp.geometry.Point).
    """
    return dolfin.cpp.geometry.Point(*tuple(p))

def polygon(points):
    """Creates a polygon given the specified list of points. The first and the last point in the list should coincide.
    
    Input
        points  (tuple)  Collection of points describing the polygon.
        
    Output
        (mshr.cpp.Polygon or dlroms.geometry.Polygon, depending on the package available).
    """
    try:
        from dlroms.geometry import Polygon
        return Polygon(*points)
    except ImportError:
        from mshr.cpp import Polygon
        return Polygon([point(p) for p in points])

def rectangle(p1, p2):
    """Creates a rectangle given two opposed vertices.
    
    Input
        p1  (tuple)     Coordinates of the first vertix.
        p2  (tuple)     Coordinates of the vertix opposed to p1.
        
    Output
        (mshr.cpp.Rectangle or dlroms.geometry.Rectangle, depending on the package available).
    """
    try:
        from dlroms.geometry import Rectangle
        return Rectangle(p1, p2)
    except ImportError:
        from mshr.cpp import Rectangle
        return Rectangle(point(p1), point(p2))

def circle(x0, r):
    """Creates a circle of given center and radius.
    
    Input
        x0  (tuple)     Coordinates of the center point.
        r   (float)     Radius.
        
    Output
        (mshr.cpp.Circle or dlroms.geometry.Circle, depending on the package available).
    """
    try:
        from dlroms.geometry import Circle
        return Circle(x0, r)
    except ImportError:
        from mshr.cpp import Circle
        return Circle(point(x0), r)

def mesh(domain, **kwargs):
    """Discretizes a given domain using the specified resolution. Note: structured meshes are only available if gmsh is installed.
    
    Input
        domain      (mshr.cpp.CSGGeometry or dlroms.geometry.Domain)  Abstract domain. Can be obtained via calls such as fespaces.polygon,
                                                                      fespaces.rectangle, etc. Domains can also be defined as union (or 
                                                                      difference) of simpler domains (simply by employing the + and -
                                                                      operators).
        Available keyword argument:                
        resolution  (int)                                             Resolution level. Intuitively, doubling the resolution halfs the
                                                                      mesh stepsize. Only used if gmsh is not available but mshr is.
                                                                      
        stepsize    (float)                                           Mesh stepsize. Only used gmsh is available.

        structured  (bool)                                            Whether to build a structured mesh or not (if possible).
                                                                      Only used gmsh is available.        
        
    Output
        (dolfin.cpp.mesh.Mesh).   
        
    Remark: this method relies on the CGAL backend and is NOT deterministic. Running the same command may yields slightly different meshes.
    For better reproducibility, it is suggested to generate the mesh once and then rely on methods such as fespaces.save and fespaces.load.
    """
    try:
        from dlroms.geometry import mesh as generate_mesh
        structured = True
        if('structured' in kwargs.keys()):
            structured = kwargs['structured']
        return generate_mesh(domain, stepsize = kwargs['stepsize'], structured = structured)
    except ImportError:
        from mshr.cpp import generate_mesh
        return generate_mesh(domain, resolution = kwargs['resolution'])        

def unitsquaremesh(n, ny = None):
    """Yields a structured triangular (rectangular) mesh on the unit square [0,1]^2.
    
    Input
        n   (int)   Number of intervals per edge, or along the x-axis if ny != None.
        ny  (int)   Number of intervals across the y-axis. Defaults to None, in
                    which case ny = n.
        
    Output
        (dolfin.cpp.mesh.Mesh).
    """
    n1 = n if (ny == None) else ny
    return dolfin.cpp.generation.UnitSquareMesh(n,n1)

def unitcubemesh(n, ny = None, nz = None):
    """Yields a structured triangular (rectangular) mesh on the unit cube [0,1]^3.
    
    Input
        n   (int)   Number of intervals per edge, or along the x-axis if either
                    ny!=None or nz!=None.
        ny  (int)   Number of intervals across the y-axis. Defaults to None, in
                    which case ny = n.
        nz  (int)   Number of intervals across the z-axis. Defaults to None, in
                    which case nz = n.
        
    Output
        (dolfin.cpp.mesh.Mesh).
    """
    n1 = n if (ny == None) else ny
    n2 = n if (nz == None) else nz
    return dolfin.cpp.generation.UnitCubeMesh(n,n1,n2)

def asvector(u, space):
    """Given a vector of dof values, returns the corresponding object in the functional space.
    
    Input
        u       (numpy.ndarray or torch.Tensor)                     Vector collecting the values of the function at the
                                                                    degrees of freedom. If u has shape (n,), then
                                                                    the functional space of interest should have n dof.
                                                                    
        space   (dolfin.function.functionspace.FunctionSpace).      Functional space to which u belongs.
    
    Output    
        (dolfin.function.function.Function). 
        
    """
    uv = Function(space)
    udata = u if(not isinstance(u, torch.Tensor)) else u.detach().cpu().numpy()
    uv.vector()[:] = udata
    return uv

def assemblegrad(mesh, nodal = False):
    V = space(mesh, 'CG', 1)
    D = space(mesh, 'DG', 0)
    A = numpy.zeros((D.dim(), 3, 3))
    A[:,:,:2] = mesh.coordinates()[mesh.cells().T].transpose(1,0,2)
    A[:,:, 2] = 1.0
    Ainv = numpy.linalg.inv(A)[:,:2,:]
    u = numpy.eye(V.dim())
    
    from fenics import vertex_to_dof_map, dof_to_vertex_map
    perm = vertex_to_dof_map(V)
    u1u2u3 = u[:, perm][:, mesh.cells().T].transpose(2,1,0)
    gradients = (Ainv@u1u2u3).transpose(1,0,2)
    
    if(nodal):
        perm = dof_to_vertex_map(V)
        M = numpy.zeros((D.dim(), V.dim()))
        M[numpy.arange(len(M)), mesh.cells().T] = 1
        M = (M/M.sum(axis = 0)).T
        M = M[perm, :]
        igrad = M@gradients
        return igrad
    else:
        return gradients

def vtk(u, space, filename):
    """Generates a VTK file (.vtu) for a given discretized function.
    
    Input
        u        (numpy.ndarray or torch.Tensor)                     Vector collecting the values of the function at the
                                                                     degrees of freedom. If u has shape (n,), then
                                                                     the functional space of interest should have n dof.
                                                                    
        space    (dolfin.function.functionspace.FunctionSpace)       Functional space to which u belongs.

        filename (str)                                               Name of the VTK file.
        
    """
    import os
    from fenics import File
    vtkfile = File("%s.pvd" % filename)
    vtkfile << asvector(u, space)
    os.remove("%s.pvd" % filename)
    os.rename("%s000000.vtu" % filename, "%s.vtu" % filename)
    
def plot(obj, space = None, vmin = None, vmax = None, colorbar = False, axis = "off", shrink = 0.8, levels = 200, cmap = 'jet', spaceticks = False):
    """Plots mesh and functional objects.
    
    Input
        obj         (dolfin.cpp.mesh.Mesh, numpy.ndarray or torch.Tensor)   Object to be plotted. It should be either a mesh
                                                                            or an array containing the values of some function
                                                                            at the degrees of freedom.
        space       (dolfin.function.functionspace.FunctionSpace)           Functional space where 'obj' belongs (assuming 'obj' is not a mesh).
                                                                            Defaults to None, in which case 'obj' is assumed to be a mesh.
        vmin        (float)                                                 If a colorbar is added, then the color legend is calibrated in such a way that vmin
                                                                            is considered the smallest value. Ignored if space = None.
        vmax        (float)                                                 Analogous to vmin.
        colorbar    (bool)                                                  Whether to add or not a colorbar. Ignored if len(*args)=1.
        axis        (obj)                                                   Axis specifics (cf. matplotlib.pyplot.axis). Defaults to "off", thus hiding the axis.
        shrink      (float)                                                 Shrinks the colorbar by the specified factor (defaults to 0.8). Ignored if colorbar = False.
    """
    try:
        if(space == None):
            dolfin.common.plotting.plot(obj)
        else:
            uv = asvector(obj, space)
            if(space.element().value_dimension(0) == 1):
                try:
                    c = dolfin.common.plotting.plot(uv, vmin = vmin, vmax = vmax, levels = numpy.linspace(float(obj.min()), float(obj.max()), levels), cmap = cmap)
                except:
                    c = dolfin.common.plotting.plot(uv, vmin = vmin, vmax = vmax, cmap = cmap)
            else:
                c = dolfin.common.plotting.plot(uv, cmap = cmap)
            if(colorbar):
                cbar = plt.colorbar(c, shrink = shrink)
                if(spaceticks):
                    cbar.set_ticks([round(tick, 2) for tick in numpy.linspace(cbar.vmin, cbar.vmax, 6)])
    except:
        raise RuntimeError("First argument should be either a dolfin.cpp.mesh.Mesh or a structure containing the dof values of some function (in which case 'space' must be != None).")
    plt.axis(axis)

def multiplot(vs, shape, space, size = 4, **kwargs):
    plt.figure(figsize = (shape[1]*size, shape[0]*size))
    for j in range(len(vs)):
        plt.subplot(shape[0], shape[1], j+1)
        plot(vs[j], space, **kwargs)
    
def gif(name, U, space, dt = None, T = None, axis = "off", figsize = (4,4), colorbar = False, cmap = 'jet'):
    """Builds a GIF animation given the values of a functional object at multiple time steps.
    
    Input
        name    (str)                                               Path where to save the GIF file.
        U       (numpy.ndarray or torch.Tensor)                     Array of shape (N,n). Each U[j] should contain the
                                                                    values of a functional object at its degrees of freedom.
        dt      (float)                                             Time step with each frame.
        T       (float)                                             Final time. The GIF will have int(T/dt) frames
        space   (dolfin.function.functionspace.FunctionSpace).      Functional space where the U[i]'s belong.
        axis    (obj)                                               Axis specifics (cf. matplotlib.pyplot.axis). Defaults to "off", thus hiding the axis.
        figsize (tuple)                                             Sizes of the window where to plot, width = figsize[0], height = figsize[1].
                                                                    See matplotlib.pyplot.plot.
    """
    frames = len(U) if(T is None) else int(T/dt)
    N = len(U)
    step = N//frames
    vmin = float(U.min())
    vmax = float(U.max())
    def drawframe(i):
        plt.figure(figsize = figsize)
        plot(U[i*step], space, axis = axis, vmin = vmin, vmax = vmax, colorbar = colorbar, cmap = cmap)
    gifs.save(drawframe, frames, name)
   
def animate(U, space, **kwargs):
    rnd = numpy.random.randint(50000)
    gif("temp%d-gif" % rnd, U, space, **kwargs)
    from IPython.display import Image, display
    display(Image("temp%d-gif.gif" % rnd), metadata={'image/gif': {'loop': True}})
    from os import remove
    remove("temp%d-gif.gif" % rnd)

def dbc(expression, where, space, degree = 1):
    from fenics import DirichletBC, Expression
    return DirichletBC(space, Expression(expression, degree = degree), where)
