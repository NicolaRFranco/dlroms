import matplotlib.pyplot as plt
import numpy
import torch
import dolfin
import mshr
from dlroms import gifs
from ufl.finiteelement.mixedelement import VectorElement, FiniteElement
from ufl.finiteelement.enrichedelement import NodalEnrichedElement
from fenics import FunctionSpace
from fenics import Function

dx = dolfin.dx
ds = dolfin.ds
grad = dolfin.grad
inner = dolfin.inner
div = dolfin.div

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
            element = FiniteElement("CG", mesh.ufl_cell(), 1) +  FiniteElement("Bubble", mesh.ufl_cell(), mesh.topology().dim() + 1)
        else:
            element = FiniteElement("CG", mesh.ufl_cell(), 1)
    else:
        if(bubble):
            element = VectorElement(NodalEnrichedElement(FiniteElement("CG",       mesh.ufl_cell(), 1),
                                                         FiniteElement("Bubble",   mesh.ufl_cell(), mesh.topology().dim() + 1)))
        else:
            element = VectorElement("CG", mesh.ufl_cell(), 1)
    
    return FunctionSpace(mesh, element)

def coordinates(space):
    """Returns the coordinates of the degrees of freedom for the given functional space.
    
    Input
        space   (dolfin.function.functionspace.FunctionSpace).      Functional space for which the dofs have to be located.
        
    Output
        (numpy.ndarray).
    """
    return space.tabulate_dof_coordinates().astype("float32")

def boundary(mesh):
    """Returns the indexes of those nodes that lie on the boundary of a given domain.
    
    Input
        mesh    (dolfin.cpp.mesh.Mesh)   Underlying mesh.
    
    Output
        (numpy.ndarray). Ex: calling mesh.coordinates()[boundary(mesh)] returns the coordinates of the nodes along the boundary.
    """
    V = dolfin.function.functionspace.FunctionSpace(mesh, 'CG', 1)
    indexes = list(dolfin.fem.dirichletbc.DirichletBC(V, 0.0, lambda x,y:y).get_boundary_values().keys())
    return dolfin.cpp.fem.dof_to_vertex_map(V)[indexes]
    
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
        (mshr.cpp.Polygon).
    """
    return mshr.cpp.Polygon([point(p) for p in points])

def rectangle(p1, p2):
    """Creates a rectangle given two opposed vertices.
    
    Input
        p1  (tuple)     Coordinates of the first vertix.
        p2  (tuple)     Coordinates of the vertix opposed to p1.
        
    Output
        (mshr.cpp.Rectangle)
    """
    return mshr.cpp.Rectangle(point(p1), point(p2))

def circle(x0, r):
    """Creates a circle of given center and radius.
    
    Input
        x0  (tuple)     Coordinates of the center point.
        r   (float)     Radius.
        
    Output
        (mshr.cpp.Circle)
    """
    return mshr.cpp.Circle(point(x0), r)

def mesh(domain, resolution):
    """Discretizes a given domain using the specified resolution. Note: always results in unstructured triangular meshes.
    
    Input
        domain      (mshr.cpp.CSGGeometry)  Abstract domain. Can be obtained via calls such as fespaces.polygon, fespaces.rectangle, etc.
                                            Domains can also be defined as union (or difference) of simpler domains (simply by employing
                                            the + and - operators).
                        
        resolution  (int)                   Resolution level. Intuitively, doubling the resolution halfs the mesh stepsize.
        
    Output
        (dolfin.cpp.mesh.Mesh).   
        
    Remark: this method relies on the CGAL backend and is NOT deterministic. Running the same command may yields slightly different meshes.
    For better reproducibility, it is suggested to generate the mesh once and then rely on methods such as fespaces.save and fespaces.load.
    """
    return mshr.cpp.generate_mesh(domain, resolution = resolution)

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
                                                                    degrees of freedom. If u has shape (,n), then
                                                                    the functional space of interest should have n dof.
                                                                    
        space   (dolfin.function.functionspace.FunctionSpace).      Functional space where u belongs.
    
    Output    
        (dolfin.function.function.Function). 
        
    """
    uv = Function(space)
    udata = u if(not isinstance(u, torch.Tensor)) else u.detach().cpu().numpy()
    uv.vector()[:] = udata
    return uv
    
def plot(obj, space = None, vmin = None, vmax = None, colorbar = False, axis = "off", shrink = 0.8):
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
                c = dolfin.common.plotting.plot(uv, vmin = vmin, vmax = vmax)
            else:
                c = dolfin.common.plotting.plot(uv)
            if(colorbar):
                plt.colorbar(c, shrink = shrink)
    except:
        raise RuntimeError("First argument should be either a dolfin.cpp.mesh.Mesh or a structure containing the dof values of some function (in which case 'space' must be != None).")
    plt.axis(axis)
    
def gif(name, U, dt, T, space, axis = "off", figsize = (4,4)):
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
    frames = int(T/dt)
    N = len(U)
    step = N//frames
    def drawframe(i):
        plt.figure(figsize = figsize)
        plot(U[i*step], space, axis = axis)
    gifs.save(drawframe, frames, name)
   

def dbc(expression, where, space, degree = 1):
    from fenics import DirichletBC, Expression
    return DirichletBC(space, Expression(expression, degree = degree), where)
