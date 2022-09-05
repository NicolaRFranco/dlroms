import matplotlib.pyplot as plt
import numpy
import torch
import dolfin
from dlroms import gifs
from ufl.finiteelement.mixedelement import VectorElement, FiniteElement
from ufl.finiteelement.enrichedelement import NodalEnrichedElement
from fenics import FunctionSpace
from fenics import Function
from dlroms.geometry import Rectangle as rectangle
from dlroms.geometry import Circle as circle
from dlroms.geometry import Box as box
from dlroms.geometry import Polygon as polygon
from dlroms.geometry import mesh

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
    
def plot(obj, space = None, vmin = None, vmax = None, colorbar = False, axis = "off", shrink = 0.8, levels = 200, cmap = None):
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
                c = dolfin.common.plotting.plot(uv, vmin = vmin, vmax = vmax, levels = numpy.linspace(float(obj.min()), float(obj.max()), levels), cmap = cmap)
            else:
                c = dolfin.common.plotting.plot(uv, cmap = cmap)
            if(colorbar):
                plt.colorbar(c, shrink = shrink)
    except:
        raise RuntimeError("First argument should be either a dolfin.cpp.mesh.Mesh or a structure containing the dof values of some function (in which case 'space' must be != None).")
    plt.axis(axis)
    
def gif(name, U, dt, T, space, axis = "off", figsize = (4,4), colorbar = False):
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
    vmin = U.min()
    vmax = U.max()
    def drawframe(i):
        plt.figure(figsize = figsize)
        plot(U[i*step], space, axis = axis, vmin = vmin, vmax = vmax, colorbar = colorbar)
    gifs.save(drawframe, frames, name)
   

def dbc(expression, where, space, degree = 1):
    from fenics import DirichletBC, Expression
    return DirichletBC(space, Expression(expression, degree = degree), where)
