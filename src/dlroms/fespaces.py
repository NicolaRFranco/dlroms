import matplotlib.pyplot as plt
import numpy
import torch
import dolfin
import mshr
from dlroms import gifs

dx = dolfin.dx
ds = dolfin.ds
grad = dolfin.grad
inner = dolfin.inner

def space(mesh, obj, deg):
    """Returns the Finite Element (FE) space of specified type (e.g. continuous/discontinuous galerkin) and degree.
    Note: only constructs FE spaces of scalar-valued functions.
    
    Input
        mesh    (dolfin.cpp.mesh.Mesh)  Underlying mesh of reference
        obj     (str)                   Type of space. 'CG' = Continuous Galerkin, 'DG' = Discontinuous Galerkin
        deg     (int)                   Polynomial degree at each element
        
    Output
        (dolfin.function.functionspace.FunctionSpace).
    """
    return dolfin.function.functionspace.FunctionSpace(mesh, obj, deg)

def dofmap(V, inverse = False):
    """Mesh vertices can be ordered geometrically or algebraically (which makes quadrature formulas more efficient).
    The present function provides a way to switch between the two.
    
    Input
        V       (dolfin.function.functionspace.FunctionSpace)   Functional space (see e.g. dlroms.fespaces.space) of P1 type.
                                                                It may describe scalar or vector fields, but they must be
                                                                Continuous Galerkin elements of degree 1.                                                          
        
        inverse (bool)                                          If True, yields a vector I such that I[j] is the index of
                                                                the jth quadrature node accordingly to the geometric ordering.
                                                                If False, I[j] is the index in 'quadrature ordering' of the
                                                                jth 'geometric' node.
    Output
        (numpy.ndarray).
    """
    if(inverse):
        return dolfin.cpp.fem.dof_to_vertex_map(V)
    else:
        return dolfin.cpp.fem.vertex_to_dof_map(V)

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

def unitsquaremesh(n):
    """Yields a structured triangular (rectangular) mesh on the unit square [0,1]^2.
    
    Input
        n   (int)   Number of intervals per edge.
        
    Output
        (dolfin.cpp.mesh.Mesh).
    """
    return dolfin.cpp.generation.UnitSquareMesh(n,n)

def unitcubemesh(n):
    """Yields a structured triangular (rectangular) mesh on the unit cube [0,1]^3.
    
    Input
        n   (int)   Number of intervals per edge.
        
    Output
        (dolfin.cpp.mesh.Mesh).
    """
    return dolfin.cpp.generation.UnitCubeMesh(n,n,n)

def asvector(u, mesh, obj = 'CG', deg = 1):
    """Given a vector of dof values, returns the corresponding object in the functional space.
    
    Input
        u       (numpy.ndarray or torch.Tensor)     Vector collecting the values of the function at the
                                                    degrees of freedom. If u has shape (,n), then
                                                    the functional space of interest should have n dof.
                                                    If u has space (k,n), then the vector field
                                                    having components u[0],..., u[k-1] is returned.
        mesh    (dolfin.cpp.mesh.Mesh)              Mesh of reference.
        obj     (str)                               Type of FE objects (cf. dlroms.fespaces.space).
        deg     (int)                               Polynomial degree at the mesh elements.
    
    Output    
        (dolfin.function.function.Function). 
        
    Ex: let x, y = mesh.coordinates().T. Then the vector u = x**2 - y has length
    n = mesh.num_vertices(), and contains the nodal values of the corresponding function f(a,b) = a**2 - b. The vector
    of values u can be converted into an element of the functional space P1-FE (Continuous Galerking elements of degree 1)
    via asvector(u, mesh).
    """
    if(isinstance(u, numpy.ndarray)):
        un = u
    elif(isinstance(u, torch.Tensor)):
        un = u.cpu().numpy()
    else:
        raise RuntimeError("Expected numpy.ndarray or torch.Tensor as first argument.")
    if(len(u.shape)==1):
        V = dolfin.function.functionspace.FunctionSpace(mesh, obj, deg)
        uv = dolfin.function.function.Function(V) 
        if(len(uv.vector()[:])!=len(u)):
            raise RuntimeError("Wrong length: the number of entries in u is not compatible with the specified FE object.")
        if(obj == 'CG' and deg == 1):
            uv.vector()[:] = un[dolfin.cpp.fem.dof_to_vertex_map(V)]
        else:
            uv.vector()[:] = un
        return uv
    else:
        ux, uy = u
        uvx, uvy = asvector(ux, mesh, obj, deg), asvector(uy, mesh, obj, deg)
        return dolfin.fem.projection.project(uvx*dolfin.function.constant.Constant((1.0, 0.0))+
                                             uvy*dolfin.function.constant.Constant((0.0, 1.0)),
                                             dolfin.function.functionspace.VectorFunctionSpace(mesh, obj, degree = deg, dim = 2))
    
def plot(*args, obj = 'CG', deg = 1, vmin = None, vmax = None, colorbar = False, axis = "off", shrink = 0.8):
    """Plots mesh and functional objects.
    
    Input
        *args       (tuple)     Two uses are available. One can pass a single object of dolfin.cpp.mesh.Mesh type: in
                                that case, the mesh is plotted. Conversely, one may pass a pair (a, b) where: a
                                is either a numpy.ndarray or torch.Tensor, while b is a dolfin.cpp.mesh.Mesh object.
                                In the latter case, the first argument should contain the values of some functional
                                object at the degrees of freedom; the second argument, should instead provide the
                                underlying mesh where the object is defined.
        obj         (str)       Type of functional object that is to be plotted (cf. dlroms.fespaces.space).
                                Ignored if len(*args)=1.
        deg         (int)       Polynomial degree at the mesh elements. Ignored if len(*args)=1.
        vmin        (float)     If a colorbar is added, then the color legend is calibrated in such a way that vmin
                                is considered the smallest value. Ignored if len(*args)=1.
        vmax        (float)     Analogous to vmin.
        colorbar    (bool)      Whether to add or not a colorbar. Ignored if len(*args)=1.
        axis        (obj)       Axis specifics (cf. matplotlib.pyplot.axis). Defaults to "off", thus hiding the axis.
        shrink      (float)     Shrinks the colorbar by the specified factor (defaults to 0.8). Ignored if colorbar = False.
    """
    if(len(args)==2):
        uv = asvector(*args, obj, deg)
        if(len(args[0].shape)==1):
            c = dolfin.common.plotting.plot(uv, vmin = vmin, vmax = vmax)
        else:
            c = dolfin.common.plotting.plot(uv)
        if(colorbar):
            plt.colorbar(c, shrink = shrink)
    elif(len(args)==1):
        if(isinstance(args[0], dolfin.cpp.mesh.Mesh)):
            dolfin.common.plotting.plot(args[0])
        else:
            raise RuntimeError("Cannot plot an object in the FE space if no mesh is provided")
    else:
        raise RuntimeError("Too much inputs. Expected at most two non-keyword arguments (data values and mesh).")
    plt.axis(axis)
    
def gif(name, U, dt, T, mesh, obj = 'CG', deg = 1, axis = "off", figsize = (4,4)):
    """Builds a GIF animation given the values of a functional object at multiple time steps.
    
    Input
        name    (str)                               Path where to save the GIF file.
        U       (numpy.ndarray or torch.Tensor)     Array of either shape (N,n) or (N,k,n). U[j] should contain the
                                                    values of a functional object at its degrees of freedom (in the
                                                    first case, U[j] is a scalar-valued map, in the second case it
                                                    is a vector field with k components; cf. dlroms.fespaces.asvector).
        dt      (float)                             Time step with each frame.
        T       (float)                             Final time. The GIF will have int(T/dt) frames
        mesh    (dolfin.cpp.mesh.Mesh)              Underlying mesh.
        obj     (str)                               Type of functional objects (cf. dlroms.fespaces.space).
        deg     (int)                               Polynomial degree at the mesh elements.
        axis    (obj)                               Axis specifics (cf. matplotlib.pyplot.axis). Defaults to "off", thus hiding the axis.
        figsize (tuple)                             Sizes of the window where to plot, width = figsize[0], height = figsize[1].
                                                    See matplotlib.pyplot.plot.
    """
    frames = int(T/dt)
    N = len(U)
    step = N//frames
    def drawframe(i):
        plt.figure(figsize = figsize)
        plot(u = U[i*step], mesh = mesh, obj = obj, deg = deg, axis = axis)
    gifs.save(drawframe, frames, name)
