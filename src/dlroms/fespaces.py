import matplotlib.pyplot as plt
import numpy
import torch
import dolfin
import mshr
from dlroms import gifs

def loadmesh(path):
    return dolfin.cpp.mesh.Mesh(path)
    
def savemesh(path, mesh):
    dolfin.cpp.io.File(path).write(mesh)
    
def point(p):
    return dolfin.cpp.geometry.Point(*tuple(p))

def polygon(points):
    return mshr.cpp.Polygon([point(p) for p in points])

def rectangle(p1, p2):
    return mshr.cpp.Rectangle(point(p1), point(p2))

def circle(x0, r):
    return mshr.cpp.Circle(point(x0), r)

def mesh(domain, resolution):
    return mshr.cpp.generate_mesh(domain, resolution = resolution)

def unitsquaremesh(n):
    return dolfin.cpp.generation.UnitSquareMesh(n,n)

def unitcubemesh(n):
    return dolfin.cpp.generation.UnitCubeMesh(n,n,n)

def asvector(u, mesh, obj = 'CG', deg = 1):
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
        ux, uy = data
        uvx, uvy = asvector(ux, mesh, obj, deg), asvector(uy, mesh, obj, deg)
        return dolfin.fem.projection.project(uvx*dolfin.function.constant.Constant((1.0, 0.0))+
                                             uvy*dolfin.function.constant.Constant((0.0, 1.0)),
                                             dolfin.function.functionspace.VectorFunctionSpace(mesh, obj, degree = deg, dim = 2))
    
def plot(*args, obj = 'CG', deg = 1, vmin = None, vmax = None, colorbar = False, axis = "off", shrink = 0.8):
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
        frames = int(T/dt)
        N = len(U)
        step = N//frames
        def drawframe(i):
            plt.figure(figsize = figsize)
            plot(u = U[i*step], mesh = mesh, obj = obj, deg = deg, axis = axis)
        gifs.save(drawframe, frames, name)
