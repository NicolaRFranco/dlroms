from cores import CPU, GPU
import fespaces
import dnns
import numpy as np
import torch
import dolfin

class Local(dnns.Sparse):
    def __init__(self, mesh1, mesh2, support, activation = dnns.leakyReLU):
        M = 0
        dim = len(mesh1.coordinates()[0])
        for j in range(dim):
            dj = mesh1.coordinates()[:,0].reshape(-1,1) -mesh2.coordinates()[:,0].reshape(1,-1)
            M = M + dj**2
        M = np.sqrt(M) < support
        super(Local, self).__init__(M, activation)

class Diagonal(dnns.Layer):
    def __init__(self, basein, baseout, svalues = None, activation = dnns.leakyReLU):
        super(Diagonal, self).__init__(activation)
        self.basein = CPU.tensor(basein)
        self.baseout = CPU.tensor(baseout)
        try:
            self.svalues = CPU.tensor(svalues).view(-1,1)
        except:
            self.svalues = CPU.tensor([1])
        self.weight = torch.nn.Parameter((CPU.zeros(len(basein)) + 1).view(-1,1))
        self.bias = torch.nn.Parameter(CPU.zeros(1, len(baseout[0])))      
        
    def moveOn(self, core):
        self.core = core
        with torch.no_grad():
            if(core == GPU):      
                self.weight = torch.nn.Parameter(self.weight.cuda())
                self.bias = torch.nn.Parameter(self.bias.cuda())
                self.svalues = self.svalues.cuda()
                self.basein = self.basein.cuda()
                self.baseout = self.baseout.cuda()
            else:
                self.weight = torch.nn.Parameter(self.weight.cpu())
                self.bias = torch.nn.Parameter(self.bias.cpu())
                self.svalues = self.svalues.cpu()
                self.basein = self.basein.cpu()
                self.baseout = self.baseout.cpu()
        
    def module(self):
        return self
    
    def forward(self, x):
        B1 = self.basein
        B2 = self.baseout
        W = self.weight*self.svalues
        return self.rho( (B1.mm(x.T)*W).T.mm(B2) + self.bias)
              

class Operator(dnns.Sparse):
    def __init__(self, matrix):
        matrix[np.abs(matrix)<1e-10] = 0
        super(Operator, self).__init__(matrix, None)
        self.load(matrix[np.nonzero(matrix)])
        self.freeze()
        
    def moveOn(self, core):
        super(Operator, self).moveOn(core)
        self.freeze()
        
class Gradient(Operator):
    def __init__(self, mesh):
        def perp(w):
            return np.stack((-w[:,1], w[:,0]), axis = 1)
        x, y = mesh.coordinates().T
        i,j,k = mesh.cells().T
        areas = 0.5*np.reshape(x[i]*y[j] - x[j]*y[i] + x[j]*y[k] - x[k]*y[j] + x[k]*y[i] - x[i]*y[k], (-1, 1))
        v = mesh.coordinates()
        nh = len(v)
        ne = len(i)
        vi, vj, vk = v[i], v[j], v[k]
        normals = perp(v[i]-v[k]), perp(v[j]-v[i])
        ci = -0.5*(normals[0]+normals[1])/areas
        cj = 0.5*normals[0]/areas
        ck = 0.5*normals[1]/areas
        ci = np.reshape(ci.T, (-1,1))
        cj = np.reshape(cj.T, (-1,1))
        ck = np.reshape(ck.T, (-1,1))
        Grad = np.zeros((len(ci), nh))
        Grad[np.arange(2*ne),np.concatenate((i,i))] = ci[:,0]
        Grad[np.arange(2*ne),np.concatenate((j,j))] = cj[:,0]
        Grad[np.arange(2*ne),np.concatenate((k,k))] = ck[:,0]
        super(Gradient, self).__init__(Grad.T)
        self.ne = ne
        
    def forward(self, x):
        res = super(Gradient, self).forward(x)
        return res.view(-1,2,self.ne) 
        
class Bilinear(Operator):
    def __init__(self, mesh, operator, obj = 'CG', degree = 1):
        W = dolfin.function.functionspace.FunctionSpace(mesh, obj, degree)
        if(degree == 1):
            perm = np.ndarray.astype(dolfin.cpp.fem.vertex_to_dof_map(W), 'int')
        else:
            perm = np.arange(mesh.num_cells())
        v1, v2 = dolfin.function.argument.TrialFunction(W), dolfin.function.argument.TestFunction(W)
        M = dolfin.fem.assembling.assemble(operator(v1, v2)).array()[:, perm][perm, :]
        super(Bilinear, self).__init__(M)
        
    def forward(self, x):
        return x[0].mm(self.W().mm(x[1].T))  
        
class Norm(Bilinear):
    def forward(self, x):
        return (x.mm(self.W())*x).sum(axis = -1).sqrt()   
        
class L2(Norm):
    def __init__(self, mesh, obj = 'CG', degree = 1):
        def operator(u,v):
            return u*v*dolfin.dx
        super(L2, self).__init__(mesh, operator, obj, degree)
    
class H1(Norm):
    def __init__(self, mesh, obj = 'CG', degree = 1):
        def operator(u,v):
            return u*v*dolfin.dx + dolfin.inner(dolfin.grad(u), dolfin.grad(v))*dolfin.dx
        super(H1, self).__init__(mesh, operator, obj, degree)
        
class Linf(dnns.Weightless):
    def forward(self, x):
        return x.abs().max(axis = -1)[0]
    
def projections(vs, hm, lm):
    vhs = [fespaces.asvector(v, hm) for v in vs]
    for vh in vhs:
        vh.set_allow_extrapolation(True)
    vls = [dolfin.fem.projection.project(vh, mesh = lm).compute_vertex_values(lm) for vh in vhs]
    return np.reshape(np.concatenate(vls), (-1, len(lm.coordinates())))
    
    
    
    
    
    