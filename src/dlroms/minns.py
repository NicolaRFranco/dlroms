from dlroms.cores import CPU, GPU
from dlroms import fespaces
from dlroms import dnns
import numpy as np
import torch
import dolfin
from fenics import FunctionSpace

class Local(dnns.Sparse):
    def __init__(self, coordinates1, coordinates2, support, activation = dnns.leakyReLU):
        M = 0
        dim = len(coordinates1[0])
        for j in range(dim):
            dj = coordinates1[:,j].reshape(-1,1) - coordinates2[:,j].reshape(1,-1)
            M = M + dj**2
        M = np.sqrt(M) < support
        super(Local, self).__init__(M, activation)

class Operator(dnns.Sparse):
    def __init__(self, matrix):
        matrix[np.abs(matrix)<1e-10] = 0
        super(Operator, self).__init__(matrix, None)
        self.load(matrix[np.nonzero(matrix)])
        self.freeze()
        
    def moveOn(self, core):
        super(Operator, self).moveOn(core)
        self.freeze()
        
class Bilinear(Operator):
    def __init__(self, operator, space, vspace = None):
        space1 = space
        space2 = space if(vspace == None) else vspace 
        v1, v2 = dolfin.function.argument.TrialFunction(space1), dolfin.function.argument.TestFunction(space2)
        M = dolfin.fem.assembling.assemble(operator(v1, v2)).array()
        super(Bilinear, self).__init__(M)
        
    def forward(self, x):
        return x[0].mm(self.W().mm(x[1].T))  
        
class Norm(Bilinear):
    def forward(self, x):
        return (x.mm(self.W())*x).sum(axis = -1).sqrt()   
        
class L2(Norm):
    def __init__(self, space):
        def operator(u,v):
            return dolfin.inner(u, v)*dolfin.dx
        super(L2, self).__init__(operator, space)
    
class H1(Norm):
    def __init__(self, space):
        def operator(u,v):
            return dolfin.inner(u, v)*dolfin.dx + dolfin.inner(dolfin.grad(u), dolfin.grad(v))*dolfin.dx
        super(H1, self).__init__(operator, space)
        
class Linf(dnns.Weightless):
    def forward(self, x):
        return x.abs().max(axis = -1)[0]
    
class Integral(dnns.Dense):
    def __init__(self, space):
        v1, v2 = dolfin.function.argument.TrialFunction(space), dolfin.function.argument.TestFunction(space)
        M = dolfin.fem.assembling.assemble(dolfin.inner(v1,v2)*dolfin.dx).array()
        super(Integral, self).__init__(M.shape[0], 1, activation = None)
        self.zeros()
        self.load(np.sum(M, axis = 1).reshape(1,-1))
        self.freeze()
        
    def moveOn(self, core):
        super(Integral, self).moveOn(core)
        self.freeze()
        
class L1(Integral):
    def forward(self, x):
        return super(L1, self).forward(x.abs())
    
    
class Divergence(Operator):
    def __init__(self, space, mesh):
        fSpace = FunctionSpace(mesh, 'DG', 0)
        vSpace = space
        a, b, c = dolfin.function.argument.TrialFunction(vSpace), dolfin.function.argument.TestFunction(fSpace), dolfin.function.argument.TrialFunction(fSpace)
        A = dolfin.fem.assembling.assemble(b*dolfin.div(a)*dolfin.dx).array()
        M = dolfin.fem.assembling.assemble(b*c*dolfin.dx).array()
        lumped = np.diag(1.0/np.sum(M, axis = 0))
        D = np.dot(lumped, A)
        super(Divergence, self).__init__(D.T)
    
    def forward(self, x):
        X = x.transpose(dim0 = 1, dim1 = 2).reshape(len(x), -1)
        return X.mm(self.W())
    
    
    
    
