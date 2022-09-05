import matplotlib.pyplot as plt
from dlroms.cores import CPU, GPU
from dlroms import fespaces
from dlroms import dnns
import numpy as np
import torch
import dolfin
from fenics import FunctionSpace
from scipy.sparse.csgraph import dijkstra

def area(P, A, B):
    x1, y1 = P.T.reshape(2,-1,1)
    x2, y2 = A.T
    x3, y3 = B.T
    return np.abs((x1*(y2 - y3) + x2*(y3 - y1) + x3*(y1 - y2))/2)

class Local(dnns.Sparse):
    def __init__(self, coordinates1, coordinates2, support, activation = dnns.leakyReLU):
        M = 0
        dim = len(coordinates1[0])
        for j in range(dim):
            dj = coordinates1[:,j].reshape(-1,1) - coordinates2[:,j].reshape(1,-1)
            M = M + dj**2
        M = np.sqrt(M) < support
        super(Local, self).__init__(M, activation)
        
class Navigator(object):
    def __init__(self, domain, mesh):        
        cells = mesh.cells()
        ne = len(cells)

        adj = np.zeros((ne, ne))
        A, B, C = cells.T
        A, B, C = mesh.coordinates()[A], mesh.coordinates()[B], mesh.coordinates()[C]
        P = (A+B+C)/3.0

        for i in range(ne):
            for j in range(i+1, ne):
                if(len( set(cells[i]).intersection(set(cells[j])) )>0):
                    adj[i,j] = np.linalg.norm(P[i]-P[j])
        
        adj = adj + adj.T
        self.A = A
        self.B = B
        self.C = C
        self.adj = adj
        self.cells = cells
        self.nodes = mesh.coordinates()
        self.D = dijkstra(csgraph = adj, directed = False)
        
    def plottri(self, k):
        v = self.cells[k]
        p = self.nodes[[v[0], v[1], v[2], v[0]]]
        plt.fill(p[:,0],p[:,1],'b')
            
    def finde(self, P):
        A, B, C = self.A, self.B, self.C
        x1, y1 = A.T
        x2, y2 = B.T
        x3, y3 = C.T
        tot = np.abs((x1*(y2 - y3) + x2*(y3 - y1) + x3*(y1 - y2))/2)
        diffs = np.abs(tot - area(P,A,B) - area(P,A,C) - area(P,B,C))
        return diffs.argmin(axis = 1)
    
class Geodesic(dnns.Sparse):
    def __init__(self, domain, coordinates1, coordinates2, support, accuracy = 1, activation = dnns.leakyReLU):
        navigator = Navigator(domain, fespaces.mesh(domain, resolution = accuracy))
        
        E1 = navigator.finde(coordinates1).reshape(-1,1)
        E2 = navigator.finde(coordinates2).reshape(1,-1)
        M = navigator.D[E1, E2]
        
        super(Geodesic, self).__init__(M <= support, activation)
        
    def He(self, seed = None):
        nw = len(self.weight)
        with torch.no_grad():
            self.weight = torch.nn.Parameter(torch.rand(nw)/np.sqrt(nw))

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
    def __init__(self, operator, space, vspace = None, bcs = []):
        space1 = space
        space2 = space if(vspace == None) else vspace 
        v1, v2 = dolfin.function.argument.TrialFunction(space1), dolfin.function.argument.TestFunction(space2)
        M = dolfin.fem.assembling.assemble(operator(v1, v2))
        for bc in bcs:
            bc.apply(M)
        super(Bilinear, self).__init__(M.array())
        
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
    def __init__(self, spacein, spaceout):
        fSpace = spaceout
        vSpace = spacein
        a, b, c = dolfin.function.argument.TrialFunction(vSpace), dolfin.function.argument.TestFunction(fSpace), dolfin.function.argument.TrialFunction(fSpace)
        A = dolfin.fem.assembling.assemble(dolfin.div(a)*b*dolfin.dx)
        A = A.array().T
        M = dolfin.fem.assembling.assemble(b*c*dolfin.dx).array()
        lumped = np.diag(1.0/np.sum(M, axis = 0))
        D = np.dot(A, lumped)
        super(Divergence, self).__init__(D) 
    
    
    
