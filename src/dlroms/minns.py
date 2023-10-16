# Written by: Nicola Rares Franco, Ph.D. (MOX, Department of Mathematics, Politecnico di Milano)
# 
# Scientific articles based on this Python package:
# [1] Franco et al. (2023). A deep learning approach to reduced order modelling of parameter dependent partial differential equations
#     DOI: https://doi.org/10.1090/mcom/3781
# [2] Franco et al. (2023). Approximation bounds for convolutional neural networks in operator learning, Neural Networks.
#     DOI: https://doi.org/10.1016/j.neunet.2023.01.029
# [3] Franco et al. (2023). Mesh-Informed Neural Networks for Operator Learning in Finite Element Spaces, Journal of Scientific Computing.
#     DOI: https://doi.org/10.1007/s10915-023-02331-1
#
# Please cite the Author if you use this code for your work/research.

import matplotlib.pyplot as plt
from dlroms.cores import CPU, GPU
from dlroms import dnns
import numpy as np
import torch
from scipy.sparse.csgraph import dijkstra

def area(P, A, B):
    x1, y1 = P.T.reshape(2,-1,1)
    x2, y2 = A.T
    x3, y3 = B.T
    return np.abs((x1*(y2 - y3) + x2*(y3 - y1) + x3*(y1 - y2))/2)

class Local(dnns.Sparse):
    def __init__(self, x1, x2, support, activation = dnns.leakyReLU):
        from dlroms.fespaces import coordinates
        coordinates1 = x1 if(isinstance(x1, np.ndarray)) else coordinates(x1)
        coordinates2 = x2 if(isinstance(x2, np.ndarray)) else coordinates(x2)
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
    def __init__(self, domain, x1, x2, support, accuracy = 1, activation = dnns.leakyReLU):
        from dlroms.fespaces import coordinates, mesh
        coordinates1 = x1 if(isinstance(x1, np.ndarray)) else coordinates(x1)
        coordinates2 = x2 if(isinstance(x2, np.ndarray)) else coordinates(x2)
        navigator = Navigator(domain, mesh(domain, resolution = accuracy))
        
        E1 = navigator.finde(coordinates1).reshape(-1,1)
        E2 = navigator.finde(coordinates2).reshape(1,-1)
        M = navigator.D[E1, E2]
        
        super(Geodesic, self).__init__(M <= support, activation)

class Operator(dnns.Sparse):
    def __init__(self, matrix):
        matrix[np.abs(matrix)<1e-10] = 0
        super(Operator, self).__init__(matrix, None)
        self.load(matrix[np.nonzero(matrix)])
        self.freeze()
        
    def moveOn(self, core):
        super(Operator, self).moveOn(core)
        self.freeze()
    
    def He(self, linear = False, a = 0.1, seed = None):
        None
        
class Bilinear(Operator):
    def __init__(self, operator, space, vspace = None, bcs = []):
        from dolfin.function.argument import TrialFunction, TestFunction
        from dolfin.fem.assembling import assemble
        space1 = space
        space2 = space if(vspace == None) else vspace 
        v1, v2 = TrialFunction(space1), TestFunction(space2)
        M = assemble(operator(v1, v2))
        for bc in bcs:
            bc.apply(M)
        super(Bilinear, self).__init__(M.array())        
        self.M = torch.sparse_coo_tensor(np.stack(self.loc, axis = 0), self.weight.detach().cpu().numpy(), (self.in_d, self.in_d))
        
    def W(self):
        return self.M
        
    def dualize(self, x):
        return torch.sparse.mm(self.M, x.T).T
    
    def cuda(self):
        self.M = self.M.cuda()

    def forward(self, x1, x2):
        return self.dualize(x1).mm(x2.T)
        
class Norm(Bilinear):
    def forward(self, x, squared = False):
        y = (self.dualize(x)*x).sum(axis = -1)
        return y if squared else y.sqrt() 
        
class L2(Norm):
    def __init__(self, space):
        def operator(u,v):
            from dolfin import inner, dx
            return inner(u, v)*dx
        super(L2, self).__init__(operator, space)
    
class H1(Norm):
    def __init__(self, space):
        def operator(u,v):
            from dolfin import inner, dx, grad
            return inner(u, v)*dx + inner(grad(u), grad(v))*dx
        super(H1, self).__init__(operator, space)
        
class Linf(dnns.Weightless):
    def forward(self, x):
        return x.abs().max(axis = -1)[0]
    
class Integral(dnns.Dense):
    def __init__(self, space):
        from dolfin.function.argument import TrialFunction, TestFunction
        from dolfin.fem.assembling import assemble
        from dolfin import inner, dx
        v1, v2 = TrialFunction(space), TestFunction(space)
        M = assemble(inner(v1,v2)*dx).array()
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
        from dolfin.function.argument import TrialFunction, TestFunction
        from dolfin.fem.assembling import assemble
        from dolfin import div, dx
        a, b, c = TrialFunction(vSpace), TestFunction(fSpace), TrialFunction(fSpace)
        A = assemble(div(a)*b*dx)
        A = A.array().T
        M = assemble(b*c*dx).array()
        lumped = np.diag(1.0/np.sum(M, axis = 0))
        D = np.dot(A, lumped)
        super(Divergence, self).__init__(D) 
    
    
    
