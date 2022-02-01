from dlroms import fespaces
import dolfin
import numpy as np
import scipy.sparse.linalg as spla

class GaussianRandomField(object):
    
    def __init__(self, mesh, kernel, upto):
        self.cov = kernel
        self.n = upto
        self.singvalues, self.eigenfunctions = KarhunenLoeve(mesh, self.cov, self.n)
        self.singvalues = np.sqrt(self.singvalues)       
        
    def sample(self, seed, coeff = False):
        np.random.seed(seed)
        c = np.random.randn(self.n)
        v = np.dot(self.eigenfunctions, self.singvalues*c)
        if(coeff):
            return v, c
        else:
            return v
        
def KarhunenLoeve(mesh, covariance, nphis):
    
    def solve_covariance_EVP(cov, k):
        V = dolfin.function.functionspace.FunctionSpace(mesh, 'P', 1)
        u = dolfin.function.argument.TrialFunction(V)
        v = dolfin.function.argument.TestFunction(V)
    
        dof2vert = dolfin.cpp.fem.dof_to_vertex_map(V)
        coords = mesh.coordinates()
        coords = coords[dof2vert]
        M = dolfin.fem.assembling.assemble(u*v*dolfin.dx)
        M = M.array()

        L = coords.shape[0]
        c0 = np.repeat(coords, L, axis=0)
        c1 = np.tile(coords, [L,1])
        r = np.abs(np.linalg.norm(c0-c1, axis=1))
        C = cov(r)
        C.shape = [L,L]

        A = np.dot(M, np.dot(C, M))
        w, v = spla.eigsh(A, k, M)
        return w, v, V
    
    lambdas, phis, V  = solve_covariance_EVP(lambda r : covariance(r), k = nphis)
    inv = [(nphis-i-1) for i in range(nphis)]
    basis = phis[dolfin.cpp.fem.vertex_to_dof_map(V), :][:, inv]
    lamb = lambdas[inv]
    
    return lamb, basis
