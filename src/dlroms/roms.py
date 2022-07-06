import numpy as np
from scipy.linalg import eigh
from scipy.linalg import solve as scisolve
import torch
from dlroms.cores import coreof, CPU, GPU


def POD(U, k):
    """Principal Orthogonal Decomposition of the snapshots matrix U into k modes."""
    if(isinstance(U, torch.Tensor)):
        U0 = U.cpu().numpy()
    else:
        U0 = U
    M = np.dot(U0, U0.T)
    N = U.shape[0]
    w, v = eigh(M, eigvals = (N-k, N-1))
    basis, eigenvalues = np.dot((v/np.sqrt(w)).T, U0), w
    basis, eigenvalues = np.flip(basis, axis = 0), np.flip(eigenvalues)
    if(isinstance(U, torch.Tensor)):
        core = coreof(U)
        return core.tensor(basis), core.tensor(eigenvalues)
    else:
        return basis, eigenvalues

def num2p(prob):
    """Converts a number to percentage format."""
    return ("%.2f" % (100*prob)) + "%"

def projectdown(vbasis, u):
    """Given a sequence of basis vbasis = [V1,..., Vk], where Vj has shape (b, Nh), and
    a sequence of vectors u = [u1,...,uk], where uj has length Nh, yields the batched
    matrix vector multiplication [Vjuj], i.e. the sequence of basis coefficients."""
    nh = len(u[0])
    return vbasis.matmul(utrue.reshape(-1,nh,1))

def projectup(vbasis, ucoeff):
    n = len(ucoeff)
    nb = len(ucoeff[0])
    return vbasis.transpose(dim0 = 1, dim1 = 2).matmul(ucoeff.reshape(-1,nb,1)).reshape(n, -1)

def project(vbasis, utrue):
    return projectup(vbasis, projectdown(vbasis, utrue))
