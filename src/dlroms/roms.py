import numpy as np
from scipy.linalg import eigh
from scipy.linalg import solve as scisolve
import torch
from dlroms.cores import coreof, CPU, GPU
from dlroms.dnns import Consecutive, Clock

mre = lambda norm: lambda utrue, upred: (norm(utrue-upred)/norm(utrue)).mean()
mse = lambda norm: lambda utrue, upred: norm(utrue-upred, squared = True).mean()

def snapshots(n, sampler, core = GPU):
    """Samples a collection of snapshots for a given FOM solver."""
    mu, u = [], []
    for seed in range(n):
        mu0, u0 = sampler(seed)
        mu.append(mu0)
        u.append(u0)
    mu, u = np.stack(mu), np.stack(u)
    return core.tensor(mu, u)

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
    basis, eigenvalues = np.flip(basis, axis = 0)+0, np.flip(eigenvalues)+0
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
    if(len(vbasis.shape)<3):
      return projectdown(vbasis.unsqueeze(0), u)
    else:
      nh = np.prod(u[0].shape)
      n, nb = vbasis.shape[:2]
      return vbasis.reshape(n, nb, -1).matmul(u.reshape(-1,nh,1))

def projectup(vbasis, c):
    """Given a sequence of basis vbasis = [V1,..., Vk], where Vj has shape (b, Nh), and
    a sequence of coefficients c = [c1,...,ck], where cj has length b, yields the batched
    matrix vector multiplication [Vj.Tcj], i.e. the sequence of expanded vectors."""
    if(len(vbasis.shape)<3):
      return projectup(vbasis.unsqueeze(0), c)
    else:
      b = c.shape[1]
      n, nb = vbasis.shape[:2]
      return vbasis.reshape(n, nb, -1).transpose(dim0 = 1, dim1 = 2).matmul(c.reshape(-1,b,1)).reshape(-1, vbasis.shape[-1])

def project(vbasis, u, orth = True):
    """Given a sequence of basis vbasis = [V1,..., Vk], where Vj has shape (b, Nh), and
    a sequence of vectors u = [u1,...,uk], where uj has length Nh, yields the batched
    matrix vector multiplication [Vj'Vjuj], i.e. the sequence of reconstructed vectors."""
    if(len(vbasis.shape)<3):
        return project(vbasis.unsqueeze(0), u, orth)
    else:
        if(orth):
            return project(gramschmidt(vbasis.transpose(1,2)).transpose(2,1), u, orth = False)
        else:
            return projectup(vbasis, projectdown(vbasis, u))

def gramschmidt(V):
    """Orthonormalizes a collection of matrices. V should be a torch tensor in the format batch dimension x space dimension x number of basis."""
    return torch.linalg.qr(V, mode = 'reduced')[0]

def PAs(V1, V2, orth = True):
    """List of principal angles between the subspaces in V1 and V2. The Vj's should be in the format
    batch dimension x space dimension x number of basis."""
    if(orth):
        A1, A2 = gramschmidt(V1), gramschmidt(V2)
    else:
        A1, A2 = V1, V2
        
def PODerrors(u, upto, ntrain, error):
    """Projection errors over the test set for an increasing number of modes."""
    pod, svalues = POD(u[:ntrain], k = upto)
    errors = []
    for n in range(1, upto+1):
        uproj = project(pod[:n], u[ntrain:])
        errors.append(error(u[ntrain:], uproj))
    return errors
    vals = torch.linalg.svdvals(A1.transpose(dim0 = 1, dim1 = 2).matmul(A2)).clamp(min=0,max=1)
    return vals.arccos()


class ROM(Consecutive):
    """Abstract class for handling Reduced Order Models as Python objects."""
    def __init__(self, *args, **kwargs):
        super(ROM, self).__init__(*args)
        self.__dict__.update(kwargs)  
        
    def forward(self, *args):
        raise RuntimeError("No forward method specified!")
           
    def train(self, mu, u, ntrain, epochs, optim = torch.optim.LBFGS, lr = 1, loss = None, error = None, nvalid = 0, 
              verbose = True, refresh = True, notation = 'e'):

        conv = (lambda x: num2p(x)) if notation == '%' else (lambda z: ("%.2"+notation) % z)
        optimizer = optim(self.parameters(), lr = lr)

        M = (mu,) if(isinstance(mu, torch.Tensor)) else (mu if (isinstance(mu, tuple)) else None)
        U = (u,) if(isinstance(u, torch.Tensor)) else (u if (isinstance(u, tuple)) else None)

        if(M == None):
             raise RuntimeError("Input data should be either a torch.Tensor or a tuple of torch.Tensors.")
        if(U == None):
             raise RuntimeError("Output data should be either a torch.Tensor or a tuple of torch.Tensors.")

        ntest = len(U[0])-ntrain
        Mtrain, Utrain = tuple([m[:(ntrain-nvalid)] for m in M]), tuple([um[:(ntrain-nvalid)] for um in U])
        Mvalid, Uvalid = tuple([m[(ntrain-nvalid):ntrain] for m in M]), tuple([um[(ntrain-nvalid):ntrain] for um in U])
        Mtest, Utest = tuple([m[-ntest:] for m in M]), tuple([um[-ntest:]for um in U])

        getout = (lambda y: y[0]) if len(U)==1 else (lambda y: y)
        errorf = (lambda a, b: error(a, b)) if error != None else (lambda a, b: loss(a, b).item())
        validerr = (lambda : numpy.nan) if nvalid == 0 else (lambda : errorf(getout(Uvalid), self(*Mvalid)))                                                          

        err = []
        clock = Clock()
        clock.start()      

        for e in range(epochs):   

            def closure():
                optimizer.zero_grad()
                lossf = loss(getout(Utrain), self(*Mtrain))
                lossf.backward()
                return lossf
            optimizer.step(closure)

            with torch.no_grad():
                if(self.l2().isnan().item()):
                    break
                err.append([errorf(getout(Utrain), self(*Mtrain)).item(),
                            errorf(getout(Utest), self(*Mtest)).item(),
                            validerr(),
                           ])
                if(verbose):
                    if(refresh):
                            clear_output(wait = True)

                    string = "\t\tTrain%s\txTest" % ("\txValid" if nvalid > 0 else "")
                    if(notation == 'e'):
                        string = string.replace("x", "\t")
                    else:
                        string = string.replace("x","")
                    print(string)
                    print("Epoch "+ str(e+1) + ":\t" + conv(err[-1][0]) + ("" if nvalid == 0 else ("\t" + conv(err[-1][2]))) + "\t" + conv(err[-1][1]) + ".")
                if(nvalid > 0 and e > 3):
                    if((err[-1][2] > err[-2][2]) and (err[-1][0] < err[-2][0])):
                            if((err[-2][2] > err[-3][2]) and (err[-2][0] < err[-3][0])):
                                    break

        clock.stop()
        if(verbose):
            print("\nTraining complete. Elapsed time: " + clock.elapsedTime() + ".")
        err = numpy.stack(err)
        return err, clock.elapsed()        
