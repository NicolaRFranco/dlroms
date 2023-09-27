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

import numpy as np
from scipy.linalg import eigh
from scipy.linalg import solve as scisolve
import torch
from dlroms.cores import coreof, CPU, GPU
from dlroms.dnns import Consecutive, Clock, num2p
from IPython.display import clear_output
import matplotlib.pyplot as plt

mre = lambda norm: lambda utrue, upred: (norm(utrue-upred)/norm(utrue)).mean()
mse = lambda norm: lambda utrue, upred: norm(utrue-upred, squared = True).mean()
mrei = lambda norm: lambda utrue, upred: (norm(utrue-upred)/norm(utrue)).mean().item()
msei = lambda norm: lambda utrue, upred: norm(utrue-upred, squared = True).mean().item()

def euclidean(v, squared = False):
    e2norm = v.pow(2).sum(axis = -1)
    return e2norm if squared else e2norm.sqrt()

def neuclidean(v, squared = False):
    e2norm = v.pow(2).mean(axis = -1)
    return e2norm if squared else e2norm.sqrt()

def snapshots(n, sampler, core = GPU, verbose = False):
    """Samples a collection of snapshots for a given FOM solver."""
    clock = Clock()    
    clock.start()
    mu, u = [], []
    for seed in range(n):
        if(verbose):
            print("Generating snapshot n.%d..." % (seed+1))
            clear_output(wait = True)
        mu0, u0 = sampler(seed)
        mu.append(mu0)
        u.append(u0)
    if(verbose):
        clear_output()
        clock.stop()
        print("Snapshots generated. Elapsed time: %s." % clock.elapsedTime())
    mu, u = np.stack(mu), np.stack(u)
    return core.tensor(mu, u)

def POD(U, k, inner = None):
    """Principal Orthogonal Decomposition of the snapshots matrix U into k modes."""
    m = inner.W().detach().cpu().numpy() if inner!=None else np.eye(U.shape[-1])
    U0 = U.cpu().numpy() if isinstance(U, torch.Tensor) else U
    
    M = np.dot(np.dot(U0, m), U0.T)
    N = U.shape[0]
    w, v = eigh(M, eigvals = (N-k, N-1))
    basis, eigenvalues = np.dot((v/np.sqrt(w)).T, U0), w
    basis, eigenvalues = np.flip(basis, axis = 0)+0, np.flip(eigenvalues)+0
    if(isinstance(U, torch.Tensor)):
        core = coreof(U)
        return core.tensor(basis), core.tensor(eigenvalues)
    else:
        return basis, eigenvalues

def projectdown(vbasis, u, inner = None):
    """Given a sequence of basis vbasis = [V1,..., Vk], where Vj has shape (b, Nh), and
    a sequence of vectors u = [u1,...,uk], where uj has length Nh, yields the batched
    matrix vector multiplication [Vjuj], i.e. the sequence of basis coefficients."""
    if(len(vbasis.shape)<3):
      return projectdown(vbasis.unsqueeze(0), u, inner = inner)
    else:
      if(inner!=None):
        return projectdown(vbasis, u.mm(inner.W()))
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

def project(vbasis, u, orth = True, inner = None):
    """Given a sequence of basis vbasis = [V1,..., Vk], where Vj has shape (b, Nh), and
    a sequence of vectors u = [u1,...,uk], where uj has length Nh, yields the batched
    matrix vector multiplication [Vj'Vjuj], i.e. the sequence of reconstructed vectors."""
    if(len(vbasis.shape)<3):
        return project(vbasis.unsqueeze(0), u, orth, inner)
    else:
        if(inner!=None):
            return projectup(vbasis, projectdown(vbasis, u, inner = inner))
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
    vals = torch.linalg.svdvals(A1.transpose(dim0 = 1, dim1 = 2).matmul(A2)).clamp(min=0,max=1)
    return vals.arccos()
        
def PODerrors(u, upto, ntrain, error, inner = None):
    """Projection errors over the test set for an increasing number of modes."""
    pod, svalues = POD(u[:ntrain], k = upto, inner = inner)
    errors = []
    for n in range(1, upto+1):
        uproj = project(pod[:n], u[ntrain:], inner = inner)
        errors.append(error(u[ntrain:], uproj))
    return errors


class ROM(Consecutive):
    """Abstract class for handling Reduced Order Models as Python objects."""
    def __init__(self, *args, **kwargs):
        super(ROM, self).__init__(*args)
        self.__dict__.update(kwargs)  
        self.__dict__.update({'errors':{'Train':[], 'Test':[], 'Validation':[]}, 'training_time':0})
        
    def forward(self, *args):
        raise RuntimeError("No forward method specified!")
           
    def train(self, mu, u, ntrain, epochs, optim = torch.optim.LBFGS, lr = 1, loss = None, error = None, nvalid = 0, 
              verbose = True, refresh = True, notation = 'e', title = None, batchsize = None):

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
        errorf = (lambda a, b: error(a, b)) if error != None else (lambda a, b: loss(a, b))
        validerr = (lambda : np.nan) if nvalid == 0 else (lambda : errorf(getout(Uvalid), self(*Mvalid)).item())                                                          

        err = []
        clock = Clock()
        clock.start()      

        for e in range(epochs):   

            if(batchsize == None):
                def closure():
                    optimizer.zero_grad()
                    lossf = loss(getout(Utrain), self(*Mtrain))
                    lossf.backward()
                    return lossf
                optimizer.step(closure)
            else:
                indexes = np.random.permutation(ntrain-nvalid)
                nbatch = ntrain//batchsize
                for j in range(nbatch):
                    ubatch = tuple([um[indexes[(j*batchsize):((j+1)*batchsize)]] for um in Utrain])
                    mubatch = tuple([m[indexes[(j*batchsize):((j+1)*batchsize)]] for m in  Mtrain])
                    def closure():
                        optimizer.zero_grad()
                        lossf = loss(getout(ubatch), self(*mubatch))
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
                    
                    string = "" if title == None else (title+"\n")
                    string += "\t\tTrain%s\txTest" % ("\txValid" if nvalid > 0 else "")
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
        err = np.stack(err)
        self.training_time = clock.elapsed()
        self.errors['Train'], self.errors['Test'], self.errors['Validation'] = err.T

    def eval(self):
        self.freeze()

        
def boxplot(dictionary, colors, outliers = True):
    keys = dictionary.keys()
    data = tuple([dictionary[key] for key in keys])
    bplot = plt.boxplot(data, patch_artist = True, showfliers = outliers)
    for patch, color in zip(bplot['boxes'], colors):
        patch.set_facecolor(color)       
    for patch in bplot['medians']:
        patch.set_color('w')
    plt.xticks(np.arange(1, len(keys)+1), [key.replace(" ", "\n").replace(";"," ") for key in keys])
    
def regcoeff(x, y):
    x0 = x.cpu().numpy() if isinstance(x, torch.Tensor) else x
    y0 = y.cpu().numpy() if isinstance(y, torch.Tensor) else y
    xmean, ymean  = np.mean(x0), np.mean(y0)
    sxx, sxy = ((x0-xmean)**2).sum(), ((x0-xmean)*(y0-ymean)).sum() 
    b1 = sxy/sxx
    b0 = ymean - b1*xmean
    return b0, b1

class DFNN(ROM):
    def forward(self, x):
        return super(ROM, self).forward(x)

class PODNN(ROM):
    def __init__(self, *args, **kwargs):
        kwargs.update({'trainable':True})
        if('V' not in kwargs.keys()):
            raise RuntimeError("POD matrix V unspecified. Please provide the POD matrix as a keyword argument (key = 'V').")
        super(PODNN, self).__init__(*args, **kwargs)

    def encode(self, u):
        return projectdown(self.V, u)

    def decode(self, c):
        return projectup(self.V, c)

    def redmap(self, mu):
        return self[0](mu)
    
    def forward(self, x):
        b = self.redmap(x)
        return b if(self.trainable) else self.decode(b)

    def freeze(self):
        super(PODNN, self).freeze()
        self.trainable = False

class DLROM(ROM):
    def __init__(self, *args, **kwargs):
        kwargs.update({'trainable':True})
        super(DLROM, self).__init__(*args, **kwargs)

    def encode(self, u):
        return self[2](u)

    def decode(self, c):
        return self[1](c)

    def redmap(self, mu):
        return self[0](mu)

    def forward(self, *args):
        if (self.trainable):
            mu, u = args
            newmu, nu = self.redmap(mu), self.encode(u)
            return self.decode(newmu), (newmu-nu), self.decode(nu) 
        else:
            return self.decode(self.redmap(args[0]))
        
    def freeze(self):
        super(DLROM, self).freeze()
        self.trainable = False
