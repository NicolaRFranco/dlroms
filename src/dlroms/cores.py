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

from numpy import load as npload
from numpy.random import rand as nprand, randn as nprandn
from torch import device as dev, Tensor, cat, zeros as tzeros, float as tfloat, tensor as ttensor

class Core(object):
    """Class for managing CPU and GPU Pytorch tensors. Objects of this class have the following attributes.
    
    Attributes
       device   (torch.device)  Underlying core. Equals either torch.device('cpu') or torch.device('cuda:0').
       
    """
    def __init__(self, device):
        """Creates a reference to the specified core.
        
        Input
            device  (str)   Device to be used (not case-sensitive). Accepted strings are 'CPU' and 'GPU'."""
        
        self.dtype = tfloat
        if(device.lower() == "cpu"):
            self.device = dev("cpu")
        else:
            self.device = dev("cuda:0")
                        
    def tensor(self, *arrays):
        """Transfers a collection of arrays to the corresponding core and turns them into a torch (float) tensors.
        
        Input
            *arrays   (numpy.ndarray, lists)     Arrays to be converted.
            
        Output 
            tuple of torch.Tensor objects.
        """
        if(len(arrays)==1):
            return ttensor(arrays[0], dtype = self.dtype, device = self.device)
        else:    
            return (*[ttensor(array, dtype = self.dtype, device = self.device) for array in arrays],)
    
    def zeros(self, *shape):
        """Returns a tensor with all entries equal to zero.
        
        Input
            *shape  (tuple of ints)     Shape of the tensor. E.g., self.zeros(2,3) creates a 2x3 tensor full of zeros.
            
        Output
            (torch.Tensor).
        """
        return tzeros(*shape, dtype = self.dtype, device = self.device)
            
    def load(self, *paths):
        """Loads a list of arrays into a single tensor.
        
        Input
            paths (tuple of str)    Paths where each array is stored. These are assumed to be in .npy format.
                                    All the arrays must have the same shape (except for the first, batch, dimension).
        
        Output
            (torch.Tensor)."""
        res = []
        for path in paths:
            res.append(self.tensor(npload(path)))
        return cat(tuple(res))
    
    def rand(self, *dims):
        """Returns a tensor with random entries sampled uniformely from [0,1].
        
        Input
            *dims   (tuple of int)  Shape of the tensor.
            
        Output
            (torch.Tensor)."""
        return self.tensor(nprand(*dims))
    
    def randn(self, *dims):
        """Returns a tensor with random entries sampled independently from the normal distribution N(0,1).
        
        Input
            *dims   (tuple of int)  Shape of the tensor.
            
        Output
            (torch.Tensor)."""
        return self.tensor(nprandn(*dims))
        
    def __eq__(self, other):
        """Compares two cores.
        
        Input
            other   (dlroms.cores.Core)     Core to be compared with the current one.
            
        Output
            (bool) returns True if the two cores both refer to the CPU or GPU respectively."""
        return self.device == other.device
    
CPU = Core("CPU")
GPU = Core("GPU")

def coreof(u):
    """Returns the core where the given tensor is stored.
    
    Input
        u   (torch.Tensor)
        
    Output
        (dlroms.cores.Core)."""
    
    if(isinstance(u, Tensor)):
        if(u.device == CPU.device):
            return CPU
        elif(u.device == GPU.device):
            return GPU
        else:
            raise RuntimeError("Tensor is stored on an unknown core.")
    else:
        raise RuntimeError("Can only retrieve the core of a torch tensor.")
