import numpy
import torch

class Core(object):
    """Class for managing CPU and GPU tensors. Objects of this class have the following attributes.
    
    Attributes
       device   (torch.device)  Underlying core. Equals either torch.device('cpu') or torch.device('cuda:0').
       
    """
    def __init__(self, device):
        """Creates a reference to the specified core.
        
        Input
            device  (str)   Device to be used (not case-sensitive). Accepted strings are 'CPU' and 'GPU'."""
        
        self.dtype = torch.float
        if(device.lower() == "cpu"):
            self.device = torch.device("cpu")
        else:
            self.device = torch.device("cuda:0")
                        
    def tensor(self, array):
        """Converts a numpy array into a torch (float) tensor to be stored on the corresponding core.
        
        Input
            array   (numpy.ndarray)     Array to be converted.
            
        Output 
            (torch.Tensor).
        """
        return torch.tensor(array, dtype = self.dtype, device = self.device)
    
    def zeros(self, *shape):
        """Returns a tensor full of zeros.
        
        Input
            *shape  (tuple of int)     Shape of the tensor. E.g., self.zeros(2,3) creates a 2x3 tensor full of zeros.
            
        Output
            (torch.Tensor).
        """
        return torch.zeros(*shape, dtype = self.dtype, device = self.device)
            
    def load(self, *paths):
        """Loads a list of arrays into a single tensor.
        
        Input
            paths (tuple of str)    Paths where each array is stored. These are assumed to be in .npy format.
                                    All the arrays must have the same shape (except for the first, batch, dimension).
        
        Output
            (torch.Tensor)."""
        res = []
        for path in paths:
            res.append(self.tensor(numpy.load(path)))
        return torch.cat(tuple(res))
    
    def rand(self, *dims):
        """Returns a tensor with randomly filled values. The latter are sampled uniformely from [0,1].
        
        Input
            *dims   (tuple of int)  Shape of the tensor.
            
        Output
            (torch.Tensor)."""
        return self.tensor(numpy.random.rand(*dims))
    
    def randn(self, *dims):
        """Returns a tensor with randomly filled values. The latter are sampled independently from the normal distribution N(0,1).
        
        Input
            *dims   (tuple of int)  Shape of the tensor.
            
        Output
            (torch.Tensor)."""
        return self.tensor(numpy.random.randn(*dims))
        
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
    """Returns the core over with a tensor is stored.
    
    Input
        u   (torch.Tensor)
        
    Output
        (dlroms.cores.Core)."""
    
    if(isinstance(u, torch.Tensor)):
        if(u.device == CPU.device):
            return CPU
        elif(u.device == GPU.device):
            return GPU
        else:
            raise RuntimeError("Tensor is stored on an unknown core.")
    else:
        raise RuntimeError("Can only retrieve the core of a torch tensor.")
