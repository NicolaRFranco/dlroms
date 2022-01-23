import numpy
import torch

class Core(object):
    """Class for managing CPU and GPU tensors. Objects of this class have the following attributes:
       
       device (torch.device): specifies the underlying core and is either torch.device('cpu') or torch.device('cuda:0').
    """
    def __init__(self, device):
        """Creates a core based on the specified device.
        Input:
            device (string): the device to be used (not case-sensitive).
            Use device = 'CPU' to use 'CPU' or device = 'GPU' to run on GPU."""
        self.dtype = torch.float
        if(device.lower() == "cpu"):
            self.device = torch.device("cpu")
        else:
            self.device = torch.device("cuda:0")
                        
    def tensor(self, array):
        """Converts a numpy array into a torch (float) tensor and stores it on the corresponding core."""
        return torch.tensor(array, dtype = self.dtype, device = self.device)
    
    def zeros(self, *shape):
        """Given a shape, returns a zero tensor that is stored on the corresponding core."""
        return torch.zeros(*shape, dtype = self.dtype, device = self.device)
            
    def load(self, *paths):
        """Loads on CPU/GPU a list of arrays.
        
        Input:
        paths (tuple of strings): the paths where each array is stored.
        
        Output:
        A single CPU/GPU tensor with all the loaded data."""
        res = []
        for path in paths:
            res.append(self.tensor(numpy.load(path)))
        return torch.cat(tuple(res))
    
    def rand(self, *dims):
        return self.tensor(numpy.random.rand(*dims))
    
    def randn(self, *dims):
        return self.tensor(numpy.random.randn(*dims))
        
    def __eq__(self, other):
        """Returns True if the two objects have the same underlying core (both on CPU or GPU), false otherwise."""
        return self.device == other.device
    
CPU = Core("CPU")
GPU = Core("GPU")