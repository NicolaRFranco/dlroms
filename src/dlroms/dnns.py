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

import numpy
import torch
import matplotlib.pyplot as plt
from time import perf_counter     
from dlroms.cores import CPU, GPU
from IPython.display import clear_output
import os
import warnings

ReLU = torch.nn.ReLU()
leakyReLU = torch.nn.LeakyReLU(0.1) 
num2p = lambda prob : ("%.2f" % (100*prob)) + "%"
        
class Layer(torch.nn.Module):
    """Layer of a neural network. It is implemented as a subclass of 'torch.nn.Module'. Acts as an abstract class.
    Objects of class layer have the following attributes:
    
    core (Core): specifies wheather the layer is stored on CPU or GPU.
    rho (function): the activation function of the layer.
    
    All objects of class layer should implement an abstract method '.module()' that returns the underlying torch.nn.Module.
    Layers can be applied to tensors using .forward() or simply .(), i.e. following the syntax of function calls.
    """
    
    def __init__(self, activation):
        """Creates a new layer with a given activation function (activation). By default, the layer is initiated on CPU."""
        super(Layer, self).__init__()
        self.rho = activation
        if(activation == None):
            self.rho = torch.nn.Identity()
        self.core = CPU

    def coretype(self):
        return self.core
        
    def w(self):
        """Returns the weights of the layer."""
        return self.module().weight
    
    def b(self):
        """Returns the bias vector of the layer."""
        return self.module().bias
    
    def scale(self, factor):
        """Sets to zero the bias and scales the weight matrix by 'factor'."""
        self.load(factor*self.w().detach().cpu().numpy(), 0.0*self.b().detach().cpu().numpy())
        
    def zeros(self):
        """Sets to zero all weights and biases."""
        self.module().weight = torch.nn.Parameter(0.0*self.module().weight)
        self.module().bias = torch.nn.Parameter(0.0*self.module().bias)
        
    def moveOn(self, core):
        """Transfers the layer structure on the specified core."""
        self.core = core
        if(core != CPU):
            self.cuda()
        else:
            self.cpu()
        
    def cuda(self):
        """Transfers the layer to the GPU."""
        self.core = GPU
        self.module().cuda()
        
    def cpu(self):
        """Transfers the layer to the CPU."""
        self.core = CPU
        self.module().cpu()
        
    def l2(self):
        """Returns the square sum of all weights within the layer."""
        return (self.module().weight**2).sum()
    
    def l1(self):
        """Returns the absolute sum of all weights within the layer."""
        return self.module().weight.abs().sum()
    
    def outdim(self, inputdim):
        """Given a tuple for the input dimension, returns the corresponding output dimension."""
        return tuple(self.forward(self.core.zeros(*inputdim)).size())
        
    def load(self, w, b = None):
        """Given a pair of weights and biases, it loads them as parameters for the Layer.
        
        Input:
        w (numpy array): weights
        b (numpy array): bias vector. Defaults to None (i.e., only loads w).
        """
        self.module().weight = torch.nn.Parameter(self.core.tensor(w))
        try:
            self.module().bias = torch.nn.Parameter(self.core.tensor(b))
        except:
            None
        
    def inherit(self, other, azzerate = True):
        """Inherits the weight and bias from another network. Additional entries are left to zero.
        It can be seen as a naive form of transfer learning.
        
        Input:
        other (Layer): the layer from which the parameters are learned.
        
        Output:
        None, but the current network has now updated parameters.
        """
        if(azzerate):
            self.zeros()
        with torch.no_grad():
            where = tuple([slice(0,s) for s in other.w().size()])
            self.module().weight[where] = torch.nn.Parameter(self.core.tensor(other.w().detach().cpu().numpy())) 
            where = tuple([slice(0,s) for s in other.b().size()])
            self.module().bias[where] = torch.nn.Parameter(self.core.tensor(other.b().detach().cpu().numpy())) 

        
    def __add__(self, other):
        """Connects the current layer to another layer (or a sequence of layers), and returns
        the corresponding nested architecture.
       
        Input:
        self (Layer): the current layer
        other (Layer / Consecutive): the architecture to be connected on top.
        
        Output:
        The full architecture, stored as 'Consecutive' object.
        """
        if(isinstance(other, Consecutive) and (not isinstance(other, Parallel))):
            n = len(other)
            layers = [self]+[other[i] for i in range(n)]
            return Consecutive(*tuple(layers))
        else:
            if(other == 0.0):
                return self
            else:
                return Consecutive(self, other)    
            
    def __pow__(self, number):
        """Creates a Parallel architecture by pasting 'number' copies of the same layer next to each other."""
        if(number > 1):
            l = [self]
            for i in range(number-1):
                l.append(self.copy())
            return Parallel(*tuple(l))
        elif(number == 1):
            return self
        else:
            return 0.0
    
    def __mul__(self, number):
        """Creates a deep neural network by pasting 'number' copies of the same layer."""
        if(number > 1):
            x = self + self
            for i in range(number-2):
                x = x+self
            return x
        elif(number == 1):
            return self
        else:
            return 0.0
    
    def __rmul__(self, number):
        """See self.__mul__."""
        return self*number
    
    def dof(self):
        """Degrees of freedom in the layer, defined as the number of active weights and biases."""
        return numpy.prod(tuple(self.module().weight.size())) + len(self.module().bias)
                     
    def He(self, linear = False, a = 0.1, seed = None):
        """He initialization of the weights."""
        if(seed != None):
            torch.manual_seed(seed)
        if(linear):
            torch.nn.init.xavier_normal_(self.module().weight)
        else:
            torch.nn.init.kaiming_normal_(self.module().weight, mode='fan_out', nonlinearity='leaky_relu', a = a)
        
    def Xavier(self):
            torch.nn.init.xavier_uniform_(self.module().weight)
        
    def inputdim(self):
        """Returns the expected input dimension for the layer."""
        return self.module().in_features
    
    def freeze(self, w = True, b = True):
        """Freezes the layer so that its parameters to not require gradients.
        Input:
        w (boolean): wheather to fix or not the weights.
        b (boolean): wheather to fix or not the bias."""
        if(w and b):
            self.module().requires_grad_(False)
        elif(w):
            self.module().weight.requires_grad_(False)
        elif(b):
            self.module().bias.requires_grad_(False)
        self.training = False
        
    def unfreeze(self):
        """Makes all the layer parameters learnable."""
        self.module().bias.requires_grad_(True)
        self.module().weight.requires_grad_(True)
        self.module().requires_grad_(True)
        self.training = True
        
    def dictionary(self, label = ""):
        """Returns a dictionary with the layer parameters. An additional label can be added."""
        return {('w'+label):self.w().detach().cpu().numpy(), ('b'+label):self.b().detach().cpu().numpy()}
    
    def parameters(self):
        ps = list(super(Layer, self).parameters())
        res = []
        for p in ps:
            if(p.requires_grad):
                res.append(p)
                
        return res
    

class Dense(Layer):
    """Fully connected Layer."""
    
    def __init__(self, input_dim, output_dim, activation = leakyReLU, bias = True):
        """Creates a Dense Layer with given input dimension, output dimension and activation function."""
        super(Dense, self).__init__(activation)
        in_d = input_dim if(isinstance(input_dim, int)) else input_dim.dim()            
        out_d = output_dim if(isinstance(output_dim, int)) else output_dim.dim()
        self.lin = torch.nn.Linear(in_d, out_d, bias = bias)
        
    def module(self):
        return self.lin
        
    def forward(self, x):
        return self.rho(self.lin(x))
    
class Residual(Layer):
    """Residual layer. Differs from a dense layer has x -> x + f(x) instead of x -> f(x)."""
    
    def __init__(self, dim, activation = leakyReLU):
        """Creates a Residual layer with input-output dimension 'dim' and activation function 'activation'."""
        super(Residual, self).__init__(activation)
        self.lin = torch.nn.Linear(dim, dim)
        
    def module(self):
        return self.lin
    
    def forward(self, x):
        return x + self.rho(self.lin(x))   

class Sparse(Layer):
    """Layer with weights that have a (priorly) fixed sparsity."""
    
    def __init__(self, mask, activation = leakyReLU):
        """Creates a Sparse layer.
        
        Input:
        mask (numpy 2D array): a 2D array that works as sample for the weight matrix. 
        It should have the same sparsity required to the weight matrix.
        activation (function): activation function of the layer.
        """
        super(Sparse, self).__init__(activation)
        self.loc = numpy.nonzero(mask)
        self.in_d, self.out_d = mask.shape
        self.weight = torch.nn.Parameter(CPU.zeros(len(self.loc[0])))
        self.bias = torch.nn.Parameter(CPU.zeros(self.out_d))
        
    def moveOn(self, core):
        self.core = core
        with torch.no_grad():
            if(core == GPU):      
                self.weight = torch.nn.Parameter(self.weight.cuda())
                self.bias = torch.nn.Parameter(self.bias.cuda())
            else:
                self.weight = torch.nn.Parameter(self.weight.cpu())
                self.bias = torch.nn.Parameter(self.bias.cpu())
        
    def module(self):
        return self
    
    def forward(self, x):
        W = self.core.zeros(self.in_d, self.out_d)
        W[self.loc] = self.weight
        return self.rho(self.bias + x.mm(W))
    
    def inherit(self, other):
        W = self.core.zeros(self.in_d, self.out_d)
        W[other.loc] = other.weight
        with torch.no_grad():
            self.weight = torch.nn.Parameter(self.core.copy(W[self.loc]))
            self.bias = torch.nn.Parameter(self.core.copy(other.bias))        

    def He(self, linear = False, a = 0.1, seed = None):
        #nw = len(self.weight)
        #with torch.no_grad():
        #    self.weight = torch.nn.Parameter(torch.rand(nw)/numpy.sqrt(nw))
        c = 1 if linear else 2
        alpha = 0 if linear else a
        A = numpy.zeros((self.out_d, self.in_d))
        A[(self.loc[1], self.loc[0])] = 1
        nnz = numpy.sum(A>0, axis = 1)[self.loc[1]]
        with torch.no_grad():
            self.weight = torch.nn.Parameter(self.core.tensor(numpy.random.randn(len(self.loc[0]))*numpy.sqrt(c/(nnz*(1.0+alpha**2)))))
        
    def deprecatedHe(self, linear = False, a = 0.1, seed = None):
        nw = len(self.weight)
        with torch.no_grad():
            self.weight = torch.nn.Parameter(torch.rand(nw)/numpy.sqrt(nw))
        
    def Xavier(self):
        A = numpy.zeros((self.out_d, self.in_d))
        A[(self.loc[1], self.loc[0])] = 1
        nnz = numpy.sum(A>0, axis = 1)[self.loc[1]]
        with torch.no_grad():
            self.weight = torch.nn.Parameter(self.core.tensor((2*numpy.random.rand(len(self.loc[0]))-1)*numpy.sqrt(3/nnz)))
        
            
    def W(self):
        W = self.core.zeros(self.in_d, self.out_d)
        W[self.loc] = self.weight
        return W

    def dictionary(self, label = ""):
        return {('w'+label):self.w().detach().cpu().numpy(), ('b'+label):self.b().detach().cpu().numpy(), ('indexes'+label):self.loc}

    def load(self, w, b = None, indexes = None):
        super(Sparse, self).load(w, b)
        if(isinstance(indexes, numpy.ndarray)):
            self.loc = indexes
        
    def cuda(self):
        self.moveOn(GPU)
        
    def cpu(self):
        self.moveOn(CPU)
        
class Weightless(Layer):
    """Subclass of Layer that handles the case of weightless layers (no trainable parameters).
    By default, this layers have .w() = None, .b() = None, .rho = None and .dof() = 0, etc.
    All redundant methods have been overwritten."""
    
    def __init__(self):
        super(Weightless, self).__init__(None)
        self.rho = None
        self.core = CPU
        self.training = False
        
    def inherit(self, other):
        None
        
    def scale(self, factor):
        None
    
    def w(self):
        return None
    
    def b(self):
        return None
    
    def zeros(self):
        None
        
    def cuda(self):
        None
        
    def cpu(self):
        None
        
    def l2(self):
        return 0.0
    
    def l1(self):
        return 0.0
        
    def dof(self):
        return 0
    
    def He(self, linear, a, seed):
        None
        
    def Xavier(self):
        None
        
    def freeze(self, w = True, b = True):
        None
        
    def unfreeze(self):
        None
        
    def load(self, w, b):
        None
        
    def dictionary(self, label):
        return dict()

class Reshape(Weightless):
    """Weightless layer used for reshaping tensors. Object of this class have the additional attribute:
    
    newdim (tuple): the new shape after reshaping. 
    
    As all Layers and Modules, it is expected to operate on multiple instances simultaneously.
    If newdim = (p1,..., pj), then an input of size [n, d1,..., dk] is reshaped to [n, p1,..., pj]
    as the new dimension is applied to each tensor in the sample.
    It is expected that d1*...*dk = p1*...*pj.
    """
    def __init__(self, *shape):
        """Creates a Reshape layer with newdim = shape."""
        super(Reshape, self).__init__()
        self.newdim = shape
        
    def forward(self, x):
        """Reshapes the input tensor x."""
        n = len(x)
        return x.reshape(n,*self.newdim)    
    
    def inputdim(self):
        """Expected input dimension."""
        return numpy.prod(self.newdim)
    
class Inclusion(Weightless):
    """Weightless layer that embedds R^m into R^n, m < n, with x -> [0,...,0, x]."""
    
    def __init__(self, in_dim, out_dim):
        super(Inclusion, self).__init__()
        self.d = out_dim - in_dim
        
    def forward(self, x):
        z = self.core.zeros(len(x), self.d)
        return torch.cat((z, x), axis = 1)
    
class Trunk(Weightless):
    
    def __init__(self, out_dim):
        super(Trunk, self).__init__()
        self.d = out_dim
        
    def forward(self, x):
        return x[:,:self.d]

class Transpose(Weightless):
        
    def __init__(self, dim0, dim1):
        super(Transpose, self).__init__()
        self.d0 = dim0
        self.d1 = dim1
        
    def forward(self, x):
        return x.transpose(dim0 = self.d0+1, dim1 = self.d1+1)
        
        
class Conv2D(Layer):
    """Layer that performs 2D convolutions (cf. Pytorch documentation, torch.nn.Conv2d)."""
    
    def __init__(self, window, channels = (1,1), stride = 1, padding = 0, groups = 1, dilation = 1, activation = leakyReLU):
        """Creates a convolutional 2D layer.
        
        Input:
        window (tuple or int): 2D shape of the rectangular filter to be applied. Passing an integer k is equivalent to passing (k, k).
        channels (tuple): pair containing the number of input and output channels.
        stride (int): stride of the filter (i.e. gap between the movements of the filter). Defaults to 1.
        padding (int): padding applied (pre or post?) to the convolution (cf. torch.nn.Conv2d documentation).
        activation (function): activation function of the layer
        
        Expects 4D tensors as input with the convention [#observations, #channels, #height, #width]."""
        super(Conv2D, self).__init__(activation)
        self.conv = torch.nn.Conv2d(channels[0], channels[1], window, stride = stride, padding = padding, groups = groups, dilation = dilation)
        
    def module(self):
        return self.conv
        
    def forward(self, x):
        return self.rho(self.conv(x))    

    def inputdim(self):
        raise RuntimeError("Convolutional2D layers do not have a fixed input dimension.")
        
    #def He(self, seed = None):
    #    if(seed != None):
    #        torch.manual_seed(seed)
    #    torch.nn.init.kaiming_normal_(self.module().weight, a = 0.1, mode='fan_out', nonlinearity='leaky_relu')        
    
class Deconv2D(Layer):
    """Layer that performs a transposed 2D convolution (cf. Pytorch documentation, torch.nn.ConvTranspose2d)."""
    
    def __init__(self, window, channels = (1,1), stride = 1, padding = 0,  groups = 1, dilation = 1, activation = leakyReLU):
        """Creates a Deconvolutional2D layer. Arguments read as in Convolutional2D.__init__."""
        super(Deconv2D, self).__init__(activation)
        self.deconv = torch.nn.ConvTranspose2d(channels[0], channels[1], window, stride = stride, padding = padding, groups = groups, dilation = dilation)
        
    def module(self):
        return self.deconv
        
    def forward(self, x):
        return self.rho(self.deconv(x))
            
    def inputdim(self):
        raise RuntimeError("Deconvolutional2D layers do not have fixed input dimension.")
        
    #def He(self, seed = None):
    #    if(seed != None):
    #        torch.manual_seed(seed)
    #    torch.nn.init.kaiming_normal_(self.module().weight, a = 0.1, mode='fan_out', nonlinearity='leaky_relu')

class Conv3D(Layer):
    """Layer that performs 3D convolutions (cf. Pytorch documentation, torch.nn.Conv2d)."""
    
    def __init__(self, window, channels = (1,1), stride = 1, padding = 0, groups = 1, dilation = 1, activation = leakyReLU):
        """Creates a convolutional 3D layer.
        
        Input:
        window (tuple or int): 3D shape of the rectangular filter to be applied. Passing an integer k is equivalent to passing (k, k).
        channels (tuple): pair containing the number of input and output channels.
        stride (int): stride of the filter (i.e. gap between the movements of the filter). Defaults to 1.
        padding (int): padding applied (pre or post?) to the convolution (cf. torch.nn.Conv2d documentation).
        activation (function): activation function of the layer
        
        Expects 5D tensors as input with the convention [#observations, #channels, #height, #width, #depth]."""
        super(Conv3D, self).__init__(activation)
        self.conv = torch.nn.Conv3d(channels[0], channels[1], window, stride = stride, padding = padding, groups = groups, dilation = dilation)
        
    def module(self):
        return self.conv
        
    def forward(self, x):
        return self.rho(self.conv(x))    

    def inputdim(self):
        raise RuntimeError("Convolutional3D layers do not have a fixed input dimension.")
        
    #def He(self, seed = None):
    #    if(seed != None):
    #        torch.manual_seed(seed)
    #    torch.nn.init.kaiming_normal_(self.module().weight, a = 0.1, mode='fan_out', nonlinearity='leaky_relu')        
    
class Deconv3D(Layer):
    """Layer that performs a transposed 3D convolution (cf. Pytorch documentation, torch.nn.ConvTranspose3d)."""
    
    def __init__(self, window, channels = (1,1), stride = 1, padding = 0,  groups = 1, dilation = 1, activation = leakyReLU):
        """Creates a Deconvolutional3D layer. Arguments read as in Convolutional3D.__init__."""
        super(Deconv3D, self).__init__(activation)
        self.deconv = torch.nn.ConvTranspose3d(channels[0], channels[1], window, stride = stride, padding = padding, groups = groups, dilation = dilation)
        
    def module(self):
        return self.deconv
        
    def forward(self, x):
        return self.rho(self.deconv(x))
            
    def inputdim(self):
        raise RuntimeError("Deconvolutional3D layers do not have fixed input dimension.")
        
    #def He(self, seed = None):
    #    if(seed != None):
    #        torch.manual_seed(seed)
    #    torch.nn.init.kaiming_normal_(self.module().weight, a = 0.1, mode='fan_out', nonlinearity='leaky_relu')

class Conv1D(Layer):    
    """Analogous to Convolutional2D but considers 1D convolutions."""
    def __init__(self, window, channels = (1,1), stride = 1, padding = 0,  groups = 1, dilation = 1, activation = leakyReLU):
        super(Conv1D, self).__init__(activation)
        self.conv = torch.nn.Conv1d(channels[0], channels[1], window, stride = stride, padding = padding, groups = groups, dilation = dilation)
        
    def module(self):
        return self.conv
        
    def forward(self, x):
        return self.rho(self.conv(x))    

    def inputdim(self):
        raise RuntimeError("Convolutional1D layers do not have fixed input dimension.")
        
    #def He(self, seed = None):
    #    if(seed != None):
    #        torch.manual_seed(seed)
    #    torch.nn.init.kaiming_normal_(self.module().weight, a = 0.1, mode='fan_out', nonlinearity='leaky_relu')
        
class Deconv1D(Layer):    
    """Analogous to Deconvolutional2D but considers trasposed 1D convolutions."""
    def __init__(self, window, channels = (1,1), stride = 1, padding = 0,  groups = 1, dilation = 1, activation = leakyReLU):
        super(Deconv1D, self).__init__(activation)
        self.deconv = torch.nn.ConvTranspose1d(channels[0], channels[1], window, stride = stride, padding = padding, groups = groups, dilation = dilation)
        
    def module(self):
        return self.deconv
        
    def forward(self, x):
        return self.rho(self.deconv(x))
            
    def inputdim(self):
        raise RuntimeError("Deconvolutional1D layers do not have fixed input dimension.")
        
    #def He(self, seed = None):
    #    if(seed != None):
    #        torch.manual_seed(seed)
    #    torch.nn.init.kaiming_normal_(self.module().weight, a = 0.1, mode='fan_out', nonlinearity='leaky_relu')

        
class Consecutive(torch.nn.Sequential):
    """Class that handles deep neural networks, obtained by connecting multiple layers. 
    It is implemented as a subclass of torch.nn.Sequential.
    
    Objects of this class support indexing, so that self[k] returns the kth Layer in the architecture.
    """

    def coretype(self):
        return self[0].coretype()
        
    def scale(self, factor):
        """Scales all layers in the architecture (see Layer.scale)."""
        for nn in self:
            nn.scale(factor)
    
    def l2(self):
        """Returns the squared sum of all weights in the architecture."""
        m = 0.0
        N = len(self)
        for i in range(N):
            m += self[i].l2()
        return m
    
    def l1(self):
        """Returns the absolute sum of all weights in the architecture."""
        m = 0.0
        N = len(self)
        for i in range(N):
            m += self[i].l1norm()
        return m
    
    def zeros(self):
        """Sets to zero all weights and biases in the architecture."""
        N = len(self)
        for i in range(N):
            self[i].zeros()
            
    def outdim(self, input_dim = None):
        """Analogous to Layer.outdim."""
        if(input_dim == None):
            input_dim = self[0].inputdim()
        m = input_dim
        N = len(self)
        for i in range(N):
            m = self[i].outdim(m)
        return m
    
    def inputdim(self):
        """Analogous to Layer.inputdim."""
        return self[0].inputdim()
    
    def moveOn(self, core):
        """Transfers all layers to the specified core."""
        for layer in self:
                    layer.moveOn(core)
    
    def cuda(self):
        """Transfers all layers to the GPU."""
        N = len(self)
        for i in range(N):
            m = self[i].cuda()
            
    def cpu(self):
        """Transfers all layers to the CPU."""
        N = len(self)
        for i in range(N):
            m = self[i].cpu()
        
    def stretch(self):
        res = []
        for nn in self:
            if(isinstance(nn, Consecutive)):
                res += nn.stretch()
            else:
                res += [nn]
        return res
    
    def dictionary(self, label = ""):
        """Returns a dictionary with all the parameters in the network. An additional label can be passed."""
        params = dict()
        k = 0
        for nn in self:
            k +=1
            params.update(nn.dictionary(str(k)+label))
        return params
    
    def save(self, path, label = ""):
        """Stores the whole architecture to the specified path. An additional label can be added (does not influence the name of
        the file, it is only used internally; defaults to '')."""
        if(len(self) == len(self.stretch())):
            params = self.dictionary(label)
            numpy.savez(path, **params)
        else:
            Consecutive(*self.stretch()).save(path, label)
        
    def load(self, path, label = ""):
        """Loads the architecture parameters from stored data.
        
        Input:
        path (string): system path where the parameters are stored.
        label (string): additional label required if the stored data had one.
        """
        if(len(self) == len(self.stretch())):
            try:
                params = numpy.load(path)
            except:
                params = numpy.load(path+".npz")
            k = 0
            for nn in self:
                k += 1
                try:
                    if(isinstance(nn, Sparse)):
                        nn.load(w = params['w'+str(k)+label], b = params['b'+str(k)+label], indexes = params['indexes'+str(k)+label])
                    else:
                        nn.load(params['w'+str(k)+label], params['b'+str(k)+label])
                except:
                    None
        else:
             Consecutive(*self.stretch()).load(path, label)   
                
    def __add__(self, other):
        """Augments the current architecture by connecting it with a second one.
        
        Input:
        self (Consecutive): current architecture.
        other (Layer / Consecutive): neural network to be added at the end.
        
        Output:
        A Consecutive object consisting of the nested neural network self+other. 
        """
        if(isinstance(other, Consecutive) and (not isinstance(other, Parallel))):
            n1 = len(self)
            n2 = len(other)
            layers = [self[i] for i in range(n1)]+[other[i] for i in range(n2)]
            return Consecutive(*tuple(layers))
        else:
            if(other == 0.0):
                return self
            else:
                n1 = len(self)
                layers = [self[i] for i in range(n1)]
                layers.append(other)
                return Consecutive(*tuple(layers))      
        
    def dof(self):
        """Total number of (learnable) weights and biases in the network."""
        res = 0
        for x in self:
            res += x.dof()
        return res  
    
    def He(self, linear = False, a = 0.1, seed = None):
        """Applies the He initialization to all layers in the architecture."""
        for x in self:
            x.He(linear = linear, a = a, seed = seed)
     
    def Xavier(self):
        """Applies the (uniform) Xavier initialization to all layers in the architecture."""
        for x in self:
            x.Xavier()
    
    def parameters(self):
        """Returns the list of all learnable parameters in the network. Used as argument for torch optimizers."""
        p = []
        for f in self:
            p += list(f.parameters())
        return p
    
    def dims(self, inputdim = None):
        """Returns the sequence of dimensions through which an input passes when transformed by the network."""
        if(inputdim == None):
            inp = (1, self[0].inputdim())
        else:
            inp = inputdim
        journey = ""
        ii = inp[1:]
        if(len(ii)==1):
            journey += str(ii[0])
        else:
            journey += str(inp[1:])
        for f in self:
            journey += " -> "
            inp = f.outdim(inp)
            ii = inp[1:]
            if(len(ii)==1):
                journey += str(ii[0])
            else:
                journey += str(inp[1:])         
        return journey
    
    def freeze(self, w = True, b = True):
        """Freezes all layers in the network (see Layer.freeze)."""
        for f in self:
            f.freeze(w, b)
        self.training = False
        
    def unfreeze(self):
        """Makes all layers learnable (see Layer.unfreeze)."""
        for f in self:
            f.unfreeze()
        self.training = True
            
    def inherit(self, other, azzerate = True):
        """Inherits the networks parameters from a given architecture (cf. Layer.inherit). 
        The NN 'other' should have a depth less or equal to that of 'self'.
        
        Input:
        other (Consecutive): the architecture from which self shoud learn."""
        for i, nn in enumerate(other):
            if(type(self[i])==type(nn)):
                self[i].inherit(nn, azzerate)
                
    def files(self, string):
        return [string+".npz"]
        
class Parallel(Consecutive):
    """Architecture with multiple layers that work in parallel but channel-wise. Implemented as a subclass of Consecutive.
    If f1,...,fk is the collection of layers, then Parallel(f1,..,fk)(x) = [f1(x1),..., fk(xk)], where x = [x1,...,xk] is
    structured in k channels."""
    
    def __init__(self, *args):
        super(Parallel, self).__init__(*args)
        
    def forward(self, x):
        res = [self[k](x[:,k]) for k in range(len(self))]
        return torch.stack(res, axis = 1)
    
                    
    def __add__(self, other):
        """Augments the current architecture by connecting it with a second one.
        
        Input:
        self (Parallel): current architecture.
        other (Layer / Consecutive): neural network to be added at the end.
        
        Output:
        A Consecutive object consisting of the nested neural network self+other. 
        """
        if(isinstance(other, Consecutive) and (not isinstance(other, Parallel))):
            n1 = len(self)
            n2 = len(other)
            layers = [self]+[other[i] for i in range(n2)]
            return Consecutive(*tuple(layers))
        else:
            if(other == 0.0):
                return self
            else:
                return Consecutive(self, other) 
    
class Channelled(Consecutive):
    """Architecture with multiple layers that work in parallel but channel-wise. Implemented as a subclass of Consecutive.
    If f1,...,fk is the collection of layers, then Channelled(f1,..,fk)(x) = fk(xk)+...+f1(x1), where x = [x1,...,xk] is
    structured in k channels."""
    
    def __init__(self, *args):
        super(Channelled, self).__init__(*args)
        
    def forward(self, x):
        res = 0.0
        k = 0
        for f in self:
            res = res + f(x[:,k])    
            k += 1
        return res


class Clock(object):
    """Class for measuring (computational) time intervals. Objects of this class have the following attributes:
    
    tstart (float): time at which the clock was started (in seconds).
    tstop  (float): time at which the clock was stopped (in seconds).
    """
    
    def __init__(self):
        """Creates a new clock."""
        self.tstart = 0
        self.tstop = 0
        
    def start(self):
        """Starts the clock."""
        self.tstart = perf_counter()
        
    def stop(self):
        """Stops the clock."""
        self.tstop = perf_counter()
        
    def elapsed(self):
        """Returns the elapsed time between the calls .start() and .stop()."""
        dt = self.tstop-self.tstart
        
        if(dt<0):
            raise RuntimeError("Clock still running.")
        else:
            return dt 
        
    def elapsedTime(self):
        """Analogous to .elapsed() but returns the output in string format."""
        return Clock.parse(self.elapsed())

        
    @classmethod
    def parse(cls, time):
        """Converts an amount of seconds in a string of the form '# hours #minutes #seconds'."""
        h = time//3600
        m = (time-3600*h)//60
        s = time-3600*h-60*m
        
        if(h>0):
            return ("%d hours %d minutes %.2f seconds" % (h,m,s))
        elif(m>0):
            return ("%d minutes %.2f seconds" % (m,s))
        else:
            return ("%.2f seconds" % s)
        
        
    @classmethod
    def shortparse(cls, time):
        """Analogous to Clock.parse but uses the format '#h #m #s'."""
        h = time//3600
        m = (time-3600*h)//60
        s = time-3600*h-60*m
        
        if(h>0):
            return ("%d h %d m %.2f s" % (h,m,s))
        elif(m>0):
            return ("%d m %.2f s" % (m,s))
        else:
            return ("%.2f s" % s)
        
        
def train(dnn, mu, u, ntrain, epochs, optim = torch.optim.LBFGS, lr = 1, lossf = None, error = None, verbose = True, until = None, nvalid = 0, conv = num2p,
          best = False, cleanup = True, dropout = 0.0):
    warnings.warn("This function is deprecated. Consider using the DFNN wrapper in dlroms.roms, and then calling the class method 'train': e.g., new_dnn = DFNN(dnn), new_dnn.train(...).")
    optimizer = optim(dnn.parameters(), lr = lr)
    ntest = len(mu)-ntrain
    mutrain, utrain, mutest, utest = mu[:(ntrain-nvalid)], u[:(ntrain-nvalid)], mu[-ntest:], u[-ntest:]
    muvalid, uvalid = mu[(ntrain-nvalid):ntrain], u[(ntrain-nvalid):ntrain]
    
    if(error == None):
        def error(a, b):
            return lossf(a, b)

    err = []
    clock = Clock()
    clock.start()
    bestv = numpy.inf
    tempcode = int(numpy.random.rand(1)*1000)
        
    validerr = (lambda : numpy.nan) if nvalid == 0 else (lambda : error(uvalid, dnn(muvalid)).item())

    for e in range(epochs):
        
        if(dropout>0.0):
            dnn.unfreeze()
            for layer in dnn:
                if(numpy.random.rand()<=dropout):
                    layer.freeze()      
        
        def closure():
            optimizer.zero_grad()
            loss = lossf(utrain, dnn(mutrain))
            loss.backward()
            return loss
        optimizer.step(closure)

        with torch.no_grad():
            if(dnn.l2().isnan().item()):
                break
            err.append([error(utrain, dnn(mutrain)).item(),
                        error(utest, dnn(mutest)).item(),
                        validerr(),
                       ])
            if(verbose):
                if(cleanup):
                        clear_output(wait = True)
                
                print("\t\tTrain%s\tTest" % ("\tValid" if nvalid > 0 else ""))
                print("Epoch "+ str(e+1) + ":\t" + conv(err[-1][0]) + ("" if nvalid == 0 else ("\t" + conv(err[-1][2]))) + "\t" + conv(err[-1][1]) + ".")
            if(nvalid > 0 and e > 3):
                if((err[-1][2] > err[-2][2]) and (err[-1][0] < err[-2][0])):
                        if((err[-2][2] > err[-3][2]) and (err[-2][0] < err[-3][0])):
                                break
            if(until!=None):
                if(err[-1][0] < until):
                        break
            if(best and e > 0):
                if(err[-1][1] < bestv):
                        bestv = err[-1][1] + 0.0
                        dnn.save("temp%d" % tempcode)
    
    if(best):
        try:
            dnn.load("temp%d" % tempcode) 
            for file in dnn.files("temp%d" % tempcode):
                os.remove(file)
        except:
            None
    clock.stop()
    if(verbose):
        print("\nTraining complete. Elapsed time: " + clock.elapsedTime() + ".")
    if(dropout>0.0):
        dnn.unfreeze()
    err = numpy.stack(err)
    return err, clock.elapsed()

def to01(T):
    vmax = T.max(axis = 0)[0]
    vmin = T.min(axis = 0)[0]
    return ((T-vmin)/(vmax-vmin))
