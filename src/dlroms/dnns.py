# Written by: Nicola Rares Franco, Ph.D. (MOX, Department of Mathematics, Politecnico di Milano)
# 
# Scientific articles based on this Python package:
#
# [1] Franco et al., Mathematics of Computation (2023).
#     A deep learning approach to reduced order modelling of parameter dependent partial differential equations.
#     DOI: https://doi.org/10.1090/mcom/3781.
#
# [2] Franco et al., Neural Networks (2023).
#     Approximation bounds for convolutional neural networks in operator learning.
#     DOI: https://doi.org/10.1016/j.neunet.2023.01.029
#
# [3] Franco et al., Journal of Scientific Computing (2023). 
#     Mesh-Informed Neural Networks for Operator Learning in Finite Element Spaces.
#     DOI: https://doi.org/10.1007/s10915-023-02331-1
#
# [4] Vitullo, Colombo, Franco et al., Finite Elements in Analysis and Design (2024).
#     Nonlinear model order reduction for problems with microstructure using mesh informed neural networks.
#     DOI: https://doi.org/10.1016/j.finel.2023.104068
#
# Please cite the Author if you use this code for your work/research.

import numpy
import torch
import matplotlib.pyplot as plt
from time import perf_counter     
from dlroms.cores import CPU, GPU
from IPython.display import clear_output
from copy import deepcopy
import os
import warnings

ReLU = torch.nn.ReLU()
leakyReLU = torch.nn.LeakyReLU(0.1) 
num2p = lambda prob : ("%.2f" % (100*prob)) + "%"
        
class Layer(torch.nn.Module):
    """Abstract class for representing layers of neural network models. It is implemented as a subclass of 'torch.nn.Module'.
    
    Attributes:
            core         (dlroms.cores.Core)        Specifies wheather the layer is stored on CPU or GPU.
            rho          (function)                 Activation function at output.
            training     (bool)                     Whether the layer is in training mode (weights and biases are learnable,
                                                    and they can be optimized) or not.
    
    All objects of this class should implement an abstract method '.module()' that returns the underlying torch.nn.Module.
    Layers can be applied to tensors using .forward() or simply .(), i.e. via the __call__ method.
    """
    
    def __init__(self, activation):
        """Creates a new layer with a given activation function (activation). By default, the layer is initiated on CPU.

        Input:
                activation        (function)        Activation function. Given a torch.tensor, it should output another torch.tensor.
                                                    Warning: only use basic operations or torch functions to ensure that autodifferentiation
                                                    is available.
        
        """
        super(Layer, self).__init__()
        self.rho = activation
        if(activation == None):
            self.rho = torch.nn.Identity()
        self.core = CPU
        self.training = True

    def coretype(self):
        """Core where the layer is stored (either CPU or GPU), returned as a dlroms.cores.Core object."""
        return self.core

    def to(self, core):
        super(Layer, self).to(core.device)
        
    def w(self):
        """Weights of the layer."""
        return self.module().weight
    
    def b(self):
        """Bias vector of the layer."""
        return self.module().bias
    
    def scale(self, factor):
        """Sets the bias vector to zero and scales the weight matrix by 'factor'.

        Input:
                factor        (float)        Value for scaling the weight matrix.
        """
        self.load(factor*self.w().detach().cpu().numpy(), 0.0*self.b().detach().cpu().numpy())
        
    def zeros(self):
        """Sets to zero all weights and biases."""
        self.module().weight = torch.nn.Parameter(0.0*self.module().weight)
        self.module().bias = torch.nn.Parameter(0.0*self.module().bias)
        
    def moveOn(self, core):
        """Transfers the layer structure onto the specified core.
        
        Input:
                core        (dlroms.cores.Core)        Where to transfer the layer (e.g., core = dlroms.cores.GPU)
        """
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
        """Squared sum of all weights within the layer. Supports backpropagation."""
        return (self.module().weight**2).sum()
    
    def l1(self):
        """Absolute sum of all weights within the layer. Supports backpropagation."""
        return self.module().weight.abs().sum()
    
    def outdim(self, inputdim):
        """Output dimension for a given input.
        
        Input:
                inputdim        (tuple)        Tuple specifying the input dimension. E.g., inputdim = (100, 3) corresponds
                                               to an input of shape 100 x 3. 
        """
        return tuple(self.forward(self.core.zeros(*inputdim)).size())
        
    def load(self, w, b = None):
        """Loads a given a pair of weights and biases as model parameters.
        
        Input:
                w        (numpy.ndarray)        Weights to be loaded. It should be of the correct shape, depending on the layer at hand.
                b        (numpy.ndarray)        Bias vector to be loaded. If None, the layer's bias remains unchanged. Defaults to None.
        """
        self.module().weight = torch.nn.Parameter(self.core.tensor(w))
        if(not(b is None)):
            self.module().bias = torch.nn.Parameter(self.core.tensor(b))
        
    def inherit(self, other, azzerate = True):
        """Inherits the weights and biases from another network. If needed, additional entries are left to zero.
        It can be seen as a naive form of transfer learning. NB: acts as an IN PLACE operation.
        
        Input:
                other        (dlroms.dnns.Layer)        Layer from which the parameters are inherited.
                azzerate     (bool)                     If True, all noninherited entries are set to zero.
        """
        if(azzerate):
            self.zeros()
        with torch.no_grad():
            where = tuple([slice(0,s) for s in other.w().size()])
            self.module().weight[where] = torch.nn.Parameter(self.core.tensor(other.w().detach().cpu().numpy())) 
            where = tuple([slice(0,s) for s in other.b().size()])
            self.module().bias[where] = torch.nn.Parameter(self.core.tensor(other.b().detach().cpu().numpy())) 

        
    def __add__(self, other):
        """Connects two layers sequentially. Also supports connecting a layer with a more complicate neural network model.
       
        Input:
                self        (dlroms.dnns.Layer)                                   Current layer
                other       (dlroms.dnns.Layer or dlroms.dnns.Compound)        Architecture to be connected on top of 'self'.
        
        Output:
                (dlroms.dnns.Consecutive) Combined architecture.
        """
        if(isinstance(other, Consecutive)):
            n = len(other)
            layers = [self]+[other[i] for i in range(n)]
            return Consecutive(*tuple(layers))
        else:
            if(other == 0.0):
                return self
            else:
                return Consecutive(self, other)    
            
    def __pow__(self, n):
        """Creates a branched architecture by stacking n INDEPENDENT copies of the same layer. See dlroms.dnns.Branched.
        
        Input:
                self        (dlroms.dnns.Layer)        Reference layer to be copied.
                n           (int)                      Number of copies to be stacked (in parallel, with shared input).

        Output:
                (dlroms.dnns.Branched) Combined architecture.
        """
        return Branched(*[deepcopy(self) for i in range(n)])
    
    def __mul__(self, n):
        """Creates a deep neural network by connecting n (INDEPENDENT) consecutive copies of the same layer.
                
        Input:
                self        (dlroms.dnns.Layer)        Reference layer to be copied.
                n           (int)                      Number of copies to be stacked (sequentially).

        Output:
                (dlroms.dnns.Consecutive) Combined architecture.
        """
        return Consecutive(*[deepcopy(self) for i in range(n)])
    
    def __rmul__(self, number):
        """See self.__mul__."""
        return self*number
    
    def dof(self):
        """Degrees of freedom in the layer, defined as the number of learnable weights and biases.

        Output:
                (int) Total number of learnable parameters.        
        """
        return numpy.prod(tuple(self.module().weight.size())) + len(self.module().bias)
                     
    def He(self, linear = False, a = 0.1, seed = None):
        """He initialization of the weights: see 'He, K., Zhang, X., Ren, S., & Sun, J. (2015). Delving deep into rectifiers:
        Surpassing human-level performance on imagenet classification. IEEE  Proceedings'.
        Note: to be used only for layers without activation at output or with leakyReLU activation.

        Input:
                linear        (bool)        Specifies whether the layer has no activation at output (self.rho = Identity). 
                                            If False, it assumes that the layer is equipped with the a-leakyReLU. 
                                            Defaults to False.

                a             (float)       Slope of the leakyReLU activation. Ignored if linear = True. Defaults to 0.1.

                seed          (int)         Random seed for reproducibility. Ignored if seed = None. Defaults to None.
        """
        if(len(self.parameters())>0):
                if(not(seed is None)):
                    torch.manual_seed(seed)
                if(linear):
                    torch.nn.init.xavier_normal_(self.module().weight)
                else:
                    torch.nn.init.kaiming_normal_(self.module().weight, mode='fan_out', nonlinearity='leaky_relu', a = a)
        
    def Xavier(self):
        """Xavier initialization of the weights: see 'Glorot, X., & Bengio, Y. (2010, March).
        Understanding the difficulty of training deep feedforward neural networks. JMLR Workshop and Conference Proceedings.'
        Recommended for layers with tanh activation.
        """
        torch.nn.init.xavier_uniform_(self.module().weight)
        
    def inputdim(self):
        """Expected input dimension (if well-defined: see, e.g., the case of convolutional layers, dlroms.dnns.Conv1D).

        Output:
                (int) dimension at input.
        """
        return self.module().in_features
    
    def freeze(self, w = True, b = True):
        """Freezes the layer, fixing its weights and biases (hereon nontrainable). To undo, see dlroms.dnns.Layer.unfreeze.
        
        Input:
                w         (bool)        wheather to fix the weights or not.
                b         (bool)        wheather to fix the bias or not.
        """
        if(w and b):
            self.module().requires_grad_(False)
        elif(w):
            self.module().weight.requires_grad_(False)
        elif(b):
            self.module().bias.requires_grad_(False)
        self.training = False
        
    def unfreeze(self):
        """Activates the layer, making all its parameters learnable. Acts in opposition to dlroms.dnns.Layer.freeze."""
        self.module().bias.requires_grad_(True)
        self.module().weight.requires_grad_(True)
        self.module().requires_grad_(True)
        self.training = True
        
    def dictionary(self, label = ""):
        """Dictionary listing all layer parameters.

        Input:
                label        (str)        Optional label to be included within the keys of the dictionary.

        Output:
                (dict)        Dictionary containing the learnable weights and biases of the layer.        
        """
        return {('w'+label):self.w().detach().cpu().numpy() + 0.0, ('b'+label):self.b().detach().cpu().numpy() + 0.0}
    
    def parameters(self):
        """List with all the learnable parameters within the layer.

        Output:
                (list).     
        """
        ps = list(super(Layer, self).parameters())
        res = []
        for p in ps:
            if(p.requires_grad):
                res.append(p)
        return res

class Dense(Layer):
    """Class implementing fully connected layers. Implemented as a subclass of dlroms.dnns.Layer.
    
    Attributes:
            lin         (torch.nn.Module)        Affine part of the layer (i.e., the learnable map x -> Wx + b).

    Other attributes: core, rho, training (see dlroms.dnns.Layer).
    """
    
    def __init__(self, input_dim, output_dim, activation = leakyReLU, bias = True):
        """Creates a Dense Layer with given input dimension, output dimension and activation function.

        Input:
                input_dim        (int)        Input dimension.
                output_dim       (int)        Output dimension.
                activation       (function)   Function (or callable object) to be used as terminal nonlinearity (cf. dlroms.dnns.Layer).
                                              Defaults to the 0.1-leakyReLU activation.
                bias             (bool)       Whether to include a bias vector or not. If False, this is equivalent to having a
                                              nonlearnable bias that is always equal to 0.

        Output:
                (dlroms.dnns.Dense).
        """
        super(Dense, self).__init__(activation)
        in_d = input_dim if(isinstance(input_dim, int)) else input_dim.dim()            
        out_d = output_dim if(isinstance(output_dim, int)) else output_dim.dim()
        self.lin = torch.nn.Linear(in_d, out_d, bias = bias)
        
    def module(self):
        """Underlying torch.nn.Module.
        
        Output:
                (torch.nn.Module).
        """
        return self.lin
        
    def forward(self, x):
        """Maps a given input x through the layer. The output is computed as rho(Wx+b), where W and b are
        the weight matrix and the bias vector, respectively, while rho is the activation function of the layer.

        Note: as all DNN modules, it operates in batches, i.e.: it expects an input of shape (batch_size, input_dim)
        and it outputs a corresponding tensor of shape (batch_size, output_dim), obtained by applying the layer to 
        each input (independently).

        Input:
                x        (torch.Tensor)        Tensor of shape (batch_size, input_dim).

        Output:
                (torch.Tensor)        Output tensor with shape (batch_size, output_dim).        
        """
        return self.rho(self.lin(x))
    
class Residual(Layer):
    """Class implementing residual layers. Differently from dense layers, which act as x -> f(x), these
    operate as x -> x + f(x). Implemented as a subclass of dlroms.dnns.Layer.

    Attributes:
            lin         (torch.nn.Module)        Affine part of the layer (i.e., the learnable map x -> Wx + b).

    Other attributes: core, rho, training (see dlroms.dnns.Layer).
    
    """
    
    def __init__(self, dim, activation = leakyReLU):
        """Creates a Dense Layer with given input dimension, output dimension and activation function.

        Input:
                input_dim        (int)        Input dimension.
                output_dim       (int)        Output dimension.
                activation       (function)   Function (or callable object) to be used as terminal nonlinearity (cf. dlroms.dnns.Layer).
                                              Defaults to the 0.1-leakyReLU activation.

        Output:
                (dlroms.dnns.Residual).
        """
        super(Residual, self).__init__(activation)
        self.lin = torch.nn.Linear(dim, dim)
        
    def module(self):
        """Underlying torch.nn.Module.
        
        Output:
                (torch.nn.Module).
        """
        return self.lin
    
    def forward(self, x):
        """Maps a given input x through the layer. The output is computed as x+rho(Wx+b), where W and b are
        the weight matrix and the bias vector, respectively, while rho is the activation function of the layer.

        Note: as all DNN modules, it operates in batches, i.e.: it expects an input of shape (batch_size, input_dim)
        and it outputs a corresponding tensor of shape (batch_size, output_dim), obtained by applying the layer to 
        each input (independently).

        Input:
                x        (torch.Tensor)        Tensor of shape (batch_size, input_dim).

        Output:
                (torch.Tensor)        Output tensor with shape (batch_size, output_dim).        
        """
        return x + self.rho(self.lin(x))   

class Sparse(Layer):
    """Class implementing sparse layers, that is, architectures obtained by pruning dense modules.
    Implemented as a subclass of dlroms.dnns.Layer.
    
    Attributes:
            loc         (tuple of numpy.ndarray)        Indexes of the active entries of the weight matrix.
                                                        The kth active entry is located at position j[k], i[k]
                                                        with j[k] = loc[0][k], i[k] = loc[1][k].
            in_d        (int)                           Input dimension.
            out_d       (int)                           Output dimension.
            weight      (torch.nn.Parameter)            Learnable nonzero entries of the weight matrix, listed sequentially
                                                        in a tensor of shape (n,).
            bias        (torch.nn.Parameter)            Learnable bias vector.

    Other attributes: core, rho, training (see dlroms.dnns.Layer).
    """
    
    def __init__(self, mask, activation = leakyReLU):
        """Creates a Sparse Layer with given sparsity pattern and a given activation function.

        Input:
                mask             (numpy.ndarray)        M x N array with float entries. It should be defined so that M is
                                                        the input dimension of the layer and N is the output dimension.
                                                        This matrix defines the sparsity pattern of layer, via the rule
                                                        "if M_ij = 0 then the jth component at output will be 
                                                        uneffected by the ith entry at input." Equivalently, if W is the N x M
                                                        tensor representing the weight matrix of the layer: "M_ij = 0 implies 
                                                        W_ji=0 and W_ji is nonlearnable."
                activation       (function)             Function (or callable object) to be used as terminal nonlinearity
                                                        (cf. dlroms.dnns.Layer). Defaults to the 0.1-leakyReLU activation.

        Output:
                (dlroms.dnns.Sparse).
        """
        super(Sparse, self).__init__(activation)
        self.loc = numpy.nonzero(mask)
        self.in_d, self.out_d = mask.shape
        self.weight = torch.nn.Parameter(CPU.zeros(len(self.loc[0])))
        self.bias = torch.nn.Parameter(CPU.zeros(self.out_d))
        
    def moveOn(self, core):
        """Transfers the layer structure onto the specified core.
        
        Input:
                core        (dlroms.cores.Core)        Where to transfer the layer (e.g., core = dlroms.cores.GPU)
        """
        self.core = core
        with torch.no_grad():
            if(core == GPU):      
                self.weight = torch.nn.Parameter(self.weight.cuda())
                self.bias = torch.nn.Parameter(self.bias.cuda())
            else:
                self.weight = torch.nn.Parameter(self.weight.cpu())
                self.bias = torch.nn.Parameter(self.bias.cpu())
        
    def module(self):
        """Returns the layer it self. This is done to ensure that calls such as self.module().weight have a result
        compatible with the one obtained for the other classes.
        
        Output:
                (dlroms.dnns.Sparse).
        """
        return self
    
    def forward(self, x):
        """Maps a given input x through the layer. The output is computed as rho(Wx+b), where W and b are
        the weight matrix and the bias vector, respectively, while rho is the activation function of the layer.

        Note: as all DNN modules, it operates in batches, i.e.: it expects an input of shape (batch_size, input_dim)
        and it outputs a corresponding tensor of shape (batch_size, output_dim), obtained by applying the layer to 
        each input (independently).

        Input:
                x        (torch.Tensor)        Tensor of shape (batch_size, input_dim).

        Output:
                (torch.Tensor)        Output tensor with shape (batch_size, output_dim).        
        """
        W = self.core.zeros(self.in_d, self.out_d)
        W[self.loc] = self.weight
        return self.rho(self.bias + x.mm(W))
    
    def inherit(self, other):
        """Inherits the weights and biases from another Sparse network with comparable input-output dimensions
        (but possibly different sparsity patterns - typically "smaller"). Additional entries in the weight matrix are left to zero.
        It can be seen as a naive form of transfer learning. NB: acts as an IN PLACE operation.
        
        Input:
                other        (dlroms.dnns.Sparse)       Sparse Layer from which the parameters are inherited.
        """
        W = self.core.zeros(self.in_d, self.out_d)
        W[other.loc] = other.weight
        with torch.no_grad():
            self.weight = torch.nn.Parameter(self.core.copy(W[self.loc]))
            self.bias = torch.nn.Parameter(self.core.copy(other.bias))        

    def He(self, linear = False, a = 0.1, seed = None):
        """Generalization of the He initialization for Sparse Layers, as proposed in 'Franco, N. R., Manzoni, A., & Zunino, P. (2023).
        Mesh-informed neural networks for operator learning in finite element spaces. Journal of Scientific Computing, 97(2), 35.'
        Only recommended for layers that use leakyReLU activations (or no activation at all).

        Input:
                linear        (bool)        Specifies whether the layer has no activation at output (self.rho = Identity). 
                                            If False, it assumes that the layer is equipped with the a-leakyReLU. 
                                            Defaults to False.

                a             (float)       Slope of the leakyReLU activation. Ignored if linear = True. Defaults to 0.1.

                seed          (int)         Random seed for reproducibility. Ignored if seed = None. Defaults to None.
        """
        c = 1 if linear else 2
        alpha = 0 if linear else a
        A = numpy.zeros((self.out_d, self.in_d))
        A[(self.loc[1], self.loc[0])] = 1
        nnz = numpy.sum(A>0, axis = 1)[self.loc[1]]
        if(not (seed is None)):
            numpy.random.seed(seed)    
        with torch.no_grad():
            self.weight = torch.nn.Parameter(self.core.tensor(numpy.random.randn(len(self.loc[0]))*numpy.sqrt(c/(nnz*(1.0+alpha**2)))))
        
    def deprecatedHe(self, linear = False, a = 0.1, seed = None):
        """Deprecated initialization routine. See dlroms.dnns.Sparse.He."""
        nw = len(self.weight)
        with torch.no_grad():
            self.weight = torch.nn.Parameter(torch.rand(nw)/numpy.sqrt(nw))
        
    def Xavier(self):
        """Generalization of the Xavier initialization for Sparse Layers, as proposed in 'Franco, N. R., Manzoni, A., & Zunino, P. (2023).
        Mesh-informed neural networks for operator learning in finite element spaces. Journal of Scientific Computing, 97(2), 35.'
        Only recommended for layers that use nonlinear activations different from leakyReLUs (e.g., tanh, sigmoid).
        """
        A = numpy.zeros((self.out_d, self.in_d))
        A[(self.loc[1], self.loc[0])] = 1
        nnz = numpy.sum(A>0, axis = 1)[self.loc[1]]
        with torch.no_grad():
            self.weight = torch.nn.Parameter(self.core.tensor((2*numpy.random.rand(len(self.loc[0]))-1)*numpy.sqrt(3/nnz)))
                 
    def W(self):
        """Transposed weight matrix of the layer (with both zero and nonzero entries).
        
        Output:
                (torch.Tensor)        Matrix of shape input_dim x output_dim.
        
        """
        W = self.core.zeros(self.in_d, self.out_d)
        W[self.loc] = self.weight
        return W

    def dictionary(self, label = ""):
        """Dictionary listing all layer parameters, together with their sparsity pattern.

        Input:
                label        (str)        Optional label to be included within the keys of the dictionary.

        Output:
                (dict)        Dictionary containing the learnable weights and biases of the layer, plus the indexes associated
                              to the active entries of the weight matrix.        
        """
        return {('w'+label):self.w().detach().cpu().numpy() + 0.0, ('b'+label):self.b().detach().cpu().numpy() + 0.0, ('indexes'+label):self.loc}

    def load(self, w, b = None, indexes = None):
        """Loads a given a pair of weights and biases as model parameters. The indexes dictating the sparsity pattern (and thus the
        collocation of the learnable weight entries) can also be loaded.
        
        Input:
                w        (numpy.ndarray)        Weights to be loaded. It should be of the correct length, depending on the layer at hand.
                b        (numpy.ndarray)        Bias vector to be loaded. If None, the layer's bias remains unchanged. Defaults to None.
                indexes  (numpy.ndarray)        2 x n array listing the i,j coordinates of the active entries. Here, n should equal
                                                the length of w.
        """
        super(Sparse, self).load(w, b)
        if(isinstance(indexes, numpy.ndarray)):
            self.loc = list(indexes)
        
    def cuda(self):
        """Transfers the layer to the GPU."""
        self.moveOn(GPU)
        
    def cpu(self):
        """Transfers the layer to the CPU."""
        self.moveOn(CPU)
        
class Weightless(Layer):
    """Abstract class for nonlearnable layers, implemented as a subclass of 'dlroms.dnns.Layers'.

    All objects of this class should implement a proper '.forward()' method.
    """
    
    def __init__(self):
        """Creates a Weightless layer.
        """
        super(Weightless, self).__init__(None)
        self.rho = None
        self.core = CPU
        self.training = False
        
    def inherit(self, other):
        """Leaves the layer unchanged. Overwrite when implementing a subclass that requires inheriting 
        certain nonlearnable parameters."""
        None
        
    def scale(self, factor):
        """Leaves the layer unchanged."""
        None
    
    def w(self):
        """Returns None (Weightless layers have no weights).
        
        Output:
                (None).
        """
        return None
    
    def b(self):
        """Returns None (Weightless layers have no bias).
        
        Output:
                (None).
        """
        return None
    
    def zeros(self):
        """Leaves the layer unchanged."""
        None
        
    def cuda(self):
        """Switches to GPU. Overwrite when implementing a subclass that uses nonlearnable tensors."""
        self.core = GPU
        None
        
    def cpu(self):
        """Switches to CPU. Overwrite when implementing a subclass that uses nonlearnable tensors."""
        self.core = CPU
        None
        
    def l2(self):
        """Returns 0.0 (Weightless layers have 0 l2-norm, as they do not come with learnable parameters).

        Output:
                (float).
        """
        return 0.0
    
    def l1(self):
        """Returns 0.0 (Weightless layers have 0 l1-norm, as they do not come with learnable parameters).

        Output:
                (float).
        """
        return 0.0
        
    def dof(self):
        """Returns 0 (Weightless layers have no dof, as they do not come with learnable parameters)."""
        return 0
    
    def He(self, linear, a, seed):            
        """Leaves the layer unchanged."""
        None
        
    def Xavier(self):
        """Leaves the layer unchanged."""
        None
        
    def freeze(self, w = True, b = True):
        """Leaves the layer unchanged."""
        None
        
    def unfreeze(self):
        """Leaves the layer unchanged."""
        None
        
    def load(self, w, b):
        """Leaves the layer unchanged. Overwrite when implementing a subclass that requires loading of certain attributes."""
        None
        
    def dictionary(self, label):
        """Returns an empty dictionary (Weightless layers have no learnable parameters).

        Output:
                (dict).
        """
        return dict()

class Matrix(Weightless):
    def __init__(self, A):
        super(Matrix, self).__init__()
        self.At = A.T
    def forward(self, x):
        return x @ self.At

class Reshape(Weightless):
    """Class implementing reshape layers, that is, nonlearnable modules that reshape tensors.
    Implemented as a subclass of dlroms.dnns.Weightless.
    
    Attributes:
            newdim      (tuple of int)        Output shape (excluding batch dimension).

    Other attributes: core, rho (None), training (False) (see dlroms.dnns.Weightless).
    """    

    def __init__(self, *shape):
        """Creates a Reshape layer with newdim = shape."""
        super(Reshape, self).__init__()
        self.newdim = shape
        
    def forward(self, x):
        """Reshapes the input tensor x.

        Input:
                x        (torch.Tensor)        Tensor to be reshaped.

        Output:
                (torch.Tensor) reshaped version of x.
        
        Note: as all DNN modules, it operates in batches, i.e.: it expects an input of shape (batch_size, d1,..., dk)
        and it outputs a corresponding tensor of shape (batch_size, p1,..., pj), where self.newdim = (p1,...,pj).
        That is, all input instances of dimension (d1,..., dk) are reshaped independently. 
        Note that, for this to work, one must ensure that d1*...*dk = p1*...*pj.           
        """
        n = len(x)
        return x.reshape(n,*self.newdim)    
    
    def inputdim(self):
        """Expected input dimension. Note: any shape compatible with such dimension can be passed as input.

        Output:
                (int) dimension at input.
        """
        return numpy.prod(self.newdim)
    
class Inclusion(Weightless):
    """Class implementing nonlearnable layers that embedd R^m into R^n, m < n, with x -> [0,...,0, x].
    Implemented as a subclass of dlroms.dnns.Weightless.
    
    Attributes:
            d      (int)        Number of new zero entries to be appended.

    Other attributes: core, rho (None), training (False) (see dlroms.dnns.Weightless).
    """   
        
    def __init__(self, in_dim, out_dim):
        """Creates an Inclusion layer for specific input and output dimensions.

        Input:
                in_dim        (int)        Input dimension.
                out_dim       (int)        Output dimension.        
        """
        super(Inclusion, self).__init__()
        self.d = out_dim - in_dim
        if(self.d<=0):
            raise RuntimeError("Output dimension of Inclusion layers should be strictly larger than the input dimension.")
        
    def forward(self, x):
        """Embedds the input tensor x onto a larger spaces by zero padding.

        Input:
                x        (torch.Tensor)        Tensor to be embedded.

        Output:
                (torch.Tensor) embededd/padded version of x.
        
        Note: as all DNN modules, it operates in batches, i.e.: it expects an input of shape (batch_size, input_dim)
        and it outputs a corresponding tensor of shape (batch_size, input_dim+self.d).
        That is, all input instances of dimension (input_dim,) are padded independently.       
        """
        z = self.core.zeros(len(x), self.d)
        return torch.cat((z, x), axis = 1)
    
class Trunk(Weightless):
    """Class implementing nonlearnable layers that trunkate inputs, [x_{1}, ..., x_{n}] -> [x_{1},..., x_{n-d}].
    Implemented as a subclass of dlroms.dnns.Weightless.
    
    Attributes:
            d      (int)        Number of entries to be kept (starting from the first one).

    Other attributes: core, rho (None), training (False) (see dlroms.dnns.Weightless).
    """   
    def __init__(self, out_dim):
        """Creates a Trunk layer with a given output dimension.

        Input:
                out_dim       (int)        Output dimension.        
        """
        super(Trunk, self).__init__()
        self.d = out_dim
        
    def forward(self, x):
        """Trunks the input tensor x by dropping extra entries.

        Input:
                x        (torch.Tensor)        Tensor to be truncated.

        Output:
                (torch.Tensor) trunked tensor.
        
        Note: as all DNN modules, it operates in batches, i.e.: it expects an input of shape (batch_size, input_dim)
        and it outputs a corresponding tensor of shape (batch_size, self.d).
        That is, all input instances of dimension (input_dim,) are trunked independently.     
        Note that, for this to work properly, one should ensure that self.d <= input_dim.
        """
        return x[:,:self.d]

class Transpose(Weightless):
    """Class implementing nonlearnable layers that transpose their input.
    Implemented as a subclass of dlroms.dnns.Weightless.
    
    Attributes:
            d0      (int)        dimension to be transposed with dim1 (ignoring batch dimension!).
            d1      (int)        dimension to be transposed with dim0 (ignoring batch dimension!).

    Other attributes: core, rho (None), training (False) (see dlroms.dnns.Weightless).
    """   
    def __init__(self, dim0, dim1):
        """Creates a Transpose layer operating on the given dimensions.

        Input:
                dim0      (int)        dimension to be transposed with dim1 (ignoring batch dimension!).
                dim1      (int)        dimension to be transposed with dim0 (ignoring batch dimension!).     
        """
        super(Transpose, self).__init__()
        self.d0 = dim0
        self.d1 = dim1
        
    def forward(self, x):
        """Transposes the input tensor x along the dimensions self.d0 and self.d1 (ignoring batch dimension).

        Input:
                x        (torch.Tensor)        Tensor to be transposed.

        Output:
                (torch.Tensor) transposed tensor.
        
        Note: as all DNN modules, it operates in batches, i.e.: it expects an input of shape (batch_size, d_1, ..., d_n)
        and it outputs a corresponding tensor of shape (batch_size, p_1, ..., p_n) where p_j=d_j if j!=self.d0, self.d1, 
        and p_self.d0 = d_self.d1, p_self.d1 = d_self.d0. That is, all input instances of dimension (d_1, ..., d_n) are
        transposed independently. For instance, if x has shape (N, 5, 3, 8) and f = Transpose(0, 2), then f(x) has shape
        (N, 8, 3, 5).
        """
        return x.transpose(dim0 = self.d0+1, dim1 = self.d1+1)

class Fourier(Weightless):
    def __init__(self, freqs, which = None):
        super(Fourier, self).__init__()
        self.k = freqs
        self.which = which
        
    def forward(self, x):
        z = [x[:,j] for j in range(x.shape[1])]
        js = range(x.shape[1]) if self.which is None else self.which
        for j in js:
            for k in range(1, self.k+1):
                z.append((k*x[:, j]).cos())
                z.append((k*x[:, j]).sin())
        return torch.stack(z, axis = 1)
        
class Convolutional(Layer):
    def module(self):
        return self.conv
            
    def forward(self, x):
        return self.rho(self.conv(x))    

    def inputdim(self):
        raise RuntimeError("Convolutional layers do not have a fixed input dimension.")   

class Deconvolutional(Layer):
    def module(self):
        return self.deconv
            
    def forward(self, x):
        return self.rho(self.deconv(x))    

    def inputdim(self):
        raise RuntimeError("Deconvolutional layers do not have a fixed input dimension.")  

class Conv2D(Convolutional):
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
    
class Deconv2D(Deconvolutional):
    """Layer that performs a transposed 2D convolution (cf. Pytorch documentation, torch.nn.ConvTranspose2d)."""
    
    def __init__(self, window, channels = (1,1), stride = 1, padding = 0,  groups = 1, dilation = 1, activation = leakyReLU):
        """Creates a Deconvolutional2D layer. Arguments read as in Convolutional2D.__init__."""
        super(Deconv2D, self).__init__(activation)
        self.deconv = torch.nn.ConvTranspose2d(channels[0], channels[1], window, stride = stride, padding = padding, groups = groups, dilation = dilation)

class Conv3D(Convolutional):
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
    
class Deconv3D(Deconvolutional):
    """Layer that performs a transposed 3D convolution (cf. Pytorch documentation, torch.nn.ConvTranspose3d)."""
    
    def __init__(self, window, channels = (1,1), stride = 1, padding = 0,  groups = 1, dilation = 1, activation = leakyReLU):
        """Creates a Deconvolutional3D layer. Arguments read as in Convolutional3D.__init__."""
        super(Deconv3D, self).__init__(activation)
        self.deconv = torch.nn.ConvTranspose3d(channels[0], channels[1], window, stride = stride, padding = padding, groups = groups, dilation = dilation)

class Conv1D(Convolutional):    
    """Analogous to Convolutional2D but considers 1D convolutions."""
    def __init__(self, window, channels = (1,1), stride = 1, padding = 0,  groups = 1, dilation = 1, activation = leakyReLU):
        super(Conv1D, self).__init__(activation)
        self.conv = torch.nn.Conv1d(channels[0], channels[1], window, stride = stride, padding = padding, groups = groups, dilation = dilation)
        
class Deconv1D(Deconvolutional):    
    """Analogous to Deconvolutional2D but considers trasposed 1D convolutions."""
    def __init__(self, window, channels = (1,1), stride = 1, padding = 0,  groups = 1, dilation = 1, activation = leakyReLU):
        super(Deconv1D, self).__init__(activation)
        self.deconv = torch.nn.ConvTranspose1d(channels[0], channels[1], window, stride = stride, padding = padding, groups = groups, dilation = dilation)

        
class Compound(torch.nn.Sequential):
    """Class that handles deep neural networks, obtained by connecting arbitrary architectures in multiple ways. 
    It is implemented as a subclass of torch.nn.Sequential.
    
    Objects of this class support indexing, so that self[k] returns the kth Layer (or sub-module) in the architecture.
    """
    def forward(self, *args):
        raise RuntimeError("No forward method specified!")
                
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
            if(isinstance(nn, Compound)):
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
            Compound(*self.stretch()).save(path, label)

    def write(self, label = ""):
        """Equivalent to self.dictionary(label), up to stretching of the architecture."""
        if(len(self) == len(self.stretch())):
            return self.dictionary(label)
        else:
            return Compound(*self.stretch()).write(label)

    def read(self, params, label = ""):
        """Equivalent to self.load(label) but relies on data available within the dictionary 'params', rather than on locally stored data."""
        if(len(self) == len(self.stretch())):
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
             Compound(*self.stretch()).read(params)
        
    def load(self, path, label = ""):
        """Loads the architecture parameters from stored data.
        
        Input:
        path (string): system path where the parameters are stored.
        label (string): additional label required if the stored data had one.
        """
        try:
            params = numpy.load(path)
        except:
            params = numpy.load(path+".npz")
        self.read(params, label)

    def download(self, gdrive_link):
        """Downloads architecture parameters from Google Drive.

        Input:
        gdrive_link (string): Google drive link
        """
        import gdown
        import os
        gdrive_id = gdrive_link[(gdrive_link.find("/d/")+3):gdrive_link.find("/view")]
        random_id = int(numpy.random.rand()*10000)
        filename = "temp_dnn_%d.npz" % random_id
        gdown.download(id = gdrive_id, output = filename, quiet=False)
        clear_output()
        self.load(filename)
        os.remove(filename)        
                
    def __add__(self, other):
        """Augments the current architecture by connecting it with a second one.
        
        Input:
        self (Consecutive): current architecture.
        other (Layer / Consecutive): neural network to be added at the end.
        
        Output:
        A Consecutive object consisting of the nested neural network self+other. 
        """
        if(isinstance(other, Consecutive)):
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

    def __pow__(self, n):
        """Creates a branched architecture by stacking n copies of the same layer."""
        return Branched(*[deepcopy(self) for i in range(n)])
            
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
        """Inherits the networks parameters from a given architecture (cf. dlroms.dnns.Layer.inherit). 
        The NN 'other' should have a depth less or equal to that of 'self'.
        
        Input:
        other (Consecutive): the architecture from which self shoud learn."""
        for i, nn in enumerate(other):
            if(type(self[i])==type(nn)):
                self[i].inherit(nn, azzerate)
                
    def files(self, string):
        return [string+".npz"]

    def jacobian(self, x, keep_diff = False):
        return torch.autograd.functional.jacobian(self, x, create_graph = keep_diff, strict = keep_diff)

class Consecutive(Compound):
    """Architecture with multiple layers that work sequentially. Implemented as a subclass of dlroms.dnns.Compound.
    If f1,...,fk is the collection of layers, then Consecutive(f1,..,fk)(x) = fk(...(f2(f1(x))))."""   
    def forward(self, x):
        for nn in self:
            x = nn(x)
        return x  
            
class Parallel(Compound):
    """Architecture with multiple layers that work in parallel but channel-wise. Implemented as a subclass of dlroms.dnns.Compound.
    If f1,...,fk is the collection of layers, then Parallel(f1,..,fk)(x) = [f1(x1),..., fk(xk)], where x = [x1,...,xk] is
    structured in k channels."""   
    def forward(self, x):
        res = [self[k](x[:,k]) for k in range(len(self))]
        return torch.stack(res, axis = 1)    
    
class Channelled(Compound):
    """Architecture with multiple layers that work in parallel but channel-wise. Implemented as a subclass of dlroms.dnns.Compound.
    If f1,...,fk is the collection of layers, then Channelled(f1,..,fk)(x) = f1(x1)+...+fk(xk), where x = [x1,...,xk] is
    structured in k channels."""        
    def forward(self, x):
        res = 0.0
        k = 0
        for f in self:
            res = res + f(x[:,k])    
            k += 1
        return res

class Branched(Parallel):
    """Architecture with multiple layers that work in parallel but have a common root. Implemented as a subclass of dlroms.dnns.Parallel.
    If f1,...,fk is the collection of layers, then Parallel(f1,..,fk)(x) = [f1(x),..., fk(x)]."""
    def forward(self, x):
        res = [self[k](x) for k in range(len(self))]
        return torch.stack(res, axis = 1)

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
            return ("%dh %dm %.2fs" % (h,m,s))
        elif(m>0):
            return ("%dm %.2fs" % (m,s))
        else:
            return ("%.2fs" % s)
        
        
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
