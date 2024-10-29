# DLROMs: a Python package for Deep Learning based Reduced Order Models


**DLROMs** is a deep learning library for model order reduction that leverages on packages such as Pytorch and FEniCS. It comes with a basic version (no FEniCS required) and an advanced one. The techniques made available within this library are reminiscent of the following scientific articles: 

- [A deep learning approach to reduced order modelling of parameter dependent partial differential equations](https://doi.org/10.1090/mcom/3781), Franco et al., *Mathematics of Computation*, 92 (340), 483-524 (2023).
     
- [Approximation bounds for convolutional neural networks in operator learning](https://doi.org/10.1016/j.neunet.2023.01.029), Franco et al., *Neural Networks*, 161: 129-141 (2023).
     
- [Mesh-Informed Neural Networks for Operator Learning in Finite Element Spaces](https://doi.org/10.1007/s10915-023-02331-1), Franco et al., *Journal of Scientific Computing*, 97(2), 35 (2023).

This library was written, and is currently maintained, by **Nicola Rares Franco**, **Ph.D.**, MOX, Politecnico di Milano. For a tensorflow alternative, we recommend the repositories by [Stefania Fresca, Ph.D](https://github.com/stefaniafresca).


NB: if you are using this repository for your own research, please cite as

     Franco, N. R. (2024). NicolaRFranco/dlroms: First release (v1.0.0). Zenodo.
     https://doi.org/10.5281/zenodo.13254758


## DL-ROMs in a nutshell

Deep learning based reduced order models are efficient model surrogates that can emulate the accuracy of classical numerical solvers (hereby referred to as FOM: Full Order Models) by learning from high-quality simulations. For instance, let $\boldsymbol{\mu}\to u$ represent the action of a FOM solver, which, given a parameter instance $\boldsymbol{\mu}\in\mathbb{R}^{p}$ returns the corresponding PDE solution $u\in\mathbb{R}^{N_{h}}$, here represented by means of a suitable dof vector. Then, the construction of a DL-ROM can be sketched as:

1. **Sampling**: collect high-fidelity data, $`M=[\boldsymbol{\mu}_{1};\dots;\boldsymbol{\mu}_{N}]\in\mathbb{R}^{N\times p}`$ and $`U=[u_{1};\dots;u_{N}]\in\mathbb{R}^{N\times N_{h}}`$, by repeatedly querying the FOM solver.
2. **Design**: initialize a DL-ROM with trainable architectures $\psi_{1},\dots,\psi_{k}$.
3. **Training**: optimize the components of the DL-ROM, thus learning to replicate the FOM.
4. **Operate**: freeze the DL-ROM and use it freely at a negligible computational cost.

In the most simple setting, a DL-ROM may consist of a single DNN architecture $\psi:\mathbb{R}^{p}\to\mathbb{R}^{N_{h}}$ directly approximating the FOM, $\psi(\boldsymbol{\mu})\approx u$. The code below shows a simple example of such situation. Note, however, that this is a very naive approach, not suited for most applications: for more advanced approaches, such as POD-NN or autoencoder-enhanced DL-ROMs, we refer to the *dlroms.roms* module.

    # Problem data
    p, Nh = 2, 501

    # Toy example of a FOM solver: given mu = [m0, m1], it returns u(x) = (m1-m0)x+m0, 
    # i.e. the solution to: -u''=0 in (0,1), u(0)=m0, u(1)=m1. 
    # The solution u=u(x) is discretized over a uniform grid with 501 nodes.
    # NB: this is merely a didactical example. In practical applications, the FOM sol-
    # ver consists of an expensive numerical solver (otherwise, we wouldn't need a ROM
    # surrogate!).
    import numpy as np
    def FOMsolver(mu):
         x = np.linspace(0, 1, Nh)
         return (mu[1]-mu[0])*x + mu[0]

    # Data generation
    def sampler(seed):
         np.random.seed(seed)
         mu = np.random.rand(p)
         return mu, FOMsolver(mu)

    from dlroms.roms import snapshots, GPU # note: if GPU is note available, simply switch to CPU!
    n = 100
    M, U =  snapshots(n, sampler, core = GPU) 

    # DNN design and initialization
    from dlroms.dnns import Dense
    from dlroms.roms import DFNN
    psi = Dense(p, 10) + Dense(10, 10) + Dense(10, Nh, activation = None)
    model = DFNN(psi) # build a ROM with "psi" as a trainable object
    model.He()   # random initialization
    model.cuda() # trasfer to GPU

    # Model training
    from dlroms.roms import euclidean, mse
    ntrain = int(0.5*n) # snapshots for training: the remaining ones are used for testing
    model.train(M, U, ntrain = ntrain, loss = mse(euclidean), epochs = 50)
    model.freeze()

    # Use it online!
    newmu = [0.25, 0.4]
    model.solve(newmu) # single evaluation

    newmus = GPU.tensor([[0.25, 0.4], [0.7, 0.1]])
    model(newmus)      # multiple evaluations (faster than iterating)

    # Plot
    from dlroms import plot
    plot(FOMsolver(newmu))
    plot(model.solve(newmu), '--')

Starting from this simple pipeline, the DL-ROMs package provides access to a whole spectrum of more advanced techniques: complex customizable models based on multiple architecture that cooperate with each other, mesh-informed architectures, integral norms for Lebesgue/Sobolev like loss functions, and more! Furthermore, the DL_ROMs packages naturally interacts with other powerful libraries such as numpy, Pytorch and FEniCS.

The whole library is documented using native Python syntax, and it can be inspected via the *help* command. E.g.,

     from dlroms.roms import POD
     help(POD)


## Installation
### Basic version
The basic version of the DLROMs package allows users to create and train sophisticated neural network models, while also granting access to classical ROM techniques such as Principal Orthogonal Decomposition (POD). This version can be easily installed on Linux, Windows and MacOS. To do so, simply run

    pip install git+https://github.com/NicolaRFranco/dlroms.git

This will automatically install all the packages required for the basic version (numpy, torch, etc.). </br>
Note: if you are using **conda**, make sure that **pip** is available. If not, you can easily install it via

    conda install pip

### Advanced version
The advanced version integrates the basic one with additional tools coming from the FEniCS library, allowing users to: compute integral and Sobolev norms, produce norm-aware POD projections, visualize mesh-based data and more. Installation is recommended on Linux and MacOS. To get it: (i) install the basic version first, (ii) integrate the installation manually by installing [FEniCS](https://fenicsproject.org/) and any mesh generator of your choice (compatible choices incluide [mshr](https://anaconda.org/conda-forge/mshr) and [gmsh](https://anaconda.org/conda-forge/gmsh)). Note: as before, make sure that pip is available if you are using conda.

### Colab installation
The advanced version of the dlroms package is also available on Google Colab. To use it, include the following instructions at the beginning of your notebook:

    try:
         from dlroms import*
    except:
         !pip install git+https://github.com/NicolaRFranco/dlroms.git
         from dlroms import*
    
The pip instruction will install the basic version, while the importation of the dlrom package will automatically trigger the complementary installation of FEniCS and gmsh. At the same time, this syntax will avoid redundant installations if the kernel is restarted.



## Modules overview
The DLROMs library consists of several modules, which we may synthesize as follows.

### Main modules
* **dlroms.roms**</br> *Construction of data-driven ROMs*. Includes basic algorithms, such as POD, and abstract classes for incorporating and training neural network models. It can be used to implement ROM strategies such as [POD-NN](https://doi.org/10.1016/j.jcp.2018.02.037), autoencoder based [DL-ROMs](https://doi.org/10.1090/mcom/3781), and more.
  
* **dlroms.dnns**</br> *Design of neural network architectures*. Pytorch based module for constructing, saving and loading basic DNN architectures.
  
* **dlroms.fespaces**</br> *Handling meshes and discretized functional data*. FEniCS based module for data visualization, conversion (dof-to-function, torch-to-fenics representation) and more.
  
* **dlroms.minns**</br> *Hybrid module bridging neural networks and finite element spaces*. Implements advanced neural network architectures for mesh-based functional data (e.g., data coming from Finite Element simulations). These include: (i) trainable architectures, such as [Mesh-Informed Neural Networks (MINNs)](https://doi.org/10.1007/s10915-023-02331-1), (ii) nonlearnable blocks, for computing, e.g., integral norms, geodesic distances and more.

### Auxiliary modules
* **dlroms.cores**</br> Pytorch based library for handling, generating, and loading CPU/GPU tensors.
* **dlroms.geometry**</br> Auxiliary module regulating the interaction between dlroms.fespaces and gmsh (if installed).
* **dlroms.gp**</br> Complementary module implementing Gaussian processes in Finite Element spaces.
* **dlroms.gifs**</br> Auxiliary library for the visualization of time-dependent solutions.
* **dlroms.colab**</br> Integrative module for compatibility with Google Colab.
  
We remark that the main algorithms, classes and routines are also included in the **\_\_init\_\_.py** module, and thus readily available after

    from dlroms import*
