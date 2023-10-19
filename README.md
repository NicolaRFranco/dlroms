# DLROMs: a Python package for Deep Learning based Reduced Order Models


**DLROMs** is a deep learning library for model order reduction that leverages on packages such as Pytorch and FEniCS. It comes with a basic version (no FEniCS required) and an advanced one. The techniques made available within this library are reminiscent of the following scientific articles: 

[1] Franco et al., [A deep learning approach to reduced order modelling of parameter dependent partial differential equations](https://doi.org/10.1090/mcom/3781), *Mathematics of Computation*, 92 (340), 483-524 (2023).
     
[2] Franco et al., [Approximation bounds for convolutional neural networks in operator learning](https://doi.org/10.1016/j.neunet.2023.01.029), *Neural Networks*, 161: 129-141 (2023).
     
[3] Franco et al., [Mesh-Informed Neural Networks for Operator Learning in Finite Element Spaces](https://doi.org/10.1007/s10915-023-02331-1), *Journal of Scientific Computing*, 97(2), 35 (2023).

This library was written, and is currently maintained, by **Nicola Rares Franco**, **Ph.D.**, MOX, Politecnico di Milano. For a tensorflow alternative, we recommend the repositories by [Stefania Fresca, Ph.D](https://github.com/stefaniafresca).


## DL-ROMs in a nutshell

Deep learning based reduced order models are efficient model surrogates that can emulate the accuracy of classical numerical solvers (hereby referred to as FOM: Full Order Models) by learning from high-quality simulations. The idea goes as follows. Let $\boldsymbol{\mu}\to u$ represent the action of a FOM solver, which, given a parameter instance $\boldsymbol{\mu}\in\mathbb{R}^{p}$ returns the corresponding PDE solution $u\in\mathbb{R}^{N_{h}}$, here represented by means of a suitable dof vector. Then, the construction of a DL-ROM can be sketched as:

1. Collect high-fidelity data, $`M=[\boldsymbol{\mu}_{1};\dots;\boldsymbol{\mu}_{N}]\in\mathbb{R}^{N\times p}`$ and $`U=[u_{1};\dots;u_{N}]\in\mathbb{R}^{N\times N_{h}}`$, using the FOM solver.
2. Initialize a DL-ROM module with trainable architectures $\psi_{1},\dots,\psi_{k}$.
3. Train the DL-ROM components to learn the FOM data.
4. Freeze the DL-ROM and use it whenever you like to: the computational cost is now negligible!

The code below shows a simple example in which a naive DNN model is used as a DL-ROM. For more advanced approaches, such as POD-NN or autoencoder-enhanced DL-ROMs, we refer to the *dlroms.roms* module.

    # Problem data
    p, Nh = 2, 501

    # Toy example of a FOM solver: given mu = [m0, m1], it returns u(x) = (m1-m0)x+m0, 
    # i.e. the solution to: -u''=0 in (0,1), u(0)=m0, u(1)=m1. 
    # The solution u=u(x) is discretized over a uniform grid with 501 nodes.
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

Starting from this simple pipeline, the DL-ROM package then allows plenty of generalizations: complex customizable models based on multiple architecture that cooperate with each other, mesh-informed architectures, integral norms for Lebesgue/Sobolev like loss functions, and more! Furthermore, it naturally interacts with other powerful libraries such as numpy, torch and fenics.

## Installation
### Basic version
The basic version of the DLROMs package allows users to create and train sophisticated neural network models, while also granting access to classical ROM techniques such as Principal Orthogonal Decomposition (POD). This version can be easily installed on Linux, Windows and MacOS. To install it, simply run

    pip install git+https://github.com/NicolaRFranco/dlroms.git

This will automatically install all the packages required for the basic version (numpy, torch, etc.). </br>
Note: if you are using **conda**, make sure that **pip** is available. If not, you can easily install it via

    conda install pip

### Advanced version
The advanced version integrates the basic one with additional tools coming from the FEniCS library, allowing users to: compute integral and Sobolev norms, produce norm-aware POD projections, visualize mesh-based data and more. Installation is recommended on Linux and MacOS. To install (and enable) the advanced version run

    pip install git+https://github.com/NicolaRFranco/dlroms.git

and integrate the installation manually by installing [FEniCS](https://fenicsproject.org/) and any mesh generator of your choice (compatible choices incluide [mshr](https://anaconda.org/conda-forge/mshr) and [gmsh](https://anaconda.org/conda-forge/gmsh)).</br>
Note: as before, make sure that pip is available if you are using conda.

### Colab installation
You can also install the advanced version on Colab by running the following instructions

    try:
      from dlroms import*
    except ImportError:
      !pip install git+https://github.com/NicolaRFranco/dlroms.git
      from dlroms.colab import setup
      setup()

Once all the dependencies have been installed, **make sure you restart your notebook.** If the runtime is not restarted, you will only have access to the basic version of the DLROMs package.



## Modules overview
The DLROMs library consists of several modules, which we may synthesize as follows.

* **dlroms.colab**</br> Integrative module for the installation in Colab.
  
* **dlroms.cores**</br> For handling, generating, and loading CPU/GPU torch tensors.
  
* **dlroms.dnns**</br> For constructing, saving and loading basic neural network architectures.
  
* **dlroms.minns**</br> An integrative module that allows users to create neural network architectures specifically tailored for mesh-based functional data, such as signals discretized via Finite Elements. It includes trainable models, such as [Mesh-Informed Neural Networks (MINNs)](https://doi.org/10.1007/s10915-023-02331-1), and nonlearnable architectures for computing integral norms, geodesic distances and more.
  
* **dlroms.fespaces**</br> FEniCS based library for handling meshes and discretized functional data (visualization, dof-to-function conversion etc.).
  
* **dlroms.geometry**</br> Auxiliary module that allows fespaces.py to interact with gmsh (if installed).
  
* **dlroms.gp**</br> Complementary module for constructing Gaussian processes in Finite Element spaces.
  
* **dlroms.gifs**</br> Auxiliary library for the visualization of time-dependent solutions.
  
* **dlroms.roms**</br> Core library that allows users to construct data-driven Reduced Order Models. Includes basic algorithms, such as POD, and abstract classes for incorporating and training neural network models. It can be used to implement ROM strategies such as [POD-NN](https://doi.org/10.1016/j.jcp.2018.02.037), autoencoder based [DL-ROMs](https://doi.org/10.1090/mcom/3781), and more.

We remark that the main algorithms, classes and routines are also included in the **\_\_init\_\_.py** module. Thus, instead of navigating the whole library, users can easily import the main features of the DLROMs package by simply running

    from dlroms import*
