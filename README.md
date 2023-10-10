# DLROMs: a Python package for Deep Learning based Reduced Order Models

**DLROMs** is a deep learning library for model order reduction that leverages on packages such as Pytorch and FEniCS. It comes with a basic version (no FEniCS required) and an advanced one. The techniques made available within this library are reminiscent of the following scientific articles: 

[1] Franco et al. (2023). A deep learning approach to reduced order modelling of parameter dependent partial differential equations, *Mathematics of Computation*, 92 (340), 483-524.
    DOI: https://doi.org/10.1090/mcom/3781
     
[2] Franco et al. (2023). Approximation bounds for convolutional neural networks in operator learning, *Neural Networks*, 161: 129-141.
    DOI: https://doi.org/10.1016/j.neunet.2023.01.029
     
[3] Franco et al. (2023). Mesh-Informed Neural Networks for Operator Learning in Finite Element Spaces, *Journal of Scientific Computing*, 97(2), 35.
    DOI: https://doi.org/10.1007/s10915-023-02331-1

This library was written, and is currently maintained, by **Nicola Rares Franco**, **Ph.D.**, MOX, Politecnico di Milano. For a tensorflow alternative, we recommend the repositories by Stefania Fresca, Ph.D. (link: https://github.com/stefaniafresca).


##### Table of Contents  
[1. DL-ROMs in a nutshell](#dlroms)  
[2. Installation](#installation)  
[3. Modules overview](#overview) 


<a name="dlroms"/>

## 1. DL-ROMs in a nutshell

    Work in progress.

Work in progress.



<a name="installation"/>

## 2. Installation
### Basic version
The basic version of the DLROMs package allows users to create and train sophisticated neural network models, while also granting access to classical ROM techniques such as Principal Orthogonal Decomposition (POD). This version can be easily installed on Linux, Windows and MacOS. To install it, simply run

    pip install git+https://github.com/NicolaRFranco/dlroms.git

This will automatically install all the packages required for the basic version (numpy, torch, etc.). </br>
Note: if you are using **conda**, make sure that **pip** is available. If not, you can easily install it via

    conda install pip

### Advanced version
The advanced version integrates the basic on with additional tools coming from the FEniCS library, and it is particularly suited for handling mesh-based functional data, allowing users to: compute integral and Sobolev norms, produce norm-aware POD projections, visualize mesh-based data and more. Installation is recommended on Linux and MacOS. To install (and enable) the advanced version run

    pip install git+https://github.com/NicolaRFranco/dlroms.git

and integrate the installation manually by installing [FEniCS](https://fenicsproject.org/) and any mesh generator of your choice (compatible choices incluide [mshr](https://anaconda.org/conda-forge/mshr) and [gmsh](https://anaconda.org/conda-forge/gmsh)).</br>
Note: as before, make sure that pip is available if you are using conda.

### Colab installation
You can also install the advanced version on Colab by running the following instructions in your Colab notebook

    try:
      from dlroms import*
    except ImportError:
      !pip install git+https://github.com/NicolaRFranco/dlroms.git
      from dlroms.colab import setup
      setup()

Once all the dependencies have been installed, **make sure to restart the notebook Runtime.** If the kernel is not restarted, you will only have access to the basic version.

<a name="overview"/>

## 3. Modules overview
Work in progress.
