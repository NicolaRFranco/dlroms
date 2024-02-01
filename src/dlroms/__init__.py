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

try:
  import google.colab
  from dlroms.colab import setup
  setup()
except:
  None

from dlroms.minns import L2, H1, Local, Geodesic, iVersion
from dlroms.dnns import Dense, train, Clock, num2p
from dlroms.roms import POD, project, snapshots, PODerrors, mre, mse, ROM, euclidean, boxplot, regcoeff, PODNN, DLROM, DFNN, mrei, msei, projectdown, projectup, neuclidean
from dlroms.cores import CPU, GPU
import dlroms.fespaces as fe
import matplotlib.pyplot as plt

def plot(*args, **kwargs):
  from torch import Tensor
  newargs = [(a if not isinstance(a, Tensor) else a.detach().cpu().numpy())  for a in args]
  plt.plot(*newargs, **kwargs)


iDense = iVersion(Dense)
iLocal = iVersion(Local)
iGeodesic = iVersion(Geodesic)
