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

from IPython.display import clear_output
import os

def setcolab():
  try:
    import dolfin
  except ImportError:
    print("Installing fenics... this should take about 30-60 seconds.")
    os.system('wget "https://fem-on-colab.github.io/releases/fenics-install.sh" -O "/tmp/fenics-install.sh" && bash "/tmp/fenics-install.sh"')

  try:
    import gmsh
  except ImportError:
    print("Installing gmsh... this should take about 30-60 seconds.")
    os.system('wget "https://fem-on-colab.github.io/releases/gmsh-install.sh" -O "/tmp/gmsh-install.sh" && bash "/tmp/gmsh-install.sh"')

  clear_output(wait = True)
  print("Both fenics and gmsh are installed. Please restart Runtime in order to operate.")
