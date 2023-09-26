from dlroms.minns import L2, H1, Local, Geodesic
from dlroms.dnns import Dense, train, Clock, num2p
from dlroms.roms import POD, project, snapshots, PODerrors, mre, mse, ROM, euclidean, boxplot, regcoeff, PODNN, DLROM, mrei, msei
from dlroms.cores import CPU, GPU
import dlroms.fespaces as fe
import matplotlib.pyplot as plt

def plot(*args, **kwargs):
  newargs = [(a if not isinstace(a, Tensor) else a.cpu().numpy())  for a in args]
  plt.plot(*newargs, **kwargs)
