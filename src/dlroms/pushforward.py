from dlroms.dnns import Dense, leakyReLU
from dlroms.roms import DFNN
import numpy as np
import torch

class RandomEnrichmentLayer(Dense):
  def __init__(self, m, n, r, activation = leakyReLU):
    super(RandomEnrichmentLayer, self).__init__(m+r, n, activation = activation)
    self.r = r
    self.share = False

  def forward(self, x):
    if(self.share):
      z = dv.zeros(*x.shape[:-1], 1) + dv.randn(1, self.r)
    else:
      z = dv.randn(*x.shape[:-1], self.r)
    return super(RandomEnrichmentLayer, self).forward(torch.cat([x, z], axis = -1))

  def set_sharing(self, share):
    self.share = share

class PushForward(DFNN):
  def forward(self, x):
    return super(PushForward, self).forward(x.unsqueeze(1) + self.coretype().zeros(1, self.repeats, 1))

  def initFrom(self, other):
    for i in range(len(other)):
      for j in range(len(other[i])):
        deterministic = other[i][j]
        random = self[i][j]
      
        w = deterministic.w().detach().cpu().numpy() + 0.0
        b = deterministic.b().detach().cpu().numpy() + 0.0
      
        if(isinstance(random, RandomEnrichmentLayer)):
          w = np.concatenate((w, np.zeros((w.shape[0], random.r))), axis = 1)
          random.load(w, b)
        else:
          random.load(w, b)
