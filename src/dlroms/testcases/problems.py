import numpy as np

def UniformSampler(p, seed):
  np.random.seed(seed)
  return np.random.rand(p)

class Problem(object):
    def __init__(self, FOM, parameters, sampler = UniformSampler):
        self.FOM = FOM
        self.parameters = parameters
        self.sampler = sampler

    def generate(self, N, filename, verbose = True):
        p = len(self.parameters['Parameter'])
        pmax = np.array(self.parameters['Max'])
        pmin = np.array(self.parameters['Min'])
        
        def FOMsampler(seed):
            mu = self.sampler(p, seed = seed)*(pmax-pmin) + pmin
            u = self.FOM(mu)
            return mu, u

        from dlroms.roms import snapshots
        snapshots(N, sampler = FOMsampler, filename = filename, verbose = verbose)

    def params_summary(self):
        import pandas as pd
        return pd.DataFrame(self.parameters, index = [""]*len(self.parameters['Parameter']))

    def FOM_summary(self):
        from dlroms.dnns import Clock
        p = len(self.parameters['Parameter'])
        pmax = np.array(self.parameters['Max'])
        pmin = np.array(self.parameters['Min'])
        clock = Clock()
        clock.start()
        u0 = self.FOM(self.sampler(p, seed = 0)*(pmax-pmin) + pmin)
        clock.stop()
        print("PDE parameters:\t%d." % p)
        print("FOM dimension:\t%d." % len(u0))
        print("FOM exec. time:\t%s." % clock.elapsedTime())

def FOMdata(label):
    import importlib
    module_name = f"dlroms.testcases.{label}"
    try:
        module = importlib.import_module(module_name)
        return getattr(module, "FOMsolver"), getattr(module, "parameters")
    except (ModuleNotFoundError, AttributeError) as e:
        raise ImportError(f"Could not import 'FOMsolver' and 'parameters' from {module_name}") from e

def generate_data(problem_label, ndata, summary = True, filename = None):
    problem = Problem(*FOMdata(problem_label))
    if(summary):
        name = filename if(not (filename is None)) else (problem_label + "_snapshots")
        problem.generate(ndata, filename = name + ".npz")
        print("\n")
        problem.FOM_summary()
        print("\n")
        print(problem.params_summary())
