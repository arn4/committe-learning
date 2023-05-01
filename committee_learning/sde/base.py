import numpy as np

class BaseSDE():
  def __init__(self, d, *args, seed = None, **kwargs):
    super().__init__(*args, **kwargs)
    self.d = d
    self.seed = seed
    self.rng = np.random.default_rng(seed)