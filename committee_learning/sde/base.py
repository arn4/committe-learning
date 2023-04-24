import numpy as np

class BaseSDE():
  def __init__(self, *args, seed = None, **kwargs):
    super().__init__(*args, **kwargs)
    self.seed = seed
    self.rng = np.random.default_rng(seed)