

from .ode import FullODEResult

## PRSDEResult History (matching the FullOdeResult numbering)
# - 0.3.0: first usable version
# - 0.4.0: update at FullOdeResult

class PhaseRetrievalSDEResult(FullODEResult):
  def __init__(self, initial_condition = None, id = 0, **kattributes):
    super().__init__(initial_condition=initial_condition, id=id, **kattributes)

    # This is an identifier for which file version I'm using to store files
    self.version = '0.3.0'

  # I need this because is called inside from_file_or_run
  def from_ode(self, sde):
    super().from_ode(sde)
    self.d = sde.d
    self.seed = sde.seed

  def from_sde(self, *args, **kwargs):
    self.from_ode(*args, **kwargs)

  @property
  def datastring(self):
    return [
      f'{self.d}',
      f'{self.seed}',
    ] + super().datastring
  