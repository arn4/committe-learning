import numpy as np
import datetime
import hashlib

from .base import BaseResult
from ..simulation.base import BaseSimulation
from ..simulation.second_layer import TwoLayerSimulation
from ..utilities import upper_bound


## SimulationResult History
# - 0.2: first usable version
# - 0.3: found a little error in ODE SymmetricIC
# - 0.4: added the macroscopic variables
# - 0.5: added a separator in the datastring
# - 0.6: extra_metrics support (btw, should be retrocompatible)
# - 0.7: added seed and made it different from id; added target_save_per_decade

class SimulationResult(BaseResult):
  def __init__(self,initial_condition = None, id = 0, **kattributes):
    super().__init__(initial_condition=initial_condition, id=id, **kattributes)

    # This is an identifier for which file version I'm using to store files
    self.version = '0.7'
  
  def from_simulation(self, simulation: BaseSimulation):
    self.timestamp = str(datetime.datetime.now())
    self.d = simulation.d
    self.p = simulation.p
    self.k = simulation.k
    self.noise = simulation.noise
    self.gamma = simulation.gamma
    self.seed = simulation.seed
    self.completed_steps = simulation._completed_steps
    self.save_per_decade = len(simulation.saved_steps)/int(np.log10(simulation._completed_steps)) if len(simulation.saved_steps)>0 else None
    self.activation = simulation.activation
    self.steps = simulation.saved_steps
    self.risks = simulation.saved_risks
    self.Qs = np.array(simulation.saved_Qs).tolist()
    self.Ms = np.array(simulation.saved_Ms).tolist()
    self.P = np.array(simulation.P).tolist()

    for extra_metric_name in simulation.extra_metrics.keys():
      setattr(self, extra_metric_name, getattr(simulation, extra_metric_name))
    
    self.extra_metrics_names = '__'.join(map(str, simulation.extra_metrics.keys()))
  
  @property
  def datastring(self):
    return [
      f'{self.d}',
      f'{self.p}',
      f'{self.k}',
      f'{self.activation}',
      f'{self.noise:.6f}',
      f'{self.gamma:.6f}',
      f'{self.completed_steps}',
      f'{self.save_per_decade_target}',
      f'{self.seed}',
      f'{self.id}',
      self.extra_metrics_names
    ]

  def from_file_or_run(self, simulation, decades, save_per_decade = 100, path='', show_progress=True, force_run=False, force_read=False):
    if force_run and force_read:
      raise ValueError('Flags force_read and force_run can be both true!')
    self.from_simulation(simulation)
    self.completed_steps = int(10**decades)
    self.save_per_decade_target = save_per_decade 
    try:
      if force_run:
        raise FileNotFoundError
      self.from_file(path=path)
    except FileNotFoundError as file_error:
      if force_read:
        print(f'Not found {self.datastring}')
        raise file_error
      simulation.fit_logscale(decades, save_per_decade = save_per_decade, show_progress=show_progress)
      self.from_simulation(simulation)
      self.to_file(path=path)
  
  @property
  def times(self):
    return np.array(self.steps)*self.gamma/(self.p*self.d)

  def _step_to_index(self,step):
    return upper_bound(step, self.steps)

  def _time_to_index(self,t):
    return self._step_to_index(int(t*self.d*self.p/self.gamma))

  def M_at_time(self, t):
    return np.array(self.Ms[self._time_to_index(t)])
  
  def Q_at_time(self, t):
    return np.array(self.Qs[self._time_to_index(t)])
  
class TwoLayerSimulationResult(SimulationResult):
  def from_simulation(self, simulation: TwoLayerSimulation):
    super().from_simulation(simulation)
    self.ass = np.array(simulation.saved_as).tolist()
    self.at = np.array(simulation.at).tolist()
    self.normalized_first = simulation.spherical_weights
    self.normalized_second = simulation.normalized_second_layer
  
  @property
  def datastring(self):
    return super().datastring + [
      str(self.normalized_first),
      str(self.normalized_second),
      'two_layer'
    ]
  
  def a_at_time(self, t):
    return np.array(self.ass[self._time_to_index(t)])