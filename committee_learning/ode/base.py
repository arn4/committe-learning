import numpy as np
import math
from tqdm import tqdm
from abc import ABC, abstractmethod

from .._config.python import scalar_type

"""
These classes should be the base for all ODE integrations.
"""

class BaseODE(ABC):
  def __init__(self, P0, Q0, M0, dt, noise_term, disable_QM_save = False):
    assert(Q0.shape[0] == Q0.shape[1])
    assert(Q0.shape[1] == M0.shape[0])
    assert(M0.shape[1] == P0.shape[0])
    assert(P0.shape[0] == P0.shape[1])
    
    self.dt = scalar_type(dt)
    self.p = Q0.shape[0]
    self.k = P0.shape[1]

    self.P = np.array(P0, ndmin=2, dtype=scalar_type)
    self.M = np.array(M0, ndmin=2, dtype=scalar_type)
    # remember to initialize Q or Qorth

    self._simulated_time = scalar_type(0.)

    self.saved_times = []
    self.saved_risks = []

    self.noise_term = noise_term

    self.disable_QM_save = disable_QM_save

    # Include here the other variables you want to store.

  def fit(self, time, n_saved_points=20, show_progress=True):
    discrete_steps = int(time/self.dt)
    n_saved_points = min(n_saved_points, discrete_steps)
    save_frequency = max(1, int(discrete_steps/n_saved_points))

    for step in tqdm(range(discrete_steps), disable=not show_progress):
      # Add data if necessary
      if step%save_frequency == 0:
        self._save_step(step)
      self._update_step()

    self._simulated_time += time

  def fit_logscale(self, decades, save_per_decade = 100, show_progress=True):
    assert(10**decades>self.dt)
    d_min = int(math.log(self.dt,10))
    for d in range(d_min,decades+1):
      self.fit(10**d-self._simulated_time, save_per_decade, show_progress=show_progress)

  @abstractmethod
  def _save_step(self, step):
    self.saved_times.append(self._simulated_time + self.dt * (step+1))
    self.saved_risks.append(self.risk())

  @abstractmethod
  def risk(self):
    return
  
  @abstractmethod
  def _update_step():
    return


class BaseFullODE(BaseODE,ABC):
  def __init__(self, P0, Q0, M0, dt, noise_term = True, gamma_over_p = None, noise = None, quadratic_terms = False, disable_QM_save=False):
    super().__init__(P0, Q0, M0, dt, noise_term, disable_QM_save)

    self.Q = np.array(Q0, ndmin=2, dtype=scalar_type)
    if noise_term:
      assert(gamma_over_p is not None)
      assert(noise is not None)
      self.noise = scalar_type(noise)
      self._gamma_over_p = scalar_type(gamma_over_p)

    self.quadratic_terms = quadratic_terms
    if quadratic_terms:
      assert(gamma_over_p is not None)
      self._gamma_over_p = scalar_type(gamma_over_p)
    

    self.saved_Ms = []
    self.saved_Qs = []

  def _save_step(self, step):
    super()._save_step(step)
    if not self.disable_QM_save:
      self.saved_Ms.append(np.copy(self.M))
      self.saved_Qs.append(np.copy(self.Q))

  def _update_step(self):
    Phi, Psi = self._compute_Phi_Psi()
    self.Q += Phi * self.dt
    self.M += Psi * self.dt

  @abstractmethod
  def _compute_Phi_Psi(self):
    return


class BaseSphericalFullODE(BaseFullODE,ABC):
  def _compute_Phi_Psi(self):
    # Unconstrainted updtes
    Phi, Psi = super()._compute_Phi_Psi()

    diagQ = np.diag(Phi)
    row_diagQ = np.tile(diagQ, (int(self.p),1))


    Phi_constraint = self.Q*(row_diagQ+row_diagQ.T)/scalar_type(2)
    Psi_constraint = self.M*np.tile(diagQ, (int(self.k),1)).T/scalar_type(2)

    Phi -= Phi_constraint
    Psi -= Psi_constraint
    return Phi, Psi

class BaseLargePODE(BaseODE,ABC):
  """
  This equations are supposed to be used in the regime Q = MM^T + diag(Q^orth).
  We just need to track the diagonal of Q^orth and M, so no need to evolve a p x p matrix,
  that would be unfesible.
  """
  def __init__(self, P0, Q0, M0, dt, offdiagonal = True, d = None, noise_term = True, noise_gamma_over_p = None, disable_QM_save=False):
    super().__init__(P0, Q0, M0, dt, noise_term, disable_QM_save)

    if offdiagonal:
      assert(d is not None)

    self.d = d
    self.offdiagonal = offdiagonal
    self.Qorth = np.array(np.diag(np.array(Q0 - M0@M0.T, ndmin=2, dtype=scalar_type)))
    if noise_term:
      assert(noise_gamma_over_p is not None)
      self._noise_gamma_over_p = scalar_type(noise_gamma_over_p)

    self.saved_Ms = []
    self.saved_Qorths = []

  @property
  def Q(self):
    return self.M @ self.M.T + np.diag(self.Qorth)

  def _save_step(self, step):
    super()._save_step(step)
    if not self.disable_QM_save:
      self.saved_Ms.append(np.copy(self.M))
      self.saved_Qorths.append(np.copy(self.Qorth))

  def _update_step(self):
    Psi, Gamma = self._compute_Psi_Gamma()
    self.Qorth += Gamma * self.dt
    self.M += Psi * self.dt

  @abstractmethod
  def _compute_Psi_Gamma(self):
    return

  