from ...ode import SquaredActivationODE, SphericalSquaredActivationODE

import numpy as np
from scipy.linalg import sqrtm

from .variance import _variance_q, _variance_m, _covariance_qm

class SquaredPhaseRetrivialSDE(SquaredActivationODE):

  def __init__(self, p0, q0, m0, d, dt, noise_term = True, gamma_over_p = None, noise = None, quadratic_terms = False, seed = None):
    super().__init__(
      P0 = np.array([[p0]]) if isinstance(p0, float) else p0,
      Q0 = np.array([[q0]]) if isinstance(q0, float) else q0,
      M0 = np.array([[m0]]) if isinstance(m0, float) else m0,
      dt = dt,
      noise_term = noise_term,
      gamma_over_p = gamma_over_p,
      noise = noise,
      quadratic_terms = quadratic_terms
    )
    self.d = float(d)
    
    # if seed is not None:
    #   seed ^= 22031998 # this line is shuffling the seed just to ensure that I'm not using it on every generator
    self.rng = np.random.default_rng(seed)

  def _variances(self):
    q = self.Q[0][0]
    m = self.M[0][0]
    rho = self.P[0][0]
    # In phase retrivial gamma_over_p = gamma
    return (
      max(_variance_q(q,m,rho,self._gamma_over_p,self.noise), 0.),
      max(_variance_m(q,m,rho,self._gamma_over_p,self.noise), 0.)
    )
  
  def _update_step(self):
    super()._update_step()
    varQ, varM = self._variances()
    self.Q += self.rng.normal() * np.sqrt(varQ * self.dt / self.d)
    self.M += self.rng.normal() * np.sqrt(varM * self.dt / self.d)


class SphericalSquaredPhaseRetrivialSDE(SquaredPhaseRetrivialSDE, SphericalSquaredActivationODE):
  def __init__(self, m0, d, dt, noise_term = True, gamma_over_p = None, noise = None, quadratic_terms = False, seed = None):
    super().__init__(
      p0 = 1., q0 = 1., m0 = m0, d = d, dt = dt,
      noise_term = noise_term, gamma_over_p = gamma_over_p, noise = noise,
      quadratic_terms = quadratic_terms, seed = seed
    )

  def _step_update(self):
    m = self.M[0][0]
    # Variance matrix
    varQ, varM = self._variances()
    Sigma = np.array([[varQ, 0.],
                      [0., varM]]) / self.d
    # std matrix
    sigma_q, sigma_m = sqrtm(Sigma)

    stochastich_term = np.einsum(
      'i,ijk->jk',
      sigma_m - m/2 * sigma_q,
      self.rng.normal(size=(2,1,1))
    )

    extra_drift = (3/8 * m * np.dot(sigma_q, sigma_q) - .5 * np.dot(sigma_q, sigma_m))

    Phi, Psi = super()._compute_Phi_Psi()
    self.M += (Psi - m/2 * Phi + extra_drift) * self.dt + stochastich_term * np.sqrt(self.dt)


