from ...ode import SquaredActivationODE, SphericalSquaredActivationODE

import numpy as np
from scipy.linalg import sqrtm

from ..base import BaseSDE
from .variance import _variance_q, _variance_m, _covariance_qm

class PhaseRetrievalSDE(BaseSDE, SquaredActivationODE):

  def __init__(self, p0, q0, m0, d, dt, noise_term = True, gamma_over_p = None, noise = None, quadratic_terms = False, seed = None, disable_QM_save=False):
    super().__init__(
      P0 = np.array([[p0]]) if isinstance(p0, float) else p0,
      Q0 = np.array([[q0]]) if isinstance(q0, float) else q0,
      M0 = np.array([[m0]]) if isinstance(m0, float) else m0,
      d = d,
      dt = dt,
      noise_term = noise_term,
      gamma_over_p = gamma_over_p,
      noise = noise,
      quadratic_terms = quadratic_terms,
      seed = seed,
      disable_QM_save=disable_QM_save
    )

  def _variances(self):
    q = self.Q[0][0]
    m = self.M[0][0]
    rho = self.P[0][0]
    # In phase retrieval gamma_over_p = gamma
    return (
      max(_variance_q(q,m,rho,self._gamma_over_p,self.noise), 0.),
      max(_variance_m(q,m,rho,self._gamma_over_p,self.noise), 0.)
    )
  
  def _update_step(self):
    super()._update_step()
    varQ, varM = self._variances()
    # I think this implementation is wrong, why don't I take the sqrt of the Covariance matrix?
    self.Q += self.rng.normal() * np.sqrt(varQ * self.dt / self.d)
    self.M += self.rng.normal() * np.sqrt(varM * self.dt / self.d)


class SphericalPhaseRetrievalSDE(PhaseRetrievalSDE, SphericalSquaredActivationODE):
  def __init__(self, m0, d, dt, noise_term = True, gamma_over_p = None, noise = None, quadratic_terms = False, extra_drift = False, seed = None, disable_QM_save=False):
    super().__init__(
      p0 = 1., q0 = 1., m0 = m0, d = d, dt = dt,
      noise_term = noise_term, gamma_over_p = gamma_over_p, noise = noise,
      quadratic_terms = quadratic_terms, seed = seed, disable_QM_save=disable_QM_save
    )
    self.extra_drift = extra_drift

  def _variances(self):
    varQ, varM = super()._variances()
    q = self.Q[0][0]
    m = self.M[0][0]
    rho = self.P[0][0]
    covQM = _covariance_qm(q, m ,rho, self._gamma_over_p, self.noise)
    return (
      varQ,
      varM,
      # Enforce the covariance matrix to be positive semi-definite
      covQM if covQM**2 > varQ*varM else np.sign(covQM) * np.sqrt(varQ*varM)
    )


  def _update_step(self):
    m = self.M[0][0]
    # Variance matrix
    varQ, varM, covQM = self._variances()

    Sigma = np.array([[varQ, covQM],
                      [covQM, varM]]) / (self.d) #* self._gamma_over_p
    # std matrix
    sigma_q, sigma_m = sqrtm(Sigma).real

    stochastich_term = np.einsum(
      'i,ijk->jk',
      sigma_m - m/2 * sigma_q,
      self.rng.normal(size=(2,1,1))
    )

    if self.extra_drift:
      extra_drift_term = (3/8 * m * np.dot(sigma_q, sigma_q) - .5 * np.dot(sigma_q, sigma_m))
    else:
      extra_drift_term = 0.

    Phi, Psi = SphericalSquaredActivationODE._compute_Phi_Psi(self)
    self.M += (Psi + extra_drift_term) * self.dt + stochastich_term * np.sqrt(self.dt)


