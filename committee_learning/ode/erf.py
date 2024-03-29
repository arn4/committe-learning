
from .base import BaseODE, BaseFullODE, BaseSphericalFullODE, BaseLargePODE
from .cython_erf import erf_updates, largep_erf_updates_onlydiag, largep_erf_updates_offdiag
from .._cython.risk import erf_risk, erf_risk_Zexpectation

class BaseErfActivationODE(BaseODE):
  def risk(self):
    return erf_risk(self.Q, self.M, self.P)

class ErfActivationFullODE(BaseFullODE, BaseErfActivationODE):
  def _compute_Phi_Psi(self):
    return erf_updates(self.Q,self.M,self.P, self.noise_term, self._gamma_over_p, self.noise, self.quadratic_terms)

class SphericalErfActivationFullODE(BaseSphericalFullODE, ErfActivationFullODE):
  pass
 
class LargePErfActivationODE(BaseLargePODE,BaseErfActivationODE):
  """
  P0 must be the identity matrix!
  """
  def __init__(self, P0, Q0, M0, dt, offdiagonal = True, d = None, noise_term = True, noise_gamma_over_p = 0.):
    super().__init__(P0, Q0, M0, dt, offdiagonal, d, noise_term, noise_gamma_over_p)
    
    if offdiagonal:
      self.risk = lambda: erf_risk_Zexpectation(self.Qorth, self.M, self.P, self.d-self.k)

      if self.d - self.k <= 2:
        raise NotImplementedError('d-k must be > 2.\nFor d-k<=0, there is no need to track Qorth; d-k = 1,2 are special cases not implemented for simplicity.')

  def _compute_Psi_Gamma(self):
    if not self.offdiagonal:
      return largep_erf_updates_onlydiag(self.Qorth, self.M, self.P, self.noise_term, self._noise_gamma_over_p)
    else:
      return largep_erf_updates_offdiag(self.Qorth, self.M, self.P, self.noise_term, self._noise_gamma_over_p, self.d-self.k)
