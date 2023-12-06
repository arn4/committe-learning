
from .base import BaseODE, BaseFullODE, BaseSphericalFullODE, BaseLargePODE
from .cython_H3 import H3_updates
from .._cython.risk import H3_risk

class BaseH3ActivationODE(BaseODE):
  def risk(self):
    return H3_risk(self.Q, self.M, self.P)

class H3ActivationFullODE(BaseFullODE, BaseH3ActivationODE):
  def _compute_Phi_Psi(self):
    return H3_updates(self.Q,self.M,self.P, self.noise_term, self._gamma_over_p, self.noise, self.quadratic_terms)

class SphericalH3ActivationFullODE(BaseSphericalFullODE, H3ActivationFullODE):
  pass