from ...ode import SquaredActivationODE, SphericalSquaredActivationODE

import numpy as np
from scipy.linalg import sqrtm

from ..base import BaseSDE
from ..._cython.numpy_extra import symmetrize
from .variances import spherical_p1variance, spherical_p2variance

"""
Order of variables:
q11, q12,..., q1p, q22, q23, q24,..., q2p,...,qpp, m11, m21,...,mp1
"""

class SphericalGeneralizedPhaseRetrievalSDE(BaseSDE, SphericalSquaredActivationODE):
    def __init__(self, Q0, M0, d, dt, noise_term = True, gamma_over_p = None, noise = None, quadratic_terms = False, seed = None, disable_QM_save=False):
        # assert(sum((np.diag(Q0)-np.ones((Q0.shape[0])))**2)==0.)
        P0 = np.array([[1.]])
        super().__init__(d, P0, Q0, M0, dt, noise_term = noise_term, gamma_over_p = gamma_over_p, noise = noise, quadratic_terms = quadratic_terms, seed=seed, disable_QM_save=disable_QM_save)

        if Q0.shape[0] == 1:
            def _variance_method(M, Q):
                return spherical_p1variance(q = Q[0][0], m = M[0][0], gamma=self._gamma_over_p, noise=self.noise)
            self._variance = _variance_method
        elif Q0.shape[0] == 2:
            def _variance_method(M, Q):
                return spherical_p2variance(m1 = M[0][0], m2 = M[1][0], q11 = Q[0][0], q12 = Q[0][1], q22 = Q[1][1], gamma=2.*self._gamma_over_p, noise=self.noise)
            self._variance = _variance_method
        else:
            raise NotImplementedError(f"Can't run with p = {Q0.shape[0]}")

    def _update_step(self):
        p = self.Q.shape[0]

        variance = self._variance(self.M, self.Q)
        try:
            # sigma = sqrtm(variance).real * self._gamma_over_p/self.d
            sigma = sqrtm(variance).real * np.sqrt(self._gamma_over_p/self.d)
            # print('some')
        except ValueError as error:
            raise error(f"Unable to compute the matrix sqrt of {variance}")
        brownian = self.rng.normal(size=variance.shape[0]) 

        unconstrainted_stochastic_term = np.einsum('ij,j->i', sigma, brownian)
        unconstraintQ_stochastic_term = symmetrize(unconstrainted_stochastic_term[:p*(p+1)//2])
        constraint = np.diag(unconstraintQ_stochastic_term)
        constraint_by_row = np.tile(constraint, (p,1))

        constraint_for_M = constraint.reshape(self.M.shape)
        unconstraintM_stochastic_term = unconstrainted_stochastic_term[p*(p+1)//2:].reshape(self.M.shape)

        super()._update_step() # Deterministic Update
        self.Q += (unconstraintQ_stochastic_term - self.Q*(constraint_by_row + constraint_by_row.T)/2.) * np.sqrt(self.dt)
        self.M += (unconstraintM_stochastic_term - self.M*constraint_for_M/2.) * np.sqrt(self.dt)

        

