# distutils: language = c++

import numpy as np
cimport numpy as np
from libc.math cimport sqrt
cimport cython

from .._config.python import DTYPE
from .._config.cython cimport DTYPE_t

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
cpdef symmetrize(np.ndarray[DTYPE_t, ndim=1] array):
  """
  Converting an array in its corresponding symmetic matrix.
  Example: 
  [1,2,3,4,5,6] [[1,2,3],
                 [2,4,5],
                 [3,5,6]]
  """
  cdef int l = array.shape[0]

  # Computing p, the dimension of the new square matrix
  cdef int sqrtdelta_l = int(sqrt(8*l+1))
  if sqrtdelta_l*sqrtdelta_l != 8*l+1:
    raise ValueError('The lenght of the array is not a tringular number!')
  cdef int p = (sqrtdelta_l - 1) // 2

  cdef np.ndarray[DTYPE_t, ndim=2] matrix = np.zeros(shape=(p,p), dtype=DTYPE)

  cdef int a, i = 0, j = 0
  for a in range(0,l):
    if j == p:
      i = i + 1
      j = i
    matrix[i][j] = array[a]
    matrix[j][i] = array[a]
    j = j + 1   
  
  return matrix


