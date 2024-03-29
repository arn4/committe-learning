import setuptools
from setuptools.extension import Extension

try:
    from Cython.Build import cythonize
except:
  print('You need Cython to install this package!')
  exit(1)
try:
  import numpy
except:
  print('You need numpy to install this package!')
  exit(1)

from committee_learning  import (
  __pkgname__ as PKG_NAME,
  __author__  as AUTHOR,
  __version__ as VERSION
)

# import platform
# if platform.system() == 'Darwin':
#   import os
#   os.environ["CC"] = "g++-12"

def get_extensions():

  include_dirs = [
    numpy.get_include(),
    'external_libraries/cpp-boost-math/include'
  ]

  extra_compile_args = [
    '-O3',
    '-funroll-loops',
    '-std=c++20'
  ]

  cython_ode_erf = Extension(
      name='committee_learning.ode.cython_erf',
      sources=['committee_learning/ode/erf.pyx', 'committee_learning/ode/erf_integrals.cpp'],
      include_dirs=include_dirs,
      language = 'cpp',
      extra_compile_args=extra_compile_args
  )

  cython_ode_H3 = Extension(
      name='committee_learning.ode.cython_H3',
      sources=['committee_learning/ode/H3.pyx', 'committee_learning/ode/H3_integrals.cpp'],
      include_dirs=include_dirs,
      language = 'cpp',
      extra_compile_args=extra_compile_args
  )

  cython_risk = Extension(
      name='committee_learning._cython.risk',
      sources=['committee_learning/_cython/risk.pyx', 'committee_learning/ode/erf_integrals.cpp'],
      include_dirs=include_dirs,
      extra_compile_args=extra_compile_args
  )

  cython_numpy_extra = Extension(
      name='committee_learning._cython.numpy_extra',
      sources=['committee_learning/_cython/numpy_extra.pyx'],
      include_dirs=include_dirs,
      extra_compile_args=extra_compile_args
  )

  return cythonize(
    [cython_risk, cython_ode_erf, cython_ode_H3, cython_numpy_extra],
    compiler_directives={'language_level':3},
    annotate=True
  )

setuptools.setup(
  setup_requires= [
    'Cython',
    'numpy',
    'setuptools>=18.0' 
  ],
  ext_modules=get_extensions(),
  package_data={
    'committee_learning/_config':['*.pxd']
  },
  name = PKG_NAME,
  author  =  AUTHOR,
  version = VERSION,
  packages = setuptools.find_packages(),
  python_requires = '>=3.7', # Probably it works even with newer version of python, but still...
  install_requires = [
    'numpy',
    'scipy',
    'torch',
    'scikit-learn',
    'matplotlib',
    'seaborn',
    'pyyaml',
    'tqdm',
  ],
  zip_safe=False
)