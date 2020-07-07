import os
from setuptools import setup, find_packages
from setuptools.extension import Extension
from Cython.Build import cythonize

import numpy as np

def read(fname):
    return open(os.path.join(os.path.dirname(__file__), fname)).read()

ext_modules=[
    Extension('hapflk.fastphase',
              sources = ["fastphase/fastphase.pyx"],
##              libraries=['m'],
                  include_dirs=[np.get_include()]),
    Extension('hapflk._pgenlib',
               sources = ['pgenlib/_pgenlib.pyx', 'pgenlib/pgenlib_python_support.cpp', 'pgenlib/pgenlib_internal.cpp', 'pgenlib/plink2_base.cpp'],
              language = "c++",
              # do not compile as c++11, since cython doesn't yet support
              # overload of uint32_t operator
              # extra_compile_args = ["-std=c++11", "-Wno-unused-function"],
              # extra_link_args = ["-std=c++11"],
              extra_compile_args = ["-std=c++98", "-Wno-unused-function"],
              extra_link_args = ["-std=c++98"],
              include_dirs = [np.get_include()]
              )
   ]
                  
setup(
    name='hapflk',
    version='2.0-dev1',
    description='haplotype-based test for differentiation in multiple populations',
    long_description=read('README'),
    license = "GPL v3",
    author='Bertrand Servin',
    author_email="bertrand.servin@inra.fr",
    url='https://forge-dga.jouy.inra.fr/projects/hapflk',
    packages=['hapflk'],
    ext_modules = cythonize(ext_modules),
    setup_requires= ['numpy','Cython'],
    install_requires = [
        'numpy',
        'scipy',
        'cython'],
    scripts=["bin/hapflk", "bin/hapflkadapt", "bin/flkpoptree", "bin/poolflkadapt", "bin/poolflkannot", "bin/flkfreq"],
    classifiers=[
        "Development Status :: 5 - Production/Stable",
        "Environment :: Console",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: GNU General Public License (GPL)",
        "Natural Language :: English",
        "Operating System :: OS Independent",
        "Programming Language :: Python",
        "Topic :: Scientific/Engineering",
        ]
)
