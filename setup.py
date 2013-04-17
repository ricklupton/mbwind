from setuptools import setup, find_packages
from distutils.extension import Extension
from Cython.Distutils import build_ext
import numpy as np

setup(
    name = "mbwind",
    version = "0.0.1",
    packages = find_packages(),

    cmdclass = {'build_ext': build_ext},
    ext_modules = [
        Extension("assemble", ["mbwind/assemble.pyx"],
                  include_dirs=[np.get_include()])
    ]
)
