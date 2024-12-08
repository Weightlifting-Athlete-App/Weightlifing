from setuptools import setup, Extension
from Cython.Build import cythonize
import numpy as np

ext_modules = [
    Extension(
        name='cpu_nms',
        sources=['cpu_nms.pyx'],
        include_dirs=[np.get_include()],
    )
]

setup(
    name='cpu_nms',
    ext_modules=cythonize(ext_modules),
)
