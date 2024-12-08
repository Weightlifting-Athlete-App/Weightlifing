from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension
import numpy as np

setup(
    name='gpu_nms',
    ext_modules=[
        CUDAExtension(
            name='gpu_nms',
            sources=['gpu_nms.pyx', 'nms_kernel.cu'],
            include_dirs=[np.get_include()],
            extra_compile_args={
                'cxx': [],
                'nvcc': ['-arch=sm_75', '-O2'],
            },
        )
    ],
    cmdclass={
        'build_ext': BuildExtension
    }
)
