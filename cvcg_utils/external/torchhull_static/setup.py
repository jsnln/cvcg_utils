#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use 
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#

from setuptools import setup
from torch.utils.cpp_extension import CUDAExtension, BuildExtension
import os
os.path.dirname(os.path.abspath(__file__))

setup(
    name="torchhull_static",
    version='0.2.0', 
    author='Patrick Stotko',
    author_email='stotko@cs.uni-bonn.de', 
    packages=['torchhull_static'],
    ext_modules=[
        CUDAExtension(
            name="torchhull_static._C",
            sources=[
                "torchhull_static/_C/src/gaussian_blur_cuda.cu",
                "torchhull_static/_C/src/marching_cubes_cuda.cu",
                "torchhull_static/_C/src/visual_hull_cuda.cu",
                
                "torchhull_static/_C/src/gaussian_blur.cpp",
                "torchhull_static/_C/src/io.cpp",
                "torchhull_static/_C/src/marching_cubes.cpp",
                "torchhull_static/_C/src/visual_hull.cpp",

                "torchhull_static/_C/python/bindings.cpp",
                ],
            extra_compile_args={"nvcc": ["-I" + os.path.join(os.path.dirname(os.path.abspath(__file__)), "torchhull_static/_C/include/"),
                                         "-I" + os.path.join(os.path.dirname(os.path.abspath(__file__)), "torchhull_static/_C/src/"),
                                         "--extended-lambda",
                                         "-gencode=arch=compute_70,code=sm_70",
                                         "-gencode=arch=compute_80,code=sm_80",
                                         "-gencode=arch=compute_80,code=compute_80",
                                         ],
                                "cxx": ["-I" + os.path.join(os.path.dirname(os.path.abspath(__file__)), "torchhull_static/_C/include/"),
                                        "-I" + os.path.join(os.path.dirname(os.path.abspath(__file__)), "torchhull_static/_C/src/"),
                                        ]},
            extra_link_args=["-lstdgpu"]
        )],
    cmdclass={
        'build_ext': BuildExtension
    }
)
