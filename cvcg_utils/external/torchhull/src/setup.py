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
    name="torchhull",
    packages=['torchhull'],
    ext_modules=[
        CUDAExtension(
            name="torchhull._C",
            sources=[
                "torchhull/_C/src/gaussian_blur_cuda.cu",
                "torchhull/_C/src/marching_cubes_cuda.cu",
                "torchhull/_C/src/visual_hull_cuda.cu",
                
                "torchhull/_C/src/gaussian_blur.cpp",
                "torchhull/_C/src/io.cpp",
                "torchhull/_C/src/marching_cubes.cpp",
                "torchhull/_C/src/visual_hull.cpp",

                "torchhull/_C/python/bindings.cpp",
                ],
            extra_compile_args={"nvcc": ["-I" + os.path.join(os.path.dirname(os.path.abspath(__file__)), "torchhull/_C/include/"),
                                         "-I" + os.path.join(os.path.dirname(os.path.abspath(__file__)), "torchhull/_C/src/"),
                                        #  "-L/usr/local/lib",
                                         "-lstdgpu",
                                         "--extended-lambda"],
                                "cxx": ["-I" + os.path.join(os.path.dirname(os.path.abspath(__file__)), "torchhull/_C/include/"),
                                        "-I" + os.path.join(os.path.dirname(os.path.abspath(__file__)), "torchhull/_C/src/"),
                                        #  "-L/usr/local/lib",
                                         "-lstdgpu",
                                        ]}
        )],
    cmdclass={
        'build_ext': BuildExtension
    }
)
