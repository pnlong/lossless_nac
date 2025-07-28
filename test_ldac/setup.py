from setuptools import setup, Extension
from Cython.Build import cythonize
import numpy as np
import os

extensions = [
    Extension(
        "bitstream",  # Changed to just "bitstream" so it can be imported directly
        ["bitstream.pyx"],  # Using relative path
        include_dirs=[np.get_include()],
        extra_compile_args=["-O3", "-march=native"],  # Enable high optimization and CPU-specific optimizations
        language="c++"
    )
]

setup(
    name="test_ldac",
    packages=['test_ldac'],
    ext_modules=cythonize(
        extensions,
        compiler_directives={
            "language_level": "3",
            "boundscheck": False,  # Disable bounds checking for better performance
            "wraparound": False,   # Disable negative indexing
            "initializedcheck": False,  # Disable checking if memoryviews are initialized
            "cdivision": True,     # Disable checking for zero division
            "nonecheck": False,    # Disable checking for None
        }
    ),
    zip_safe=False,
    install_requires=[
        "numpy",
        "cython"
    ]
) 