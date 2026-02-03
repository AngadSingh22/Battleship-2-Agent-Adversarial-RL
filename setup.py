from setuptools import setup, Extension
from distutils.command.build_ext import build_ext
import os

# Define the extension
# note: we want a shared library, usually Extension builds a python module (.pyd). 
# But for ctypes we just need the .dll. 
# Better yet, let's build it as a proper C-Extension (BattleshipC) and wrap it?
# Or stick to the plan: Shared Library.
# setuptools isn't great for just "DLLs", but we can abuse it or just build a .pyd and load it via ctypes?
# Actually, loading a .pyd via ctypes works on Windows (it's a DLL).

# Let's map the source
module = Extension(
    'csrc.libbattleship',
    sources=['csrc/src/battleship.c'],
    include_dirs=['csrc/include'],
    extra_compile_args=['/O2'] if os.name == 'nt' else ['-O3']
)

setup(
    name='battleship_c',
    version='1.0',
    description='C Optimized Battleship Kernel',
    ext_modules=[module],
    packages=[], # Disable auto-discovery
)
