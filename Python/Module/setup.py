from setuptools import setup, Extension
from Cython.Build import cythonize
import numpy

ext = Extension(
    name="wrapper",
    sources=["wrapper.pyx", "GDmod.c"],
    include_dirs=[numpy.get_include()],   # <-- add this line
    language="c",
)

setup(
    name="GDmodWrapper",
    ext_modules=cythonize(ext, compiler_directives={"language_level": "3"}),
    zip_safe=False,
)
