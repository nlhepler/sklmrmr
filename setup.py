#!/usr/bin/env python

import os.path, sys

from setuptools import Extension, setup

import numpy

np_inc = [os.path.join(os.path.dirname(numpy.__file__), 'core', 'include')]

ext_modules = [
    Extension('sklmrmr._mrmr',
        sources=['sklmrmr/_mrmr.c'],
        include_dirs=np_inc,
        libraries=['m'],
        extra_compile_args=['-O3']
        )
    ]

setup(name='sklmrmr',
      version='0.2.0',
      description='minimum redundancy maximum relevance feature selection',
      author='N Lance Hepler',
      author_email='nlhepler@gmail.com',
      url='http://github.com/nlhepler/sklmrmr',
      license='GNU GPL version 3',
      packages=['sklmrmr'],
      package_dir={
            'sklmrmr': 'sklmrmr'
      },
      package_data={
            'sklmrmr': ['*.csv']
      },
      ext_modules=ext_modules,
      requires=['numpy (>=1.6)', 'sklearn (>=0.14.0)']
     )
