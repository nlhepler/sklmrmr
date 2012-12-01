#!/usr/bin/env python

import os.path, sys

from setuptools import Extension, setup

import numpy

np_inc = [os.path.join(os.path.dirname(numpy.__file__), 'core', 'include')]

ext_modules = [
    Extension('skmrmr._mrmr',
        sources=['skmrmr/_mrmr.c'],
        include_dirs=np_inc,
        libraries=['m'],
        extra_compile_args=['-O3']
        )
    ]

setup(name='skmrmr',
      version='0.1.0',
      description='minimum redundancy maximum relevance feature selection',
      author='N Lance Hepler',
      author_email='nlhepler@gmail.com',
      url='http://github.com/nlhepler/skmrmr',
      license='GNU GPL version 3',
      packages=['skmrmr'],
      package_dir={
            'skmrmr': 'skmrmr'
      },
      package_data={
            'skmrmr': ['*.csv']
      },
      ext_modules=ext_modules,
      requires=['numpy (>=1.6)', 'sklearn (>=0.12.1)']
     )
