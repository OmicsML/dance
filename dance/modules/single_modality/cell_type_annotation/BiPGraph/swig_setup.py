#!/usr/bin/env python

"""
    setup.py file for SWIG
"""
import sys
import os
from setuptools import setup
from setuptools.command.build_ext import build_ext
from distutils.core import setup, Extension

# 获取当前 Conda 环境的根目录
# 这样可以自动找到安装在 Conda 环境里的 cnpy 头文件和库文件
conda_prefix = sys.prefix

class BuildExt(build_ext):
    def build_extensions(self):
        # Check if the flag exists before removing to avoid ValueError
        if '-Wstrict-prototypes' in self.compiler.compiler_so:
            self.compiler.compiler_so.remove('-Wstrict-prototypes')
        super(BuildExt, self).build_extensions()

test_module = Extension('_BiP',
                        sources=['precompute_wrap.cxx','precompute/Propagation.cpp'],
                        swig_opts=['-c++'],
                        
                        # 1. 告诉编译器去哪里找头文件 (cnpy.h)
                        include_dirs=[os.path.join(conda_prefix, 'include')],
                        
                        # 2. 告诉链接器去哪里找库文件 (libcnpy.so)
                        library_dirs=[os.path.join(conda_prefix, 'lib')],
                        
                        # 3. 指定要链接的库名称 (去掉 lib 前缀和后缀)
                        libraries=['cnpy', 'z'],
                        
                        # 4. 仅保留编译参数 (去掉 -L 和 -l)
                        extra_compile_args=['-std=c++11', '-O3', '-pthread', '-march=core2'],
                        
                        # 5. 链接参数 (通常只需保留 pthread，去掉重复的 compile args)
                        extra_link_args=['-pthread']
                        )
                        
setup(name = 'BiP',
        version = '0.1',
        cmdclass={'build_ext': BuildExt},
        ext_modules = [test_module],
        py_modules = ['BiP'],)