# Adapted from https://github.com/pybind/cmake_example/blob/master/setup.py
import os
import re
import sys
import platform, sysconfig
import subprocess
import importlib
from sysconfig import get_paths

import importlib
from setuptools import setup, Extension
from setuptools.command.build_ext import build_ext
from setuptools.command.install import install
from distutils.sysconfig import get_config_var
from distutils.version import LooseVersion

class CMakeExtension(Extension):
    def __init__(self, name, sourcedir, build_with_cuda):
        Extension.__init__(self, name, sources=[])
        self.sourcedir = os.path.abspath(sourcedir)
        self.build_with_cuda = build_with_cuda

class Build(build_ext):
    def run(self):
        try:
            out = subprocess.check_output(['cmake', '--version'])
        except OSError:
            raise RuntimeError("CMake must be installed to build the following extensions: " +
                               ", ".join(e.name for e in self.extensions))

        super().run()

    def build_extension(self, ext):
        if isinstance(ext, CMakeExtension):
            extdir = os.path.abspath(os.path.dirname(self.get_ext_fullpath(ext.name)))
            info = get_paths()

            if platform.system() == "Windows":
                # change this to fit your python install
                libdir = "C:\\Users\\micha\\miniforge3\\envs\\venv\\libs\\python310.lib"
                ex = "C:\\Users\\micha\\miniforge3\\envs\\venv\\python.exe"
            else:   
                libdir = get_config_var('LIBDIR')

            python_exe = sys.executable
            python_root = sys.prefix
            include_path = info['include'] #sysconfig.get_path("include")

            cmake_args = [
                '-DCMAKE_LIBRARY_OUTPUT_DIRECTORY=' + extdir,
                '-DPython_EXECUTABLE=' + python_exe,
                '-DPython_ROOT_DIR=' + python_root,
                '-DPython_INCLUDE_DIR=' + include_path,
            ]

            # include_path = info['include']

            # cmake_args = ['-DCMAKE_LIBRARY_OUTPUT_DIRECTORY=' + extdir,
            #               '-DPython_LIBRARY=' + libdir,
            #               '-DPython_INCLUDE_DIR=' + include_path]

            cfg = 'Debug' if self.debug else 'Release'
            build_args = ['--config', cfg]

            if platform.system() == "Windows":
                cmake_args += ['-DCMAKE_LIBRARY_OUTPUT_DIRECTORY_{}={}'.format(cfg.upper(), extdir),
                               '-DPython_EXECUTABLE=' + ex,
                               '-DCMAKE_RUNTIME_OUTPUT_DIRECTORY_{}={}'.format(cfg.upper(), extdir)]
                
                if sys.maxsize > 2**32:
                    cmake_args += ['-A', 'x64']
                build_args += ['--', '/m']
            else:
                cmake_args += ['-DCMAKE_BUILD_TYPE=' + cfg]
                build_args += ['--', '-j8']

            if ext.build_with_cuda:
                cmake_args += [
                    '-DDIFFVG_CUDA=1',
                    '-DCUDAToolkit_ROOT=' + sys.prefix,
                ]
                cmake_args += ['-DDIFFVG_CUDA=1']

                # cuda_root = os.environ.get("CUDA_TOOLKIT_ROOT_DIR") or sys.prefix
                # cuda_target = os.path.join(cuda_root, "targets", "x86_64-linux")

                # cmake_args += [
                #     "-DCUDA_TOOLKIT_ROOT_DIR=" + cuda_root,
                # ]

                # nvcc = os.path.join(cuda_root, "bin", "nvcc")
                # if os.path.exists(nvcc):
                #     cmake_args += ["-DCUDA_NVCC_EXECUTABLE=" + nvcc]

                # # Prefer the singular cache/input form for old FindCUDA
                # cuda_include = os.path.join(cuda_target, "include")
                # if os.path.exists(cuda_include):
                #     cmake_args += ["-DCUDA_INCLUDE_DIR=" + cuda_include]

                # cudart = os.path.join(cuda_target, "lib", "libcudart.so")
                # if os.path.exists(cudart):
                #     cmake_args += ["-DCUDA_CUDART_LIBRARY=" + cudart]

                # print("CUDA root:", cuda_root)
                # print("nvcc:", nvcc, os.path.exists(nvcc))
                # print("cuda include:", cuda_include, os.path.exists(cuda_include))
                # print("cudart:", cudart, os.path.exists(cudart))
                # print("cmake args:", cmake_args)
                
                # cuda_root = os.environ.get("CUDA_HOME") or os.environ.get("CUDA_PATH") or "/usr/local/cuda"
                # cmake_args += [
                #     '-DCUDA_TOOLKIT_ROOT_DIR=' + cuda_root,
                # ]
            else:
                cmake_args += ['-DDIFFVG_CUDA=0']

                
            env = os.environ.copy()
            env['CXXFLAGS'] = '{} -DVERSION_INFO=\\"{}\\"'.format(env.get('CXXFLAGS', ''),
                                                                  self.distribution.get_version())
            if not os.path.exists(self.build_temp):
                os.makedirs(self.build_temp)
            subprocess.check_call(['cmake', ext.sourcedir] + cmake_args, cwd=self.build_temp, env=env)
            subprocess.check_call(['cmake', '--build', '.'] + build_args, cwd=self.build_temp)
        else:
            super().build_extension(ext)

torch_spec = importlib.util.find_spec("torch")
tf_spec = importlib.util.find_spec("tensorflow")
packages = []
build_with_cuda = False
if torch_spec is not None:
    packages.append('pydiffvg')
    import torch
    if torch.cuda.is_available():
        build_with_cuda = True
if False: #tf_spec is not None and sys.platform != 'win32':
    assert(False)
    packages.append('pydiffvg_tensorflow')
    if not build_with_cuda:
        import tensorflow as tf
        if tf.test.is_gpu_available(cuda_only=True, min_cuda_compute_capability=None):
            build_with_cuda = True
if len(packages) == 0:
    print('Error: PyTorch or Tensorflow must be installed. For Windows platform only PyTorch is supported.')
    exit()
# Override build_with_cuda with environment variable
if 'DIFFVG_CUDA' in os.environ:
    build_with_cuda = os.environ['DIFFVG_CUDA'] == '1'

setup(name = 'diffvg',
      version = '0.0.1',
      install_requires = ["svgpathtools"],
      description = 'Differentiable Vector Graphics',
      ext_modules = [CMakeExtension('diffvg', '', build_with_cuda)],
      cmdclass = dict(build_ext=Build, install=install),
      packages = packages,
      zip_safe = False)
