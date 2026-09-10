from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension
import os

# 获取CUDA路径
CUDA_HOME = os.environ.get("CUDA_HOME", "/usr/local/cuda")

# CUDA编译参数
nvcc_flags = [
    '-O3',
    '-x', 'cu',  # 强制以CUDA方式编译
    '-gencode=arch=compute_90,code=sm_90',
    '--extended-lambda',
    '--expt-relaxed-constexpr',
    '--use_fast_math',
#    '-D__CUDA_ARCH__=860',
    '-DCUDA_HAS_FP16=1',
    '--maxrregcount=255',
#    '--ptxas-options=-v',
    '-lcublas',
    '-lcusparse',
]
cpp_flags = [
    '-O3',
    '-std=c++14',
    '-fPIC',  # 生成位置无关代码
    '-Dtf',
    '-Dtranspose_',
]
setup(
    name='spmm_cnic',
    ext_modules=[
        CUDAExtension(
            name='spmm_cnic',
            sources=[
                'src/bindings/module.cu',
                'src/core/class.cu',
                'src/core/spmm_dispatcher.cu',
                'src/core/spmm_kernel.cu',
                'src/core/spmm_launcher.cu',
                'src/core/utils.cpp'
            ],
            include_dirs=[os.path.abspath('./include')], # 需要使用绝对路径
            extra_compile_args={
                # 'cxx': cpp_flags,  # C++编译参数
                # 'nvcc': nvcc_flags
                #'cxx': ['-std=c++17', '-O3'],
                # TODO 添加预处理选项，但是目前这个简单的名称会和内部库的符号发生冲突，导致编译错误 '-Dtf', '-Dtranspose_'
                #'nvcc': ['-std=c++17', '-O3', '--extended-lambda'] 
                #'nvcc': ['-std=c++17', '-g', '-G', '-O0', '-arch=sm_90'] 
                'cxx': ['-std=c++17', '-g', '-O0'],
                'nvcc': ['-std=c++17', '-g', '-G', '-O0', '--extended-lambda', '-lineinfo']
            }
        )
    ],
    cmdclass={
        'build_ext': BuildExtension.with_options(use_ninja=True)
    }
)