# Makefile for test with class implementations
CONDA_PREFIX ?= $(shell echo $$CONDA_PREFIX)
CUDA_HOME ?= /usr/local/cuda

# Get Python information
PYTHON_VERSION = $(shell python3 -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')")
PYTHON_INCLUDE_DIR = $(shell python3 -c "from sysconfig import get_paths; print(get_paths()['include'])")
PYTHON_LIB_DIR = $(shell python3 -c "import sysconfig; print(sysconfig.get_config_var('LIBDIR'))")
PYTHON_LIBS = $(shell python3-config --libs --embed)

# pybind11 paths
PYBIND11_INCLUDE = $(shell python3 -c "import pybind11; print(pybind11.get_include())")

# Torch paths
TORCH_PATH = $(shell python3 -c "import torch; import os; print(os.path.dirname(torch.__file__))")
TORCH_INCLUDE = -I$(TORCH_PATH)/include \
                -I$(TORCH_PATH)/include/torch/csrc/api/include

# Include paths
INCLUDES = -I$(PYTHON_INCLUDE_DIR) \
           -I$(PYBIND11_INCLUDE) \
           $(TORCH_INCLUDE) \
           -I$(CUDA_HOME)/include \
           -I.

# Compiler settings
CXX = g++
NVCC = $(CUDA_HOME)/bin/nvcc

# Compiler flags
CXX_FLAGS = -O3 \
            -std=c++14 \
            -fPIC \
            -pthread \
            $(INCLUDES) \
            -D_GLIBCXX_USE_CXX11_ABI=0

NVCC_FLAGS = -O3 \
             -x cu \
             -std=c++14 \
             -Xcompiler -fPIC \
             -gencode=arch=compute_90,code=sm_90 \
             --extended-lambda \
             --expt-relaxed-constexpr \
             --use_fast_math \
             -DCUDA_HAS_FP16=1 \
             --maxrregcount=255 \
             $(INCLUDES)

# Library paths and flags
LDFLAGS = -L$(TORCH_PATH)/lib \
          -L$(CUDA_HOME)/lib64 \
          -L$(PYTHON_LIB_DIR) \
          -lcublas \
          -lcusparse \
          -ltorch -ltorch_cpu -ltorch_cuda \
          -lcudart \
          -lc10 -lc10_cuda \
          $(PYTHON_LIBS)

# Rpath settings
RPATH = -Wl,-rpath,$(TORCH_PATH)/lib,-rpath,$(CUDA_HOME)/lib64,-rpath,$(PYTHON_LIB_DIR)

# Source files
CPP_SOURCES = test.cpp \
              spmm_cnic.cpp \
              class.cpp \
              common.h

CUDA_SOURCES = mma_tf32.cu \
               ptx_tf32.cu

# Object files
CPP_OBJECTS = $(CPP_SOURCES:.cpp=.o)
CUDA_OBJECTS = $(CUDA_SOURCES:.cu=.o)
OBJECTS = $(CPP_OBJECTS) $(CUDA_OBJECTS)

# Target executable
TARGET = test_spmm

.PHONY: all clean print-paths

all: $(TARGET)

$(TARGET): $(OBJECTS)
	$(CXX) $(CXX_FLAGS) -o $@ $(OBJECTS) $(LDFLAGS) $(RPATH)

# C++ compilation rules
%.o: %.cpp
	$(CXX) $(CXX_FLAGS) -c $< -o $@

# CUDA compilation rules
%.o: %.cu
	$(NVCC) $(NVCC_FLAGS) -c $< -o $@

clean:
	rm -f $(TARGET) $(OBJECTS)

print-paths:
	@echo "Python Version: $(PYTHON_VERSION)"
	@echo "Python Include Dir: $(PYTHON_INCLUDE_DIR)"
	@echo "Python Lib Dir: $(PYTHON_LIB_DIR)"
	@echo "Python Libs: $(PYTHON_LIBS)"
	@echo "pybind11 Include: $(PYBIND11_INCLUDE)"
	@echo "Torch Path: $(TORCH_PATH)"
	@echo "Sources:"
	@echo "  CPP: $(CPP_SOURCES)"
	@echo "  CUDA: $(CUDA_SOURCES)"
	@echo "Objects: $(OBJECTS)"