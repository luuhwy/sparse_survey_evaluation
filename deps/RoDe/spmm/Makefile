TARGET = librode_spmm.a

SRCS = $(wildcard *.cu) $(wildcard *.cc)
OBJS = $(patsubst %cu, %cu.o, $(SRCS)) $(patsubst %cc, %cc.o, $(SRCS))

CUDA_HOME ?= /usr/local/cuda
NVCC = $(CUDA_HOME)/bin/nvcc

# GPU architecture: 89=Ada Lovelace (RTX4090/L40), 90=Hopper (H100)
GPU_ARCH ?= 90
# Host C compiler (override via CC= on command line or parent Makefile)
CC ?= gcc

NVCC_FLAGS =
NVCC_FLAGS += -O3 -lcudart
NVCC_FLAGS += -shared -std=c++17
NVCC_FLAGS += -Xcompiler -fPIC
NVCC_FLAGS += -Xcompiler -fopenmp
NVCC_FLAGS += -ccbin $(CC)
NVCC_FLAGS += -m64
NVCC_FLAGS += -gencode arch=compute_$(GPU_ARCH),code=sm_$(GPU_ARCH)
NVCC_FLAGS += -diag-suppress=191,177

CC_FLAGS = -O3 -Wall -std=c++17 -fPIC

INC = -I$(CUDA_HOME)/include

all: $(TARGET)

$(TARGET): $(OBJS)
	ar cr $@ $^
	ranlib $@

%.cu.o: %.cu
	$(NVCC) $(NVCC_FLAGS) $(INC) -c $< -o $@

%.cc.o: %.cc
	$(CPP) $(CC_FLAGS) $(INC) -c $< -o $@

clean:
	rm -f *.o $(TARGET)
