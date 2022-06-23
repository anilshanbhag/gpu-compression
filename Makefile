CUDA_PATH       ?= /usr/local/cuda
CUDA_INC_PATH   ?= $(CUDA_PATH)/include
CUDA_BIN_PATH   ?= $(CUDA_PATH)/bin

NVCC = nvcc

SM_TARGETS 	= -gencode=arch=compute_70,code=\"sm_70,compute_70\" 
SM_DEF 		= -DSM700
TEST_ARCH 	= 520

GENCODE_SM50    := -gencode arch=compute_52,code=sm_52
GENCODE_FLAGS   := $(GENCODE_SM50)

NVCCFLAGS += --std=c++11 $(SM_DEF) -Xptxas="-dlcm=cg -v" -lineinfo -Xcudafe -\# 

# Folder structure
SRC = src
BIN = bin
OBJ = obj
INC = includes

CUB_DIR = ./cub/
INCLUDES = -I$(CUB_DIR) -I$(CUB_DIR)test -I. -I$(INC)

CFLAGS = -O3 -march=native -std=c++14 -ffast-math
LDFLAGS = -ltbb
CINCLUDES = -I$(INC)
CXX = clang++
#CXX = /big_fast_drive/anil/intel/bin/icc

DEPS = $(SRC)/kernel.cuh $(SRC)/simplebinpack_kernel.cuh $(SRC)/deltabinpack_kernel.cuh $(SRC)/ssb/ssb_gpu_utils.h $(SRC)/ssb/ssb_utils.h

$(OBJ)/cpu/%.o: $(SRC)/cpu/%.cpp
	$(CXX) $(CFLAGS) $(CINCLUDES) -c $< -o $@

$(BIN)/%: $(OBJ)/%.o $(DEPS)
	$(NVCC) $(SM_TARGETS) -lcurand $< -o $@	

$(OBJ)/%.o: $(SRC)/%.cu
	$(NVCC) -lcurand $(SM_TARGETS) $(NVCCFLAGS) $(CPU_ARCH) $(INCLUDES) $(LIBS) -O3 -dc $< -o $@

$(BIN)/cpu/%: $(OBJ)/cpu/%.o
	$(CXX) -ltbb $^ -o $@

ssb/binpack: src/ssb/binpack.cpp
	g++ $(CFLAGS) src/ssb/binpack.cpp -o bin/ssb/binpack

ssb/deltabinpack: src/ssb/deltabinpack.cpp
	g++ $(CFLAGS) src/ssb/deltabinpack.cpp -o bin/ssb/deltabinpack

bench/gen: src/bench/gen.cpp
	g++ $(CFLAGS) src/bench/gen.cpp -o bin/bench/gen

bench/gen_d1: src/bench/gen_d1.cpp
	g++ $(CFLAGS) src/bench/gen_d1.cpp -o bin/bench/gen_d1

bench/gen_d2: src/bench/gen_d2.cpp
	g++ $(CFLAGS) src/bench/gen_d2.cpp -o bin/bench/gen_d2

bench/gen_d3: src/bench/gen_d3.cpp
	g++ $(CFLAGS) src/bench/gen_d3.cpp -o bin/bench/gen_d3

bench/binpack: src/bench/binpack.cpp
	g++ $(CFLAGS) src/bench/binpack.cpp -o bin/bench/binpack

bench/deltabinpack: src/bench/deltabinpack.cpp
	g++ $(CFLAGS) src/bench/deltabinpack.cpp -o bin/bench/deltabinpack

testelem_bin: src/ssb/testelem_bin.cpp
	g++ $(CFLAGS) src/ssb/testelem_bin.cpp -o bin/testelem_bin

testelem_dbin: src/ssb/testelem_dbin.cpp
	g++ $(CFLAGS) src/ssb/testelem_dbin.cpp -o bin/testelem_dbin

ssb_q11: bin/ssb/q11
ssb_q12: bin/ssb/q12
ssb_q13: bin/ssb/q13
ssb_q21: bin/ssb/q21
ssb_q22: bin/ssb/q22
ssb_q23: bin/ssb/q23
ssb_q31: bin/ssb/q31
ssb_q32: bin/ssb/q32
ssb_q33: bin/ssb/q33
ssb_q34: bin/ssb/q34
ssb_q41: bin/ssb/q41
ssb_q42: bin/ssb/q42
ssb_q43: bin/ssb/q43

ssb:ssb_q11 ssb_q12  ssb_q13 ssb_q21 ssb_q22 ssb_q23 ssb_q31 ssb_q43 ssb_q32 ssb_q33 ssb_q34 ssb_q41 ssb_q42 ssb_q43

ssb_q11e: bin/ssb/q11e
ssb_q12e: bin/ssb/q12e
ssb_q13e: bin/ssb/q13e
ssb_q21e: bin/ssb/q21e
ssb_q22e: bin/ssb/q22e
ssb_q23e: bin/ssb/q23e
ssb_q31e: bin/ssb/q31e
ssb_q32e: bin/ssb/q32e
ssb_q33e: bin/ssb/q33e
ssb_q34e: bin/ssb/q34e
ssb_q41e: bin/ssb/q41e
ssb_q42e: bin/ssb/q42e
ssb_q43e: bin/ssb/q43e

ssbe:ssb_q11e ssb_q12e  ssb_q13e ssb_q21e ssb_q22e ssb_q23e ssb_q31e ssb_q43e ssb_q32e ssb_q33e ssb_q34e ssb_q41e ssb_q42e ssb_q43e

CUB_VER=1.8.0
setup:
	if [ ! -d "cub"  ]; then \
		wget https://github.com/NVlabs/cub/archive/$(CUB_VER).zip; \
		unzip $(CUB_VER).zip; \
		mv cub-$(CUB_VER) cub; \
		rm $(CUB_VER).zip; \
	fi
	mkdir -p obj/queries bin/queries

all: setup cpu gpu

codegentest:
	$(NVCC) $(DEFINES) -lcurand $(SM_TARGETS) -o $(BIN_DIR)codegentest_$(BIN_SUFFIX) codegentest.cu $(NVCCFLAGS) $(CPU_ARCH) $(INC) $(LIBS) -O3

gentest:
	$(NVCC) $(DEFINES) -lcurand $(SM_TARGETS) -o $(BIN_DIR)gentest_$(BIN_SUFFIX) gentest.cu $(NVCCFLAGS) $(CPU_ARCH) $(INC) $(LIBS) -O3

clean:
	rm -rf bin/* obj/gpu/*.o obj/cpu/*.o obj/*.o

