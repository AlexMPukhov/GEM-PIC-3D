# make                 CPU build with OpenMP (build/gempic3d, build/tests)
# make MPI=1           with MPI (mpicxx)
# make CUDA=1          GPU build with nvcc and cuFFT (NVCC=..., CUDA_ARCH=-arch=sm_80); combine with MPI=1
SRC = src/deck.cpp src/sim.cpp src/comm.cpp
HDR = $(wildcard src/*.hpp)
BUILD ?= build
ifeq ($(CUDA),1)
  NVCC ?= nvcc
  CUDA_ARCH ?= -arch=native
  CXX := $(NVCC)
  CXXFLAGS = -O3 -std=c++17 $(CUDA_ARCH) -x cu --extended-lambda -Xcompiler -Wall
  LDLIBS = -lcufft
  ifeq ($(MPI),1)
    CXXFLAGS += -ccbin mpicxx -DGEMPIC_MPI -DOMPI_SKIP_MPICXX -DMPICH_SKIP_MPICXX
  endif
else
  ifeq ($(MPI),1)
    CXX := mpicxx
    CXXFLAGS = -O3 -std=c++17 -Wall -Wextra -fopenmp -DGEMPIC_MPI -DOMPI_SKIP_MPICXX -DMPICH_SKIP_MPICXX
  else
    CXX ?= g++
    CXXFLAGS = -O3 -std=c++17 -Wall -Wextra -fopenmp
  endif
  LDLIBS =
endif

all: $(BUILD)/gempic3d $(BUILD)/tests
$(BUILD)/gempic3d: $(SRC) src/main.cpp $(HDR)
	@mkdir -p $(BUILD)
	$(CXX) $(CXXFLAGS) -o $@ $(SRC) src/main.cpp $(LDLIBS)
$(BUILD)/tests: $(SRC) tests/tests.cpp $(HDR)
	@mkdir -p $(BUILD)
	$(CXX) $(CXXFLAGS) -Isrc -o $@ $(SRC) tests/tests.cpp $(LDLIBS)
test: $(BUILD)/tests
	$(BUILD)/tests
clean:
	rm -rf build build-mpi build-gpu
.PHONY: all test clean
