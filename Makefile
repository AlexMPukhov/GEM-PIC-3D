# make                 CPU build with OpenMP (build/gempic3d, build/tests)
# make MPI=1           with MPI (mpicxx)
# make CUDA=1          GPU build with nvcc and cuFFT (NVCC=..., CUDA_ARCH=-arch=sm_80); combine with MPI=1
# make EMU=1           GPU code paths on the CPU (src/emu: CUDA runtime and cuFFT emulated, device memory
#                      protected outside kernels); combine with MPI=1
# make FFTW=1          CPU build with FFTW 3 instead of the own FFT (-lfftw3 -lfftw3_omp)
# make physics         larger physics checks (ion collapse), minutes on a GPU or a 24-core node
# make HDF5=1          openPMD output (output.format = openpmd) with HDF5 1.12 or newer (HDF5_DIR or pkg-config hdf5;
#                      an older HDF5: warning, build without HDF5); combine with the others
SIM = src/sim_bunch_diag.cpp src/sim_bunch_file.cpp src/beam_import.cpp src/sim_config.cpp src/sim_setup.cpp src/sim_particles.cpp src/sim_fields.cpp src/sim_plasma.cpp src/sim_ions.cpp src/sim_bunches.cpp src/sim_external.cpp src/sim_checkpoint.cpp src/sim_output.cpp
SRC = src/deck.cpp src/grid.cpp $(SIM) src/comm.cpp src/openpmd.cpp src/laser_file.cpp src/atomic_data.cpp src/ionization.cpp src/expr.cpp
HDR = $(wildcard src/*.hpp src/emu/*.h)
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
  ifeq ($(EMU),1)
    CXXFLAGS += -DGEMPIC_EMU
  endif
  ifeq ($(FFTW),1)
    CXXFLAGS += -DGEMPIC_FFTW
    LDLIBS += -lfftw3_omp -lfftw3
  endif
endif

# HDF5 1.12 or newer. Flags: HDF5_CFLAGS / HDF5_LIBS if given, else HDF5_DIR (include/, lib/; set by the HDF5
# modules on atto and at JSC), else pkg-config hdf5, else the Ubuntu serial paths. The version is read from
# H5public.h (the -I paths, CPLUS_INCLUDE_PATH, CPATH, the system paths); an older HDF5 or no header: a warning
# and a build without HDF5. A parallel HDF5 (H5_HAVE_PARALLEL) includes mpi.h: a build without MPI=1 takes the
# include and library paths of $(MPICXX) (mpicxx --showme or -show). Test: tests/hdf5_version.sh (make test).
ifeq ($(HDF5),1)
  ifdef HDF5_DIR
    HDF5_CFLAGS ?= -I$(HDF5_DIR)/include
    ifeq ($(CUDA),1)
      HDF5_LIBS ?= -L$(HDF5_DIR)/lib -Xlinker -rpath=$(HDF5_DIR)/lib -lhdf5
    else
      HDF5_LIBS ?= -L$(HDF5_DIR)/lib -Wl,-rpath,$(HDF5_DIR)/lib -lhdf5
    endif
  endif
  HDF5_CFLAGS ?= $(shell pkg-config --cflags hdf5 2>/dev/null || echo -I/usr/include/hdf5/serial)
  HDF5_LIBS ?= $(shell pkg-config --libs hdf5 2>/dev/null || echo -L/usr/lib/x86_64-linux-gnu/hdf5/serial -lhdf5)
  H5_INCDIRS := $(patsubst -I%,%,$(filter -I%,$(HDF5_CFLAGS))) $(subst :, ,$(CPLUS_INCLUDE_PATH) $(CPATH)) \
                /usr/include /usr/local/include /usr/include/hdf5/serial
  H5_HEADER := $(firstword $(wildcard $(addsuffix /H5public.h,$(H5_INCDIRS))))
  ifeq ($(H5_HEADER),)
    $(warning warning: HDF5: H5public.h not found (HDF5_CFLAGS, HDF5_DIR, pkg-config hdf5): building without HDF5 (openPMD output and file input disabled))
    override HDF5 := 0
  else
    # "<version> <1 if >= 1.12> <1 if parallel>"
    H5_INFO := $(shell awk '/define H5_VERS_MAJOR/{a=$$3} /define H5_VERS_MINOR/{b=$$3} /define H5_VERS_RELEASE/{c=$$3} \
                 /define H5_HAVE_PARALLEL 1/{p=1} END{print a"."b"."c, (a>1||(a==1&&b>=12))?1:0, p?1:0}' \
                 $(H5_HEADER) $(dir $(H5_HEADER))H5pubconf.h 2>/dev/null)
    H5_VERSION := $(word 1,$(H5_INFO))
    ifneq ($(word 2,$(H5_INFO)),1)
      $(warning warning: HDF5 $(H5_VERSION) in $(dir $(H5_HEADER)) is too old (need 1.12 or newer): building without HDF5 (openPMD output and file input disabled); set HDF5_DIR to HDF5 1.12 or newer)
      override HDF5 := 0
    else
      CXXFLAGS += -DGEMPIC_HDF5 $(HDF5_CFLAGS)
      LDLIBS += $(HDF5_LIBS)
      ifeq ($(word 3,$(H5_INFO))$(filter 1,$(MPI)),1)
        MPICXX ?= mpicxx
        H5_MPI := $(shell $(MPICXX) --showme 2>/dev/null || $(MPICXX) -show 2>/dev/null)
        ifeq ($(filter -I%,$(H5_MPI)),)
          $(warning warning: HDF5 $(H5_VERSION) is a parallel build, but $(MPICXX) gives no MPI include path (mpi.h))
        endif
        CXXFLAGS += $(filter -I%,$(H5_MPI)) -DOMPI_SKIP_MPICXX -DMPICH_SKIP_MPICXX
        LDLIBS += $(filter -L%,$(H5_MPI)) $(filter -l%,$(H5_MPI))
      endif
    endif
  endif
endif

all: $(BUILD)/gempic3d $(BUILD)/tests
$(BUILD)/gempic3d: $(SRC) src/main.cpp $(HDR)
	@mkdir -p $(BUILD)
	$(CXX) $(CXXFLAGS) -o $@ $(SRC) src/main.cpp $(LDLIBS)
TESTS = tests/tests.cpp tests/test_fields.cpp tests/test_bunch.cpp tests/test_plasma.cpp tests/test_ions.cpp tests/test_ionization.cpp tests/test_bunches.cpp tests/test_external.cpp tests/test_expr.cpp tests/test_checkpoint.cpp tests/test_radiation.cpp tests/test_lasy.cpp tests/test_lessons.cpp
$(BUILD)/tests: $(SRC) $(TESTS) tests/test_util.hpp $(HDR)
	@mkdir -p $(BUILD)
	$(CXX) $(CXXFLAGS) -Isrc -o $@ $(SRC) $(TESTS) $(LDLIBS)
test: $(BUILD)/tests
	$(BUILD)/tests
	sh tests/hdf5_version.sh
TESTSRC = $(filter-out tests/tests.cpp,$(TESTS))
$(BUILD)/physics: $(SRC) $(TESTSRC) tests/physics.cpp tests/test_util.hpp $(HDR)
	@mkdir -p $(BUILD)
	$(CXX) $(CXXFLAGS) -Isrc -o $@ $(SRC) $(TESTSRC) tests/physics.cpp $(LDLIBS)
$(BUILD)/lessons: $(SRC) $(TESTSRC) tests/lessons_main.cpp tests/test_util.hpp $(HDR)
	@mkdir -p $(BUILD)
	$(CXX) $(CXXFLAGS) -Isrc -o $@ $(SRC) $(TESTSRC) tests/lessons_main.cpp $(LDLIBS)
physics: $(BUILD)/physics
	$(BUILD)/physics
emu-selftest: tests/emu_selftest.cpp $(HDR)
	@mkdir -p $(BUILD)
	$(CXX) $(CXXFLAGS) -Isrc -o $(BUILD)/emu_selftest tests/emu_selftest.cpp
	$(BUILD)/emu_selftest
clean:
	rm -rf build build-mpi build-gpu build-emu build-fftw build-h5
.PHONY: all test physics emu-selftest clean
