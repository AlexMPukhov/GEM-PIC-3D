CXX ?= g++
CXXFLAGS ?= -O3 -std=c++17 -Wall -Wextra -fopenmp
SRC = src/deck.cpp src/sim.cpp
all: build/gempic3d build/tests
build/gempic3d: $(SRC) src/main.cpp src/*.hpp
	@mkdir -p build
	$(CXX) $(CXXFLAGS) -o $@ $(SRC) src/main.cpp
build/tests: $(SRC) tests/tests.cpp src/*.hpp
	@mkdir -p build
	$(CXX) $(CXXFLAGS) -Isrc -o $@ $(SRC) tests/tests.cpp
test: build/tests
	build/tests
clean:
	rm -rf build
.PHONY: all test clean
