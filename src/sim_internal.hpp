// Internal header of the Simulation sources (src/sim_*.cpp): common includes, physical constants, MPI tags.
#pragma once
#include "sim.hpp"
#include <algorithm>
#include <cmath>
#include <complex>
#include <fstream>
#include <random>
#include <stdexcept>

namespace {
constexpr double c_SI = 299792458.0, e_SI = 1.602176634e-19, me_SI = 9.1093837015e-31, eps0 = 8.8541878128e-12;
constexpr double q = -1.0;                       // electron charge in units of e
enum Tag { TAG_HAND = 11, TAG_CUR = 12, TAG_BND = 13, TAG_GHOST = 14, TAG_TRIF = 15, TAG_TRIB = 16, TAG_JZU = 17, TAG_JZD = 18 };
HD inline bool is_half(int comp) { return comp == 1 || comp == 3 || comp == 5; }   // E_y, B_x, B_z
}  // namespace
