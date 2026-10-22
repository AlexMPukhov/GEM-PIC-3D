// Common part of the tests of GEM-PIC-3D (tests/tests.cpp runs them): the checks, default grids, helpers.
// Device data are read through host copies, so the same tests run in the CPU and the GPU build.
#pragma once
#include <algorithm>
#include <cmath>
#include <complex>
#include <cstdio>
#include <random>
#include "sim.hpp"

extern int failures;
extern Comm comm;
inline void check(bool ok, const char* name, double val, double ref, double tol) {
    std::printf("%-66s %s  value %.6g  reference %.6g  tol %.2g\n", name, ok ? "PASS" : "FAIL", val, ref, tol);
    std::fflush(stdout);
    if (!ok) ++failures;
}
inline void close(const char* name, double val, double ref, double rtol) {
    check(std::fabs(val - ref) <= rtol * std::fabs(ref), name, val, ref, rtol);
}

// default grids: Fourier modes in x and y, nz uniform slices over the window Lw
inline Config base(int nx, int ny, int nz, double Lx, double Ly, double Lw) {
    Config c = Config::base_units(10e-6);
    c.nx = nx; c.Lx = Lx;
    c.y_uniform(ny, Ly);
    c.zeta_uniform(nz, Lw);
    c.sponge = c.sponge_y = 0; c.move_x = false;
    return c;
}
inline double mean_pz(Simulation& sim) {
    double s = 0; long n = 0;
    for (auto& p : sim.host_particles()) if (p.alive == 1) { s += p.pz; ++n; }
    return s / n;
}
inline int nearest(const std::vector<double>& v, double x) {
    int k = 0;
    for (int i = 1; i < int(v.size()); ++i) if (std::fabs(v[i] - x) < std::fabs(v[k] - x)) k = i;
    return k;
}

void test_fft(int nx, int ny);
void test_grids();
void test_dispersion(const std::string& kind);
double test_diffraction(const std::string& kind, double ref_fd);
void test_plane_wave(const std::string& pol);
void test_gauss(const std::string& kind);
void test_equilibrium(bool freehand);
void test_sponge(bool freehand);
void test_constant_tables(bool yfourier);
void test_plasma_wake_1d(bool extrap);
void test_plasma_wake_beam(bool freehand);
void test_smoothing_divergence();
void test_streaming_kinematics();
