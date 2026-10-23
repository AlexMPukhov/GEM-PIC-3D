// Common part of the tests of GEM-PIC-3D (tests/tests.cpp runs them): the checks, default grids, helpers.
// Device data are read through host copies, so the same tests run in the CPU and the GPU build.
#pragma once
#include <algorithm>
#include <cmath>
#include <complex>
#include <cstdarg>
#include <cstdio>
#include <random>
#include <string>
#include "sim.hpp"

using cd = std::complex<double>;

extern int failures;
extern Comm comm;
inline void check(bool ok, const std::string& name, double val, double ref, double tol) {
    std::printf("%-66s %s  value %.6g  reference %.6g  tol %.2g\n", name.c_str(), ok ? "PASS" : "FAIL", val, ref, tol);
    std::fflush(stdout);
    if (!ok) ++failures;
}
inline void close(const std::string& name, double val, double ref, double rtol) {
    check(std::fabs(val - ref) <= rtol * std::fabs(ref), name, val, ref, rtol);
}
inline void info(const std::string& name, double val, const char* note) {   // reported, not checked
    std::printf("%-66s INFO  value %.6g  (%s)\n", name.c_str(), val, note);
    std::fflush(stdout);
}
inline std::string strf(const char* fmt, ...) {
    char buf[160];
    va_list ap; va_start(ap, fmt); std::vsnprintf(buf, sizeof buf, fmt, ap); va_end(ap);
    return buf;
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
inline cd zc(cpx v) { return {v.re, v.im}; }
inline double gamma_of(const Particle& p) { return std::sqrt(1 + p.px * p.px + p.py * p.py + p.pz * p.pz); }
// host copy of F: component comp (Ex Ey Ez Bx By Bz), slice n, plane index i = j*nx + ix
inline double field(const std::vector<float>& F, const Simulation& sim, int comp, int n, size_t i) {
    return F[(size_t(comp) * (sim.ns + 2) + n) * sim.nk + i];
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

void test_fft_pair();
void test_fft_sizes();
void test_fft_batched(int nx, int ny);
void test_grids();
void test_dispersion(const std::string& kind);
double test_diffraction(const std::string& kind, double ref_fd);
void test_plane_wave(const std::string& pol);
void test_gauss(const std::string& kind);
void test_equilibrium(bool freehand);
void test_sponge(bool freehand);
void test_constant_tables(bool yfourier);
void test_plasma_wake_1d(bool extrap);
void test_plasma_wake_beam();
void test_smoothing_divergence();
void test_streaming_kinematics();
