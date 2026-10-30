// Field ionization (ported from GEM-PIC-RZ: src/ionization.*, src/rng.hpp): ADK tunnel-ionization rates of the
// levels of an ion species in code units, usable on the device, and the counter-based random numbers.
//
// Rate of the transition k -> k+1 (Ammosov, Delone, Krainov, JETP 64, 1191 (1986); in the form of Chen et al.,
// JCP 236, 220 (2013) and FBPIC, s-states with l* = n* - 1, m = 0):
//   w = omega_a C^2 (U/2U_H) [2 (U/U_H)^{3/2} E_a/E]^{2n*-1} exp[-(2/3)(U/U_H)^{3/2} E_a/E]
//   n* = (k+1) sqrt(U_H/U),   C^2 = 2^{2n*} / (n* Gamma(n*+1) Gamma(n*))
// U: NIST ionization energy of the level, E: instantaneous |E| at the ion (the laser carrier is resolved, so the
// static-field rate applies). For hydrogen this is the Landau rate 4 omega_a (E_a/E) exp(-2 E_a/3E). Beyond the
// maximum of the expression (barrier suppression) the rate stays at its maximum.
#pragma once
#include <cstdint>
#include <vector>
#include "portable.hpp"

constexpr double U_H_EV = 13.605693122994;           // Rydberg energy [eV]
constexpr double E_ATOMIC_VM = 5.14220674763e11;     // atomic field [V/m]
constexpr double OMEGA_ATOMIC = 4.1341373335e16;     // atomic frequency [1/s]

// reference in SI units: rate [1/s] of a level with energy U [eV] of an ion that has the charge Zafter after
// the ionization, in the field E [V/m]
double adk_rate_SI(double U_eV, int Zafter, double E_Vm);

// one level in code units: w = exp(logpre - power ln E + expo / E) for Emin <= E (E capped at Epeak), else 0
struct AdkLevel { double logpre, power, expo, Epeak, Emin; };
HD inline double adk_rate(const AdkLevel& L, double E) {
    if (!(E >= L.Emin) || !(E > 0)) return 0.0;
    E = E < L.Epeak ? E : L.Epeak;
    return ::exp(L.logpre - L.power * ::log(E) + L.expo / E);
}
// levels Z0 ... Zmax-1 (ip_eV[k]: energy of k -> k+1); E0_SI = field unit [V/m], omega0 [1/s]; T = longest time
// in the window (below Emin a level is not ionized within T with probability > 1e-10)
std::vector<AdkLevel> adk_levels(const std::vector<double>& ip_eV, int Z0, int Zmax, double E0_SI, double omega0, double T);

// counter-based random numbers: a hash of (seed, step, slice, pass, particle id, draw), so all backends and
// thread schedules draw the same numbers for the same particle
HD inline uint64_t mix64(uint64_t z) {
    z += 0x9e3779b97f4a7c15ULL;
    z = (z ^ (z >> 30)) * 0xbf58476d1ce4e5b9ULL;
    z = (z ^ (z >> 27)) * 0x94d049bb133111ebULL;
    return z ^ (z >> 31);
}
HD inline double uniform01(uint64_t key) { return double(mix64(key) >> 11) * (1.0 / 9007199254740992.0); }
HD inline uint64_t ion_key_step(uint64_t seed, long step) { return mix64(seed * 0x9E3779B97F4A7C15ULL + uint64_t(step)); }
HD inline uint64_t ion_key_slice(uint64_t key_step, int n, int pass) { return mix64(key_step ^ ((uint64_t(n) << 8) | uint64_t(pass))); }
HD inline uint64_t ion_key_particle(uint64_t key_slice, uint64_t id) { return mix64(key_slice ^ mix64(id)); }
