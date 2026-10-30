// External (applied) fields acting on the particles, as in GEM-PIC-RZ (src/external.hpp): undulators and the
// transport elements quadrupole, plasma lens and solenoid. Static magnetic fields in the lab frame at the lab
// position (x, y, z) of a particle, added to the gathered self-fields; the field solver does not see them.
// Code units: B in m omega0 / e, k in k0. With k_u = 2 pi / period, B0 = K k_u, psi = k_u (z - z_start) and the
// envelope g(z) (linear ramps of `ramp` at both ends, 0 outside):
//   planar:   B_y = B0 g cosh(k_u y) cos(psi), B_z = -B0 g sinh(k_u y) sin(psi)   (natural focusing in y);
//             without natural focusing B_y = B0 g cos(psi)
//   helical:  B_x = B0 g cos(psi), B_y = B0 g sin(psi)
//   quadrupole  B_x = -G y, B_y = -G x            (G > 0 focuses electrons in x)
//   plasma lens B_phi = -G r inside R, -G R^2/r outside (G > 0 focuses electrons)
//   solenoid    B_z = B f(z), B_r = -(r/2) B f'(z) (f: linear fringes of length `ramp`)
#pragma once
#include "portable.hpp"

struct Undulator {
    int helical, focusing;
    double ku, B0, z0, z1, ramp;
    int nk, koff; double dzk;        // taper (K_function): nk values B0(z0 + j dzk) at External::kt[koff + j]
};
struct Magnet {
    int kind;                        // 0 quadrupole, 1 plasma lens, 2 solenoid
    double z0, z1, g, R, ramp;       // g: gradient (quadrupole, lens) or field (solenoid)
    double dsmax;                    // largest push sub-step inside the element (0.1 / k_beta of the reference)
};
struct External {
    int nu = 0, nm = 0;
    const Undulator* u = nullptr;
    const Magnet* m = nullptr;
    const double* kt = nullptr;
    HD bool any() const { return nu > 0 || nm > 0; }
};

HD inline double undulator_envelope(const Undulator& U, double z) {
    if (z <= U.z0 || z >= U.z1) return 0.0;
    if (U.ramp <= 0) return 1.0;
    const double a = (z - U.z0) / U.ramp, b = (U.z1 - z) / U.ramp;
    return a < b ? (a < 1 ? a : 1.0) : (b < 1 ? b : 1.0);
}

// adds the external B at the lab position (x, y, z) (E is not changed: all elements are magnetic)
HD inline void add_external(const External& X, double x, double y, double z, double B[3]) {
    for (int k = 0; k < X.nu; ++k) {
        const Undulator& U = X.u[k];
        const double g = undulator_envelope(U, z);
        if (g == 0) continue;
        const double psi = U.ku * (z - U.z0);
        double b = U.B0 * g;
        if (U.nk > 0) {                              // taper: linear interpolation in the table
            const double t = (z - U.z0) / U.dzk;
            const int j = t < double(U.nk - 2) ? int(t) : U.nk - 2;
            const double w = t - j;
            b = g * ((1 - w) * X.kt[U.koff + j] + w * X.kt[U.koff + j + 1]);
        }
        if (U.helical) { B[0] += b * ::cos(psi); B[1] += b * ::sin(psi); }
        else if (U.focusing) { B[1] += b * ::cosh(U.ku * y) * ::cos(psi); B[2] -= b * ::sinh(U.ku * y) * ::sin(psi); }
        else B[1] += b * ::cos(psi);
    }
    for (int k = 0; k < X.nm; ++k) {
        const Magnet& G = X.m[k];
        if (z < G.z0 || z >= G.z1) continue;
        if (G.kind == 0) { B[0] -= G.g * y; B[1] -= G.g * x; }
        else if (G.kind == 1) {
            const double r2 = x * x + y * y, R2 = G.R * G.R, s = r2 <= R2 ? G.g : G.g * R2 / r2;
            B[0] += s * y; B[1] -= s * x;
        } else {
            double f = 1, df = 0;
            if (G.ramp > 0) {
                if (z - G.z0 < G.ramp) { f = (z - G.z0) / G.ramp; df = 1 / G.ramp; }
                else if (G.z1 - z < G.ramp) { f = (G.z1 - z) / G.ramp; df = -1 / G.ramp; }
            }
            B[2] += G.g * f; B[0] -= 0.5 * x * G.g * df; B[1] -= 0.5 * y * G.g * df;
        }
    }
}

// largest push sub-step from z: lambda_u / 32 inside an undulator (RK4: the slippage to 1e-5), dsmax inside a
// magnet, and never across a breakpoint of an element (entrance, ends of the ramps, exit), where the field or its
// derivative jumps (RK4 would drop to first order there)
HD inline double external_step(const External& X, double z, double ds) {
    auto land = [&](double b) { if (b > z + 1e-9 && b < z + ds) ds = b - z; };
    for (int k = 0; k < X.nu; ++k) {
        const Undulator& U = X.u[k];
        if (z >= U.z0 - 1e-9 && z < U.z1) { const double s = 6.283185307179586 / (32 * U.ku); ds = s < ds ? s : ds; }
        land(U.z0); land(U.z0 + U.ramp); land(U.z1 - U.ramp); land(U.z1);
    }
    for (int k = 0; k < X.nm; ++k) {
        const Magnet& G = X.m[k];
        if (z >= G.z0 - 1e-9 && z < G.z1 && G.dsmax < ds) ds = G.dsmax;
        land(G.z0); land(G.z0 + G.ramp); land(G.z1 - G.ramp); land(G.z1);
    }
    return ds;
}
