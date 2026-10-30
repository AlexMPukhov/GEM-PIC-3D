// Tests: undulators and transport elements (single test electrons, no self-fields).
#include "test_util.hpp"

// one electron (gamma, offset x0, y0) through the external elements of c; the state at every step
struct Track { std::vector<double> z, x, y, zeta, px, py, pz; };
static Track track(Config c, double gamma, double x0, double y0, double zc) {
    c.self_fields = false;
    Config::Bunch b;
    b.charge = 1e-18; b.gamma = gamma; b.nmacro = 1; b.zeta_center = zc; b.x0 = x0; b.y0 = y0;
    c.bunches = {b};
    Simulation sim(c, comm);
    sim.init();
    Track t;
    for (;;) {
        const Particle p = sim.host_particles()[0];
        t.z.push_back(sim.s); t.x.push_back(p.x + sim.X0); t.y.push_back(p.y + sim.Y0); t.zeta.push_back(p.zeta);
        t.px.push_back(p.px); t.py.push_back(p.py); t.pz.push_back(p.pz);
        if (sim.finished()) break;
        sim.step();
    }
    return t;
}
static Config ext_base(double dz, double zend) {
    Config c = base(4, 4, 8, 200, 200, 400);
    c.move_x = c.move_y = false; c.dz = dz; c.dz_push = dz; c.z_end = zend;
    return c;
}

// 20. Undulators, one electron with gamma = 100, K = 1, lambda_u = 200 k0^-1 (pushed with lambda_u / 16):
//     planar: u_x = K sin(psi) in the flat part (amplitude, no drift), and the slippage over 6 periods against the
//     exact integral of (gamma - p_z)/p_z for u_x = K sin(psi); helical: |u_perp| = K; natural focusing: the
//     betatron wavenumber K k_u / (sqrt(2) gamma) of an electron at y0 = 1 k0^-1 (half period from two zero
//     crossings of y).
void test_undulator() {
    const double lu = 200, K = 1, g = 100;
    Config c = ext_base(lu / 8, 13 * lu);
    Undulator u{};
    u.ku = 2 * M_PI / lu; u.B0 = K * u.ku; u.z0 = 0; u.z1 = 12 * lu; u.ramp = lu; u.focusing = 1;
    c.undulators = {u};
    Track t = track(c, g, 0, 0, 300);
    double amax = 0, mean = 0; int cnt = 0; double za = 0, zb = 0;
    for (size_t k = 0; k < t.z.size(); ++k) {
        if (t.z[k] >= 2 * lu - 1e-9 && t.z[k] < 10 * lu - 1e-9) { amax = std::max(amax, std::fabs(t.px[k])); mean += t.px[k]; ++cnt; }
        if (std::fabs(t.z[k] - 3 * lu) < 1e-6) za = t.zeta[k];
        if (std::fabs(t.z[k] - 9 * lu) < 1e-6) zb = t.zeta[k];
    }
    close("undulator (planar): amplitude of u_x = K", amax, K, 1e-4);
    check(std::fabs(mean / cnt) < 1e-3 * K, "undulator (planar): no drift of u_x in the flat part", mean / cnt, 0, 1e-3);
    double slip = 0;                                       // exact: 6 lambda_u <(gamma - p_z)/p_z>
    const int M = 20000;
    for (int k = 0; k < M; ++k) {
        const double ux = K * std::sin(2 * M_PI * (k + 0.5) / M), pz = std::sqrt(g * g - 1 - ux * ux);
        slip += (g - pz) / pz / M;
    }
    close("undulator (planar): slippage over 6 periods (exact integral)", za - zb, 6 * lu * slip, 1e-4);
    c.undulators[0].helical = 1; c.undulators[0].focusing = 0;
    t = track(c, g, 0, 0, 300);
    double dev = 0;
    for (size_t k = 0; k < t.z.size(); ++k)
        if (t.z[k] >= 2 * lu - 1e-9 && t.z[k] < 10 * lu - 1e-9) dev = std::max(dev, std::fabs(std::hypot(t.px[k], t.py[k]) - K));
    check(dev < 1e-4 * K, "undulator (helical): |u_perp| = K in the flat part", dev, 0, 1e-4);
    // natural focusing: 160 periods, y0 = 1
    Config cf = ext_base(lu / 4, 162 * lu);
    u.helical = 0; u.focusing = 1; u.z1 = 161 * lu;
    cf.undulators = {u};
    t = track(cf, g, 0, 1, 300);
    std::vector<double> zc;
    for (size_t k = 1; k < t.z.size(); ++k)
        if (t.z[k] > 2 * lu && t.z[k] < 159 * lu && (t.y[k - 1] > 0) != (t.y[k] > 0))
            zc.push_back(t.z[k - 1] + (t.z[k] - t.z[k - 1]) * t.y[k - 1] / (t.y[k - 1] - t.y[k]));
    const double kb = K * u.ku / (std::sqrt(2.0) * g);
    const bool ok = zc.size() >= 2;
    close("undulator (planar): natural focusing, k_beta = K k_u / (sqrt(2) gamma)", ok ? M_PI / (zc[1] - zc[0]) : 0, kb, 2e-3);
}

// 21. Transport elements, one electron (gamma = 1000) at x0 = 1 k0^-1 (or y0): quadrupole (sqrt(k) L = 1, k =
//     G / p): x = x0 cos(sqrt(k) L), y = y0 cosh(sqrt(k) L); plasma lens: x = x0 cos(sqrt(k) L); solenoid with
//     fringes: the canonical angular momentum x p_y - y p_x - B(z) r^2 / 2 (electron) stays 0.
void test_magnets() {
    const double g = 1000, p = std::sqrt(g * g - 1), L = 2000, k = 1 / (L * L);
    Config c = ext_base(L / 4, L);
    Magnet m{};
    m.kind = 0; m.z0 = 0; m.z1 = L; m.g = k * p; m.R = 1e300; m.dsmax = 0.02 / std::sqrt(k);
    c.magnets = {m};
    Track tx = track(c, g, 1, 0, 300), ty = track(c, g, 0, 1, 300);
    close("quadrupole: x = x0 cos(sqrt(k) L)", tx.x.back(), std::cos(1.0), 1e-4);
    close("quadrupole: y = y0 cosh(sqrt(k) L)", ty.y.back(), std::cosh(1.0), 1e-4);
    c.magnets[0].kind = 1;
    tx = track(c, g, 1, 0, 300);
    close("plasma lens: x = x0 cos(sqrt(k) L)", tx.x.back(), std::cos(1.0), 1e-4);
    const double B = 2 * p / L;                            // Larmor angle B L / (2 p) = 1
    m.kind = 2; m.g = B; m.ramp = 0.1 * L; m.dsmax = 0.02 * L; m.z1 = 0.9 * L;
    c.magnets = {m};
    tx = track(c, g, 1, 0, 300);
    double lmax = 0;
    for (size_t i = 0; i < tx.z.size(); ++i) {
        const double z = tx.z[i];
        double f = 0;
        if (z > m.z0 && z < m.z1) f = std::min(1.0, std::min((z - m.z0) / m.ramp, (m.z1 - z) / m.ramp));
        const double r2 = tx.x[i] * tx.x[i] + tx.y[i] * tx.y[i];
        lmax = std::max(lmax, std::fabs(tx.x[i] * tx.py[i] - tx.y[i] * tx.px[i] - 0.5 * B * f * r2));
    }
    check(lmax < 1e-6 * 0.5 * B, "solenoid: canonical angular momentum conserved (relative to B r^2/2)", lmax / (0.5 * B), 0, 1e-6);
}
