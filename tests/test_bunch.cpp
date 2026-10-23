// Tests: one electron in the plane wave, self-fields of bunches (Gauss), equilibrium, sponge, tables.
#include "test_util.hpp"

// 5. One electron in the analytic plane wave, no self-fields: h = gamma - p.khat is conserved, the peak gamma
//    follows from h and p_perp + q A, and the electron leaves the pulse with its initial momentum.
void test_plane_wave(const std::string& pol) {
    Config c = base(4, 8, 4, 100, 200, 1e7);
    c.self_fields = false; c.laser_on = true; c.a0 = 5; c.kL = 1; c.theta = 0.1; c.pol = pol;
    c.tau = 150; c.start_fwhm = 3.5; c.stop_fwhm = 3.5; c.push_phase = 0.025;
    c.gamma = 50; c.charge = 1e-15; c.nmacro = 1; c.zeta_center = 0.5e7;
    c.dz = 30; c.dz_push = 15; c.z_end = 1e9;
    Simulation sim(c, comm);
    sim.init();
    auto hlf = [&](const Particle& p) { return gamma_of(p) - p.px * sim.laser.kx - p.pz * sim.laser.kz; };
    Particle p = sim.host_particles()[0];
    const double h0 = hlf(p), pz0 = p.pz, px0 = p.px;
    const double pp0 = px0 * sim.laser.kz - pz0 * sim.laser.kx;
    double dh = 0, gmax = 0;
    for (;;) {
        sim.step();
        p = sim.host_particles()[0];
        dh = std::max(dh, std::fabs(hlf(p) - h0));
        gmax = std::max(gmax, gamma_of(p));
        const double et = sim.laser.eta(p.x + sim.X0, sim.s, sim.s - (p.zeta - c.zeta_center));
        if (et - sim.laser.eta_c > c.stop_fwhm * c.tau) break;
    }
    const double a = c.a0;
    const double pmax2 = pol == "s" ? pp0 * pp0 + a * a : (std::fabs(pp0) + a) * (std::fabs(pp0) + a);
    const double gref = 0.5 * ((1 + pmax2) / h0 + h0);
    const std::string tag = "plane wave (" + pol + "-pol): ";
    check(p.alive == 1 && dh / h0 < 1e-5, tag + "h = gamma - p.khat conserved", dh / h0, 0, 1e-5);
    close(tag + "peak gamma from h and p_perp - A", gmax, gref, 2e-3);
    close(tag + "p_z after the pulse (Lawson-Woodward)", p.pz, pz0, 1e-5);
    check(std::fabs(p.px - px0) < 1e-4, tag + "p_x after the pulse", p.px, px0, 1e-4);
}

// 6. Field of a long round Gaussian bunch against Gauss's law, E_r = lambda (1 - exp(-r^2/2 sigma^2)) / (2 pi r):
//    E_x along x, E_y along y (half-nodes of the free-hand y), B_y; on the default grids, on a free-hand y grid
//    that grows inside the bunch, on a free-hand zeta grid with the transition at the measured slice, and with
//    120 x 120 modes.
void test_gauss(const std::string& kind) {
    const double sig = 6.0, sz = 60.0, Lw = 8 * sz;
    const int nm = kind == "120 x 120 modes" ? 120 : 128;
    Config c = base(nm, nm, 129, 40 * sig, 40 * sig, Lw);
    if (kind == "y") c.y_table({0, 4, 30}, {1.5, 1.5, 6}, 20 * sig);
    if (kind == "zeta") c.zeta_table({0, 200, 300}, {4, 4, 1.5}, Lw);
    c.gamma = 1000; c.sigx = c.sigy = sig; c.sigz = sz; c.charge = 1e-12; c.nmacro = 400000;
    c.zeta_center = 0.5 * Lw; c.dz = 10; c.k_filter = false;
    Simulation sim(c, comm);
    sim.init();
    double Qcode = 0; for (auto& p : sim.host_particles()) Qcode += -p.w;
    const double lam = Qcode / (std::sqrt(2 * M_PI) * sz);
    // fields and reference averaged over the five slices nearest to the centre (the macroparticle noise of the
    // backward field L scatters from slice to slice)
    const int n0 = nearest(c.zeta, c.zeta_center);
    double lam_n = 0;
    for (int n = n0 - 2; n <= n0 + 2; ++n) lam_n += 0.2 * lam * std::exp(-std::pow(c.zeta[n] - c.zeta_center, 2) / (2 * sz * sz));
    const std::vector<float> F = sim.F.host();
    auto at = [&](int comp, int i, int j) {
        double v = 0;
        for (int n = n0 - 2; n <= n0 + 2; ++n) v += 0.2 * field(F, sim, comp, n, size_t(j) * c.nx + i);
        return v;
    };
    // TSC deposition widens the bunch: sigma^2 + d^2/4 per direction (d = dx, local dy); the round-beam formula
    // takes the mean of both directions
    auto Er = [&](double r) {
        const int j = nearest(c.ynode, r);
        const double dyl = c.ynode[std::min(j + 1, c.ny)] - c.ynode[std::max(j - 1, 0)];
        const double s2 = sig * sig + (sim.dx * sim.dx + 0.25 * dyl * dyl) / 8;
        return lam_n * (1 - std::exp(-r * r / (2 * s2))) / (2 * M_PI * r);
    };
    const int jc = c.ny / 2, ic = c.nx / 2;
    std::vector<double> yh(c.ny);
    for (int j = 0; j < c.ny; ++j) yh[j] = c.y_fourier ? c.ynode[j] : 0.5 * (c.ynode[j] + c.ynode[j + 1]);   // E_y points
    const std::string tag = "Gauss" + std::string(kind == "uniform" ? "" : kind == "120 x 120 modes" ? " (" + kind + ")" : " (free-hand " + kind + ")") + ": ";
    for (double rr : {1.0, 2.0}) {
        const int k = int(std::lround(rr * sig / sim.dx));
        const double Ex = 0.5 * (at(0, ic + k, jc) - at(0, ic - k, jc));
        close(strf("%sE_x of a long bunch at x = %.0f sigma", tag.c_str(), rr), Ex, Er(k * sim.dx), 0.02);
        const int jp = nearest(yh, rr * sig), jm = c.y_fourier ? c.ny - jp : c.ny - 1 - jp;
        const double Ey = 0.5 * (at(1, ic, jp) - at(1, ic, jm));
        close(strf("%sE_y of a long bunch at y = %.0f sigma", tag.c_str(), rr), Ey, Er(yh[jp]), 0.02);
    }
    const int k = int(std::lround(sig / sim.dx));
    const double By = 0.5 * (at(4, ic + k, jc) - at(4, ic - k, jc));
    close(tag + "B_y of the bunch at x = 1 sigma", By, Er(k * sim.dx), 0.02);
}

// 7. A relativistic bunch in its own field (no laser) keeps its energy and its field.
void test_equilibrium(bool freehand) {
    const double sig = 6.0, sz = 6.0, Lw = 16 * sz;
    Config c = base(64, 64, 257, 30 * sig, 30 * sig, Lw);
    if (freehand) {
        c.y_table({0, 15, 40}, {1.5, 1.5, 6}, 15 * sig);
        c.zeta_table({0, 30, 45}, {1.0, 1.0, 0.375}, Lw);
    }
    c.gamma = 1000; c.sigx = c.sigy = sig; c.sigz = sz; c.charge = 10e-12; c.nmacro = 50000;
    c.zeta_center = 0.6 * Lw; c.dz = 50; c.dz_push = 50;
    Simulation sim(c, comm);
    sim.init();
    auto mean_g = [&]() { double s = 0; long n = 0;
        for (auto& p : sim.host_particles()) if (p.alive == 1) { s += gamma_of(p); ++n; }
        return s / n; };
    const double g0 = mean_g(), W0 = sim.total_field_energy();
    for (int k = 0; k < 100; ++k) sim.step();
    const double g1 = mean_g();
    const std::string tag = freehand ? "equilibrium (free-hand y and zeta): " : "equilibrium: ";
    check(std::fabs(g1 - g0) / g0 < 1e-6, tag + "<gamma> of a gamma = 1000 bunch over 100 steps", (g1 - g0) / g0, 0, 1e-6);
    close(tag + "field energy kept over 100 steps", sim.total_field_energy(), W0, 0.02);
}

// 8. The absorbing sponge must not drain the bunch through its own bound field: a dense gamma = 50 bunch
//    evolves the same with and without the sponge (damping all of R instead loses ~1e-3 of p_z per mm).
void test_sponge(bool freehand) {
    double pz[2];
    for (int k = 0; k < 2; ++k) {
        Config c = base(32, 32, 129, 100, 100, 64);
        if (freehand) c.y_table({0, 15, 30}, {2, 2, 6}, 90);
        c.gamma = 50; c.sigx = c.sigy = 6; c.sigz = 6; c.charge = 160e-12; c.nmacro = 20000;
        c.zeta_center = 40; c.dz = 30; c.dz_push = 30; c.sponge = k ? 15 : 0; c.sponge_y = k ? (freehand ? 30 : 15) : 0;
        c.sponge_rate = 0.1;
        Simulation sim(c, comm);
        sim.init();
        for (int st = 0; st < 40; ++st) sim.step();
        pz[k] = mean_pz(sim);
    }
    check(std::fabs(pz[1] - pz[0]) / pz[0] < 2e-6,
          freehand ? "sponge (free-hand y): no drag on a dense bunch" : "sponge: no drag on a dense bunch (with vs without)",
          (pz[1] - pz[0]) / pz[0], 0, 2e-6);
}

// 9. A zeta table with a constant step reproduces the uniform zeta grid exactly, with Fourier y (the default)
//    and with the free-hand y.
void test_constant_tables(bool yfourier) {
    std::vector<Particle> res[2];
    for (int k = 0; k < 2; ++k) {
        Config c = base(32, 32, 129, 100, 64, 64);
        if (!yfourier) c.y_table({0, 10, 20}, {2, 2, 4}, 40);
        if (k) c.zeta_table({0}, {0.5}, 64);
        c.gamma = 50; c.sigx = c.sigy = 6; c.sigz = 6; c.charge = 160e-12; c.nmacro = 4000;
        c.zeta_center = 40; c.dz = 30; c.dz_push = 30; c.sponge = 15; c.sponge_y = 10;
        c.laser_on = true; c.a0 = 5; c.theta = 0.1; c.tau = 100; c.start_fwhm = 1;
        Simulation sim(c, comm);
        sim.init();
        for (int st = 0; st < 10; ++st) sim.step();
        res[k] = sim.host_particles();
    }
    double d = res[0].size() == res[1].size() ? 0 : 1;
    for (size_t i = 0; i < res[0].size() && d == 0; ++i)
        d = std::max({d, std::fabs(res[0][i].pz - res[1][i].pz), std::fabs(res[0][i].y - res[1][i].y), std::fabs(res[0][i].zeta - res[1][i].zeta)});
    check(d < 1e-10, yfourier ? "grid: constant-step zeta table = uniform zeta (to rounding)"
                           : "grid: constant-step zeta table = uniform zeta, free-hand y (to rounding)", d, 0, 1e-10);
}
