// Regression checks taken from the lessons of GEM-PIC-RZ (its "Corrections of earlier versions"): each of them
// would have caught one of the expensive errors there early.
#include "test_util.hpp"

// 29. A static bunch (gamma = 1e6, rigid) in vacuum must be a steady state of the scheme from the start.
//     Measured, in units of the largest transverse field: E_z and the backward field E_x - B_y at the start
//     (both are zero for a v = c bunch in the continuum), and the change of all fields in the first and in the
//     second step. Without numerics.start_relax the co-moving field of the continuum equations is not the steady
//     state of the discrete scheme, and the stiff modes keep the mismatch with alternating sign (odd-even in s).
struct StaticBunch { double ez, back, d1, d2, d20; };
static StaticBunch static_bunch(bool relax, int nz, double sz) {
    const double sig = 6.0, Lw = 96;
    Config c = base(64, 64, nz, 30 * sig, 30 * sig, Lw);
    c.gamma = 1e6; c.sigx = c.sigy = sig; c.sigz = sz; c.charge = 10e-12; c.nmacro = 100000;
    c.zeta_center = 0.5 * Lw; c.dz = 50; c.dz_push = 50; c.freeze_bunch = true; c.start_relax = relax;
    Simulation sim(c, comm);
    sim.init();
    const std::vector<float> F0 = sim.F.host();
    sim.step();
    const std::vector<float> F1 = sim.F.host();
    sim.step();
    const std::vector<float> F2 = sim.F.host();
    StaticBunch r{0, 0, 0, 0, 0};
    double m = 0;
    for (int n = 0; n < c.nzeta; ++n)
        for (size_t i = 0; i < size_t(sim.nk); ++i) {
            m = std::max(m, std::fabs(field(F0, sim, 0, n, i)));
            r.ez = std::max(r.ez, std::fabs(field(F0, sim, 2, n, i)));
            r.back = std::max(r.back, std::fabs(field(F0, sim, 0, n, i) - field(F0, sim, 4, n, i)));
            for (int comp = 0; comp < 6; ++comp) {
                r.d1 = std::max(r.d1, std::fabs(field(F1, sim, comp, n, i) - field(F0, sim, comp, n, i)));
                r.d2 = std::max(r.d2, std::fabs(field(F2, sim, comp, n, i) - field(F1, sim, comp, n, i)));
                r.d20 = std::max(r.d20, std::fabs(field(F2, sim, comp, n, i) - field(F0, sim, comp, n, i)));
            }
        }
    r.ez /= m; r.back /= m; r.d1 /= m; r.d2 /= m; r.d20 /= m;
    return r;
}
void test_static_bunch() {
    for (int fine = 0; fine < 2; ++fine) {
        const int nz = fine ? 257 : 129;
        const std::string tag = strf("static bunch, gamma = 1e6, sigma_z = %s h: ", fine ? "16" : "8");
        const StaticBunch a = static_bunch(false, nz, 6.0), b = static_bunch(true, nz, 6.0);
        info(tag + "without start_relax: change of the fields in step 1", a.d1, "relative to the largest E_x");
        info(tag + "without start_relax: change in step 2", a.d2, "the sign alternates");
        info(tag + "without start_relax: steps 0 -> 2", a.d20, "odd-even mode in s");
        info(tag + "E_z at the start", b.ez, "0 in the continuum for v = c");
        info(tag + "E_x - B_y at the start", b.back, "0 in the continuum for v = c");
        check(b.d1 < 5e-5, tag + "steady from the start (step 1)", b.d1, 0, 5e-5);
        check(b.d2 < 5e-5, tag + "steady from the start (step 2)", b.d2, 0, 5e-5);
    }
    // a short bunch (sigma_z = 2 h): the sharp-edge case of the static j_z mismatch
    const StaticBunch s = static_bunch(true, 129, 1.5);
    info("static bunch, sigma_z = 2 h: E_z at the start", s.ez, "relative to the largest E_x");
    check(s.d1 < 5e-5, "static bunch, sigma_z = 2 h: steady from the start (step 1)", s.d1, 0, 5e-5);
}

// 30. An intense beam in a drift: 4 kA, gamma = 1000, sigma_r = 10 um, in its own field. The transverse force on
//     an electron is q (E_r - beta B_theta) = q E_r / gamma^2, a millionth of the two terms: the push must keep
//     the cancellation at its largest sub-step (GEM-PIC-RZ: Boris with large sub-steps gave 17 times too much).
//     Reference: E_r of a long round Gaussian bunch from Gauss's law with the local line density and the TSC
//     widening of the deposit and of the gather. Measured: the radial momentum gained in one step by the
//     electrons with 0.5 < r / sigma < 2 near the centre of the bunch, summed, against the same sum of
//     |E_r| Delta z / gamma^2.
void test_beam_drift(double dz) {
    const double sig = 2 * M_PI, sz = 60.0, Lw = 8 * sz, gam = 1000;           // lambda0 = 10 um: sigma = 10 um
    Config c = base(128, 128, 129, 40 * sig, 40 * sig, Lw);
    const double Ipeak = 4000.0;                                                 // A
    c.gamma = gam; c.sigx = c.sigy = sig; c.sigz = sz; c.nmacro = 400000;
    c.charge = Ipeak * std::sqrt(2 * M_PI) * sz * c.Lu / 299792458.0;
    c.zeta_center = 0.5 * Lw; c.dz = dz; c.dz_push = dz; c.k_filter = false;
    Simulation sim(c, comm);
    sim.init();
    const std::vector<Particle> p0 = sim.host_particles();
    sim.step();
    const std::vector<Particle> p1 = sim.host_particles();
    double Qcode = 0; for (const auto& p : p0) Qcode += p.w;
    const double lam = Qcode / (std::sqrt(2 * M_PI) * sz);
    const double s2 = sig * sig + sim.dx * sim.dx / 2;                           // TSC twice: deposit and gather
    double got = 0, ref = 0; long n = 0;
    const bool same = p0.size() == p1.size();
    for (size_t k = 0; same && k < p0.size(); ++k) {
        const Particle& a = p0[k];
        if (a.alive != 1 || p1[k].alive != 1) continue;
        const double x = a.x + sim.X0, y = a.y + sim.Y0, r = std::hypot(x, y);
        if (r < 0.5 * sig || r > 2 * sig || std::fabs(a.zeta - c.zeta_center) > 0.5 * sz) continue;
        const double l = lam * std::exp(-std::pow(a.zeta - c.zeta_center, 2) / (2 * sz * sz));
        const double Er = l * (1 - std::exp(-r * r / (2 * s2))) / (2 * M_PI * r);
        got += ((p1[k].px - a.px) * x + (p1[k].py - a.py) * y) / r;
        ref += Er * dz / (gam * gam);
        ++n;
    }
    check(same && n > 10000, strf("beam in a drift, dz_push = %g / k0: particles kept and counted", dz), double(n), 0, 0);
    close(strf("beam in a drift (4 kA, gamma 1000), dz_push = %g / k0: radial force = E_r / gamma^2", dz), got, ref, 0.03);
}

// 31. Long linear wake with large steps (GEM-PIC-RZ: tests/long_wake.sh, the check of the predictor of the plasma
//     current along zeta): a weak rigid bunch (k_p sigma_r = k_p sigma_z = 1, n_b = 1e-3 n0), window 60 c/omega_p,
//     8 steps of 100 c/omega_p. An error of the predicted current of the half-cell behind a slice drives the
//     plasma resonantly along zeta and grows from step to step. The k_p component of E_z on the axis, fitted over
//     5 ... 45 c/omega_p behind the bunch, must equal that of the linear theory of the scheme (the mode sum of
//     the 3D beam-wake check, fitted in the same way) at step 8 and must not change from step 6 to step 8;
//     with numerics.predictor = previous it grows.
static double kp_amplitude(const std::vector<double>& ez, const Config& c, double kp, double z0, double z1) {
    double cc = 0, ss = 0, cs = 0, ce = 0, se = 0;
    for (int n = 0; n < c.nzeta; ++n) {
        const double z = c.zeta[n];
        if (z < z0 || z > z1) continue;
        const double co = std::cos(kp * z), si = std::sin(kp * z);
        cc += co * co; ss += si * si; cs += co * si; ce += co * ez[n]; se += si * ez[n];
    }
    const double d = cc * ss - cs * cs, a = (ce * ss - se * cs) / d, b = (se * cc - ce * cs) / d;
    return std::sqrt(a * a + b * b);
}
void test_long_wake(bool extrap, int passes) {
    Config c = Config::base_units(10e-6);
    const double kp = 0.1, sig = 1 / kp, sz = 1 / kp, h = 0.0625 / kp, Lw = 60 / kp;       // the grid of the RZ test
    c.nx = 48; c.Lx = 24 / kp; c.y_uniform(48, 24 / kp);
    c.zeta_uniform(int(Lw / h) + 1, Lw);
    c.sponge = c.sponge_y = 0; c.move_x = c.move_y = false; c.k_filter = false;
    c.plasma_on = true; c.n0 = kp * kp; c.ppc_x = 1; c.ppc_y = 1; c.predictor_extrapolate = extrap; c.iterations = passes;
    const double Ne = 1e-3 * c.n0 * std::pow(2 * M_PI, 1.5) * sig * sig * sz;
    c.gamma = 1000; c.sigx = c.sigy = sig; c.sigz = sz; c.nmacro = 200000; c.zeta_center = Lw - 5 * sz;
    c.charge = Ne * c.nc * std::pow(c.Lu, 3) * 1.602176634e-19;
    c.dz = 100 / kp; c.dz_push = c.dz; c.freeze_bunch = true;
    Simulation sim(c, comm);
    sim.init();
    const int ic = c.nx / 2, jc = c.ny / 2;
    auto axis = [&]() {
        const std::vector<float> F = sim.F.host();
        std::vector<double> ez(c.nzeta);
        for (int n = 0; n < c.nzeta; ++n) ez[n] = field(F, sim, 2, n, size_t(jc) * c.nx + ic);
        return ez;
    };
    const double z0 = c.zeta_center - 45 / kp, z1 = c.zeta_center - 5 / kp;
    double amp[9];
    for (int k = 1; k <= 8; ++k) { sim.step(); amp[k] = kp_amplitude(axis(), c, kp, z0, z1); }
    // the linear theory of the scheme, as in the 3D beam-wake check
    auto sinc = [](double x) { return x == 0 ? 1.0 : std::sin(x) / x; };
    const double lam0 = Ne / (std::sqrt(2 * M_PI) * sz), dx = sim.dx, dy = sim.dy;
    std::vector<double> E(c.nzeta, 0.0);
    for (int m = -c.nx / 2 + 1; m < c.nx / 2; ++m)
        for (int l = -c.ny / 2 + 1; l < c.ny / 2; ++l) {
            const double kx = 2 * M_PI * m / c.Lx, ky = 2 * M_PI * l / c.Ly, k2 = kx * kx + ky * ky;
            const double S = std::pow(sinc(0.5 * kx * dx) * sinc(0.5 * ky * dy), 3), K = kp * S;
            const double A = std::exp(-0.5 * k2 * sig * sig) * S * K * K / (k2 + K * K) / (c.Lx * c.Ly);
            double Cc = 0, Ss = 0;
            for (double zp = c.zeta_center - 7 * sz; zp < c.zeta_center + 7 * sz; zp += 0.05) {
                const double g = lam0 * std::exp(-std::pow(zp - c.zeta_center, 2) / (2 * sz * sz)) * 0.05;
                Cc += std::cos(K * zp) * g; Ss += std::sin(K * zp) * g;
            }
            for (int n = 0; n < c.nzeta; ++n) E[n] += A * (std::cos(K * c.zeta[n]) * Cc + std::sin(K * c.zeta[n]) * Ss);
        }
    const double th = kp_amplitude(E, c, kp, z0, z1);
    const std::string tag = strf("long wake (8 steps of 100 c/omega_p, %d passes%s): ", passes, extrap ? "" : ", predictor = previous");
    if (!extrap) {
        check(amp[8] > 3 * th, tag + "the wake grows (amplitude at step 8 / theory)", amp[8] / th, 0, 0);
        return;
    }
    close(tag + "k_p amplitude of E_z at step 8 = linear theory of the scheme", amp[8], th, 0.03);
    close(tag + "amplitude at step 8 = step 6", amp[8], amp[6], 0.03);
    info(tag + "amplitude at step 1 / theory", amp[1] / th, "the start-up transient: never read the first steps");
}

// 32. Plasma on one side of the beam (plasma.density_function of x and y: no axis of symmetry). A bunch
//     (n_b = 0.5 n0, k_p sigma_r = 0.5, gamma = 1000) passes a round plasma column at 1.25 c/omega_p from its
//     surface, 6 steps of 10 c/omega_p. The column on the other side is the mirror image x -> -x of the same
//     problem: the mean transverse momentum gained by the bunch must be equal and opposite (the grid, the plasma
//     loading and the solver have no preferred side; the random bunch is the same in both runs and is not
//     mirrored, which limits the agreement), it must point to the plasma (the bunch expels the plasma electrons
//     of the surface, the ions left behind attract it), and <p_y> must stay zero.
struct SideKick { double px, py; };
static SideKick one_sided(double side) {
    Config c = Config::base_units(10e-6);
    const double kp = 0.1, sig = 0.5 / kp, sz = 1.2 / kp, h = 0.15 / kp, Lw = 12 / kp, L = 19.2 / kp;
    c.nx = 64; c.Lx = L; c.y_uniform(64, L);
    c.zeta_uniform(int(Lw / h) + 1, Lw);
    c.sponge = c.sponge_y = 0; c.move_x = c.move_y = false;
    c.plasma_on = true; c.n0 = kp * kp; c.ppc_x = 1; c.ppc_y = 1;
    Deck d;                                              // only for the units of the expression (code units here)
    d.set_code_units(c.Lu, c.Tu, c.nc);
    d.read_constants();
    c.density_fn = Expr(strf("min(max((35 - sqrt((x - (%g))^2 + y^2)) / 3 + 0.5, 0), 1)", side * 47.5), {"x", "y", "z", "r"}, {},
                        d.unit_fn());
    const double Ne = 0.5 * c.n0 * std::pow(2 * M_PI, 1.5) * sig * sig * sz;
    c.gamma = 1000; c.sigx = c.sigy = sig; c.sigz = sz; c.nmacro = 200000; c.zeta_center = Lw - 4 * sz;
    c.charge = Ne * c.nc * std::pow(c.Lu, 3) * 1.602176634e-19;
    c.dz = 10 / kp; c.dz_push = 1 / kp;
    Simulation sim(c, comm);
    sim.init();
    for (int k = 0; k < 6; ++k) sim.step();
    SideKick r{0, 0};
    double w = 0;
    for (const auto& p : sim.host_particles())
        if (p.alive == 1) { r.px += p.w * p.px; r.py += p.w * p.py; w += p.w; }
    r.px /= w; r.py /= w;
    return r;
}
void test_one_sided() {
    const SideKick a = one_sided(1), b = one_sided(-1);
    const std::string tag = "plasma on one side of the bunch (density_function(x, y)): ";
    check(a.px > 0 && b.px < 0, tag + "the bunch is pulled to the plasma, <p_x>", a.px, 0, 0);
    close(tag + "mirrored plasma gives the mirrored kick, -<p_x>", -b.px, a.px, 0.02);
    check(std::fabs(a.py) < 0.01 * std::fabs(a.px), tag + "<p_y> / <p_x> stays zero", a.py / a.px, 0, 0.01);
}
