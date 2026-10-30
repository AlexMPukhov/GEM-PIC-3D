// Tests: several bunches (bunch blocks), positrons, flat-top profile and transverse offset.
#include "test_util.hpp"

// 19. Several bunches. (a) An electron and a positron bunch drawn from the same seed are the same macroparticles
//     with opposite charge: their fields cancel (relative to the field of the electron bunch alone, after the
//     start and two steps). (b) Superposition: the fields of a frozen electron bunch and a frozen positron bunch
//     (offset in x and zeta, other gamma) together equal the sum of their fields alone. (c) Loading: a flat-top
//     bunch of length L has the rms length L / sqrt(12) and the mean x = x0, mean y = y0 of its offset.
static Config two_bunch_base() {
    Config c = base(32, 32, 121, 60, 60, 120);
    c.move_x = c.move_y = false; c.dz = 20; c.dz_push = 20; c.freeze_bunch = true;
    return c;
}
static Config::Bunch gauss_bunch(double q, double zc, double x0, double gamma, unsigned long seed) {
    Config::Bunch b;
    b.name = q > 0 ? "pos" : "ele"; b.q = q; b.charge = 1e-12; b.gamma = gamma;
    b.sigx = b.sigy = 4; b.sigz = 6; b.zeta_center = zc; b.x0 = x0; b.nmacro = 20000; b.seed = seed;
    return b;
}
static std::vector<float> two_bunch_fields(const std::vector<Config::Bunch>& list) {
    Config c = two_bunch_base();
    c.bunches = list;
    Simulation sim(c, comm);
    sim.init();
    for (int k = 0; k < 2; ++k) sim.step();
    return sim.F.host();
}
void test_bunches() {
    const Config::Bunch e = gauss_bunch(-1, 60, 0, 200, 7);
    Config::Bunch p = e; p.q = 1; p.name = "pos";
    const std::vector<float> fe = two_bunch_fields({e}), fep = two_bunch_fields({e, p});
    double m = 0, d = 0;
    for (size_t i = 0; i < fe.size(); ++i) { m = std::max(m, double(std::fabs(fe[i]))); d = std::max(d, double(std::fabs(fep[i]))); }
    check(m > 0 && d <= 1e-12 * m, "bunches: electron + positron bunch from the same seed: fields cancel", d / m, 0, 1e-12);
    const Config::Bunch b = gauss_bunch(1, 35, 8, 50, 3);
    const std::vector<float> fb = two_bunch_fields({b}), fs = two_bunch_fields({e, b});
    double ds = 0, ms = 0;
    for (size_t i = 0; i < fe.size(); ++i) {
        ds = std::max(ds, std::fabs(double(fs[i]) - double(fe[i]) - double(fb[i])));
        ms = std::max(ms, double(std::fabs(fs[i])));
    }
    check(ds <= 1e-6 * ms, "bunches: fields of two bunches = sum of the fields of each (float output)", ds / ms, 0, 1e-6);
    Config c = two_bunch_base();
    Config::Bunch f = gauss_bunch(-1, 60, 5, 1000, 11);
    f.sigz = 0; f.length = 40; f.y0 = -3; f.nmacro = 100000;
    c.bunches = {f};
    Simulation sim(c, comm);
    sim.init();
    double s0 = 0, s1 = 0, s2 = 0, sx = 0, sy = 0;
    for (const auto& q : sim.host_particles()) {
        if (q.alive != 1) continue;
        s0 += 1; s1 += q.zeta; s2 += q.zeta * q.zeta; sx += q.x + sim.X0; sy += q.y + sim.Y0;
    }
    const double rms = std::sqrt(s2 / s0 - (s1 / s0) * (s1 / s0));
    close("bunches: flat-top bunch, rms length = L / sqrt(12)", rms, 40 / std::sqrt(12.0), 0.01);
    check(std::fabs(sx / s0 - 5) < 0.03 && std::fabs(sy / s0 + 3) < 0.03, "bunches: transverse offset x0 = 5, y0 = -3 (mean x - x0)",
          sx / s0 - 5, 0, 0.03);
}
