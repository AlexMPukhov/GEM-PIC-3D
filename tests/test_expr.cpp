// Tests: expressions of the deck (parser, constants, units), profiles and tapers given as functions.
#include "test_util.hpp"

// a deck with the code units of lambda0 = 10 um and the constants w0 = 12 um, L = 2 w0
static Deck expr_deck(const Config& c) {
    Deck d;
    d.set("constants", "w0", "12 um", 1);
    d.set("constants", "L", "\"2 * w0\"", 2);
    d.set_code_units(c.Lu, c.Tu, c.nc);
    d.read_constants();
    return d;
}

// 22. Expressions: values against direct evaluation (units in code units, constants, precedence, functions),
//     and a value of the deck given as an expression of the constants.
void test_expr_parser() {
    const Config c = Config::base_units(10e-6);
    Deck d = expr_deck(c);
    const double um = 1e-6 / c.Lu;
    struct Case { const char* t; double v; };
    const double x = 0.7, y = -1.3;
    const Case cases[] = {
        {"-x^2", -x * x}, {"2^3^2", 512}, {"x ** 2 * y", x * x * y}, {"(x + y) / 2 - y", (x + y) / 2 - y},
        {"x < y ? 3 : 4", 4}, {"if(x > 0 && y < 0, 1, 2)", 1}, {"!(x > 0) || y > 0", 0},
        {"gauss(x, 2) + rect(y, 3) + step(-1e-300)", std::exp(-x * x / 8) + 1}, {"atan2(y, x) + min(x, y) + max(x, y)", std::atan2(y, x) + x + y},
        {"sqrt(abs(y)) + exp(x) + ln(2) + log10(1000) + floor(y) + ceil(y) + sign(y)", std::sqrt(1.3) + std::exp(x) + std::log(2.0) + 3 - 2 - 1 - 1},
        {"pow(x, 3) + sinh(x) + cosh(y) + tanh(x)", x * x * x + std::sinh(x) + std::cosh(y) + std::tanh(x)},
        {"10 um + 2 * w0", 34 * um}, {"L / (1 k0^-1)", 24 * um}, {"3e18 cm^-3 / (1 n_c)", 3e24 / c.nc}, {"50% * pi", 0.5 * M_PI},
        {"5 fs", 5e-15 / c.Tu}};
    double worst = 0;
    for (const Case& k : cases) {
        const Expr e(k.t, {"x", "y"}, {{"w0", 12 * um}, {"l", 24 * um}}, d.unit_fn());
        worst = std::max(worst, std::fabs(e({x, y}) - k.v) / std::max(1.0, std::fabs(k.v)));
    }
    check(worst < 1e-14, "expressions: operators, functions, units, constants against direct values", worst, 0, 1e-14);
    d.set("grid", "Lx", "\"4 * w0 + 1 um\"", 3);
    close("expressions: a deck value given as an expression of the constants", d.num("grid", "Lx", Kind::Length), 49e-6, 1e-14);
    d.set("grid", "dz", "8 fs", 4);
    bool err = false;
    try { d.num("grid", "dz", Kind::Length); } catch (const std::runtime_error&) { err = true; }
    check(err, "expressions: a number with a unit of the wrong kind stays an error", err, 1, 0);
}

// 23. Profiles and tapers as functions. (a) plasma density_function "1 + 0.5 (r / 10 k0^-1)^2" against the built-in
//     parabolic channel (channel_depth 0.5, r_channel 10): fields of 3 steps with a laser wake (to rounding).
//     (b) undulator K_function = "1" against K = 1 (orbit to rounding); a linear taper K(z) = 0.6 + 0.4 z / L:
//     the slippage between 3 and 9 periods against the integral of <(gamma - p_z)/p_z> at the local K.
static std::vector<float> channel_run(bool fn) {
    Config c = Config::base_units(10e-6);
    const double h = 2 * M_PI / 16;
    const int N = int(120 / h) + 1;
    c.nx = 16; c.Lx = 60; c.y_uniform(16, 60); c.zeta_uniform(N, h * (N - 1));
    c.sponge = c.sponge_y = 0; c.move_x = c.move_y = false;
    c.plasma_on = true; c.n0 = 0.01; c.ppc_x = 1; c.ppc_y = 1;
    c.nmacro = 0; c.dz = 25; c.dz_push = 25; c.zeta_center = 0.75 * c.Lwin;
    Config::GridLaser g; g.a0 = 0.5; g.kL = 1; g.waist = 12; g.tau = 25; g.zeta_c = 90;
    c.glasers.push_back(g);
    if (fn) c.density_fn = Expr("1 + 0.5 * (r / (10 k0^-1))^2", {"x", "y", "z", "r"}, {}, expr_deck(c).unit_fn());
    else { c.parabolic = true; c.channel_depth = 0.5; c.r_channel = 10; }
    Simulation sim(c, comm);
    sim.init();
    for (int k = 0; k < 3; ++k) sim.step();
    return sim.F.host();
}
void test_expr_profiles() {
    const std::vector<float> a = channel_run(false), b = channel_run(true);
    double d = 0, m = 0;
    for (size_t i = 0; i < a.size(); ++i) { d = std::max(d, double(std::fabs(a[i] - b[i]))); m = std::max(m, double(std::fabs(a[i]))); }
    check(m > 0 && d <= 1e-5 * m, "expressions: density_function = built-in parabolic channel (fields, float)", d / m, 0, 1e-5);
    // undulator: constant K_function against K, and a linear taper
    const double lu = 200, g = 100, L = 12 * lu;
    auto run = [&](const char* kf) {
        Config c = base(4, 4, 8, 200, 200, 400);
        c.move_x = c.move_y = false; c.dz = lu / 8; c.dz_push = lu / 8; c.z_end = 13 * lu; c.self_fields = false;
        Deck d = expr_deck(c);
        d.set("undulator", "periods", "12", 2);
        d.set("undulator", kf ? "K_function" : "K", kf ? kf : "1", 3);
        d.set("undulator", "period", std::to_string(lu * c.Lu) + " m", 1);
        c.parse_external(d);
        Config::Bunch bb;
        bb.charge = 1e-18; bb.gamma = g; bb.nmacro = 1; bb.zeta_center = 300;
        c.bunches = {bb};
        Simulation sim(c, comm);
        sim.init();
        std::vector<double> zeta, px;
        while (!sim.finished()) {
            sim.step();
            const Particle p = sim.host_particles()[0];
            zeta.push_back(p.zeta); px.push_back(p.px);
        }
        return std::make_pair(zeta, px);
    };
    const auto k1 = run(nullptr), kf = run("1 + 0 * z");
    double dk = 0;
    for (size_t i = 0; i < k1.first.size(); ++i) dk = std::max(dk, std::fabs(k1.second[i] - kf.second[i]));
    check(dk < 1e-12, "expressions: undulator K_function = 1 gives the orbit of K = 1", dk, 0, 1e-12);
    const auto tp = run("\"0.6 + 0.4 * z / (2400 k0^-1)\"");
    // steps of lu/8: index i is z = (i + 1) lu / 8; 3 and 9 periods
    const double slip = tp.first[3 * 8 - 1] - tp.first[9 * 8 - 1];
    double ref = 0;
    const int M = 400;
    for (int j = 0; j < 6 * M; ++j) {                        // z from 3 lu to 9 lu
        const double z = 3 * lu + (j + 0.5) * lu / M, K = 0.6 + 0.4 * z / L;
        double s = 0;
        for (int q = 0; q < 64; ++q) {
            const double ux = K * std::sin(2 * M_PI * (q + 0.5) / 64), pz = std::sqrt(g * g - 1 - ux * ux);
            s += (g - pz) / pz / 64;
        }
        ref += s * lu / M;
    }
    close("expressions: tapered undulator K(z), slippage = integral at the local K", slip, ref, 1e-3);
}

// 26. Functions of lasers and bunches. (a) grid laser with profile_function exp(-(r/w0)^2) and envelope_function
//     exp(-2 ln2 (zeta - zc)^2 / tau^2) against the built-in Gaussian focused at z = 0: the fields at the start.
//     (b) bunch: profile_z = rect(zeta - 60, 20) on [45, 75]: rms length 20 / sqrt(12); gamma_function = 200 + 2
//     (zeta - 60): each particle; profile_r = gauss(r - 5, 0.5) in r < 10: the mean radius (5 + 0.5^2/5); ux, uy
//     functions of a rigid rotation (-0.5 y, 0.5 x): each particle (at t = 0), with |p| = sqrt(gamma^2 - 1) kept.
static std::vector<float> laser_start(bool fn) {
    Config c = base(32, 32, 161, 80, 80, 160);
    c.move_x = c.move_y = false; c.dz = 20; c.dz_push = 20;
    Config::GridLaser g; g.a0 = 1; g.kL = 1; g.waist = 12; g.tau = 25; g.zeta_c = 100;
    if (fn) {
        const Deck d = expr_deck(c);
        g.profile = Expr("exp(-(r / (12 k0^-1))^2)", {"r"}, {}, d.unit_fn());
        g.envelope = Expr("exp(-2 * ln(2) * ((zeta - 100 k0^-1) / (25 k0^-1))^2)", {"zeta"}, {}, d.unit_fn());
    }
    c.glasers.push_back(g);
    Simulation sim(c, comm);
    sim.init();
    return sim.F.host();
}
void test_expr_functions() {
    const std::vector<float> a = laser_start(false), b = laser_start(true);
    double d = 0, m = 0;
    for (size_t i = 0; i < a.size(); ++i) { d = std::max(d, double(std::fabs(a[i] - b[i]))); m = std::max(m, double(std::fabs(a[i]))); }
    check(m > 0 && d <= 1e-6 * m, "functions: grid laser profile_function, envelope_function = built-in Gaussian", d / m, 0, 1e-6);
    Config c = base(32, 32, 121, 60, 60, 120);
    c.move_x = c.move_y = false; c.dz = 20; c.dz_push = 20; c.self_fields = false;
    const Deck dk = expr_deck(c);
    const Expr::UnitFn uf = dk.unit_fn();
    Config::Bunch bz;
    bz.charge = 1e-12; bz.nmacro = 100000; bz.sigx = bz.sigy = 3;
    bz.profile_z = Expr("rect(zeta - 60, 20)", {"zeta"}, {}, uf); bz.zeta_min = 45; bz.zeta_max = 75; bz.zeta_center = 60;
    bz.gamma_fn = Expr("200 + 2 * (zeta - 60)", {"zeta"}, {}, uf); bz.gamma = 200;
    Config::Bunch br = bz;
    br.name = "ring"; br.seed = 2; br.profile_z = Expr(); br.sigz = 5; br.gamma_fn = Expr(); br.gamma = 50;
    br.profile_r = Expr("gauss(r - 5, 0.5)", {"x", "y", "r", "zeta"}, {}, uf); br.r_extent = 10;
    br.ux_fn = Expr("-0.5 * y", {"x", "y", "r", "zeta"}, {}, uf); br.uy_fn = Expr("0.5 * x", {"x", "y", "r", "zeta"}, {}, uf);
    c.bunches = {bz, br};
    Simulation sim(c, comm);
    sim.init();
    double s0 = 0, s1 = 0, s2 = 0, gd = 0, r0 = 0, rs = 0, ud = 0;
    for (const auto& p : sim.host_particles()) {
        const double g = std::sqrt(1 + p.px * p.px + p.py * p.py + p.pz * p.pz);
        if (p.sp == 0) { s0 += 1; s1 += p.zeta; s2 += p.zeta * p.zeta; gd = std::max(gd, std::fabs(g - (200 + 2 * (p.zeta - 60)))); }
        else {                                     // (x, y) at t = 0: the loading drifts them to the plane z = 0
            const double x = p.x + sim.X0 + p.px * (p.zeta - 60) / g, y = p.y + sim.Y0 + p.py * (p.zeta - 60) / g;
            r0 += 1; rs += std::hypot(x, y);
            ud = std::max({ud, std::fabs(p.px + 0.5 * y), std::fabs(p.py - 0.5 * x), std::fabs(g - 50)});
        }
    }
    close("functions: bunch profile_z = rect: rms length L / sqrt(12)", std::sqrt(s2 / s0 - s1 * s1 / (s0 * s0)), 20 / std::sqrt(12.0), 0.01);
    check(gd < 1e-6, "functions: bunch gamma_function at each particle", gd, 0, 1e-6);
    double m0 = 0, m1 = 0;                         // <r> of the density gauss(r - 5, 0.5) in the plane (weight r dr)
    for (int k = 0; k < 10000; ++k) { const double r = 10.0 * (k + 0.5) / 10000, f = std::exp(-0.5 * std::pow((r - 5) / 0.5, 2)) * r; m0 += f; m1 += f * r; }
    close("functions: bunch profile_r = ring: mean radius", rs / r0, m1 / m0, 0.003);
    check(ud < 1e-9, "functions: ux, uy functions (rigid rotation), energy kept", ud, 0, 1e-9);
}
