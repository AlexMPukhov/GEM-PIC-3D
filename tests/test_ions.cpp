// Tests: mobile ions (immobile limit, neutral pairs at rest, ion collapse in a dense bunch).
#include "test_util.hpp"

// 14. Mobile ions, immobile limit: the 1D laser wake of test 10 with hydrogen-like ions of mass 1e15 u gives the
//     fields of the run without ions (to rounding).
void test_ions_immobile() {
    auto run = [](bool ions) {
        Config c = Config::base_units(10e-6);
        const double h = 2 * M_PI / 32;
        const int N = int(300 / h) + 1;
        c.nx = 4; c.Lx = 40; c.y_uniform(4, 40); c.zeta_uniform(N, h * (N - 1));
        c.sponge = c.sponge_y = 0; c.move_x = c.move_y = false;
        c.plasma_on = true; c.n0 = 0.01; c.ppc_x = 1; c.ppc_y = 1;
        c.nmacro = 0; c.dz = 25; c.dz_push = 25; c.zeta_center = 0.75 * c.Lwin;
        Config::GridLaser g; g.a0 = 0.3; g.kL = 1; g.waist = 1e7; g.tau = 2 * M_PI / std::sqrt(c.n0) * 0.4; g.zeta_c = 240;
        c.glasers.push_back(g);
        if (ions) { Config::IonSpecies s; s.mass = 1e15 * 1822.888486; c.ions.push_back(s); }
        Simulation sim(c, comm);
        sim.init();
        for (int k = 0; k < 3; ++k) sim.step();
        return sim.F.host();
    };
    const std::vector<float> a = run(false), b = run(true);
    double d = 0, m = 0;
    for (size_t i = 0; i < a.size(); ++i) { d = std::max(d, double(std::fabs(a[i] - b[i]))); m = std::max(m, double(std::fabs(a[i]))); }
    check(d <= 1e-6 * m, "ions: mass 1e15 u gives the fields without ions (max |dF| / max |F|)", d / m, 0, 1e-6);
}

// 15. Neutral pairs at rest: a parabolic channel with a z ramp, H+ (fraction 0.5) and He2+ (0.5), no driver: the
//     fields stay zero, no particle moves, and Z n_i of the ions equals n_e on the lines y = 0 and x = 0.
void test_ions_neutral() {
    Config c = Config::base_units(10e-6);
    c.nx = 16; c.Lx = 40; c.y_uniform(16, 40); c.zeta_uniform(41, 40);
    c.sponge = c.sponge_y = 0; c.move_x = c.move_y = false;
    c.plasma_on = true; c.n0 = 0.01; c.parabolic = true; c.channel_depth = 0.5; c.r_channel = 10; c.r_max = 15;
    c.pz_points = {0, 100}; c.pz_values = {0.2, 1};
    c.nmacro = 0; c.dz = 20; c.dz_push = 20;
    Config::IonSpecies h, he;
    h.atoms = 0.5; he.mass = 4.0026 * 1822.888486; he.Z0 = he.Zmax = 2; he.atoms = 0.25;
    c.ions = {h, he};
    Simulation sim(c, comm);
    sim.init();
    for (int k = 0; k < 2; ++k) sim.step();
    double fm = 0;
    for (float v : sim.F.host()) fm = std::max(fm, double(std::fabs(v)));
    double pm = 0; long ni = 0;
    std::vector<Particle> sp(sim.nstrm);
    sim.strm.to_host(sp.data(), 0, sp.size());
    for (const auto& p : sp) { pm = std::max({pm, std::fabs(p.px), std::fabs(p.py), std::fabs(p.pz)}); if (p.sp >= SP_ION) ++ni; }
    const std::vector<double> re = sim.rho_lines.host(), ri = sim.rho_lines_i.host();
    double dm = 0, em = 0;
    for (size_t k = 0; k < re.size(); ++k) { dm = std::max(dm, std::fabs(re[k] + ri[k])); em = std::max(em, std::fabs(re[k])); }
    check(fm == 0 && pm == 0 && ni > 0, "ions: neutral pairs at rest: max |F| + max |p| stays zero", fm + pm, 0, 0);
    check(em > 0 && dm <= 1e-12 * em, "ions: neutral pairs at rest: Z n_i = n_e on the lines (relative)", em > 0 ? dm / em : 1, 0, 1e-12);
}

// 16. Ion collapse in a dense, ultra-relativistic electron bunch (Rosenzweig et al., PRL 95, 195002 (2005); as the
//     check of GEM-PIC-RZ): rigid Gaussian bunch, n_b = 100 n0 (n0 = 0.002 n_c), sigma_r = 2.5, sigma_z = 80 k0^-1,
//     gamma = 1e5; hydrogen and deuterium; 320 x 320 modes over 80 k0^-1 (dx = sigma_r / 10), dzeta = 1, 3 steps
//     (make physics). Reference: a radial test-ion model with full blowout. Ion shells (initially uniform, at
//     rest) move in E_r = (Q_i(r) - Q_b(r)) / r, with Q_b(r) = n_b(zeta) s^2 (1 - exp(-r^2 / 2 s^2)) the bunch
//     charge inside r per 2 pi and Q_i(r) that of the ion shells inside r (Gauss; the ion self-field is
//     included), for the time t = zeta_front - zeta. The scheme deposits the bunch and gathers the field with TSC
//     (variance dx^2/4 each), so the model uses s^2 = sigma^2 + dx^2/2, and its on-axis density is the TSC average
//     of the shells at the axis node. Compared in the compression phase (model 3 ... 30 n0, from the front until
//     the model first exceeds 30); earlier the blowout is not yet complete at the head of the bunch, which the
//     model assumes, and the code focuses up to 4 % more. A coarser grid (dx = sigma_r / 5) or n_b = 20 n0, where
//     the compressed ions pull plasma electrons back to the axis, do not converge to the model.
struct IonCollapse { std::vector<double> zeta, ni, model; };
IonCollapse ion_collapse_run(double mass, int nx, double Lx, double hz, int steps, double dz, double n0) {
    Config c = Config::base_units(10e-6);
    const double sig = 2.5, sz = 80, nb = 0.2, Lw = 640, zc = 400;
    c.nx = nx; c.Lx = Lx; c.y_uniform(nx, Lx); c.zeta_uniform(int(Lw / hz) + 1, Lw);
    c.sponge = c.sponge_y = 0; c.move_x = c.move_y = false; c.k_filter = false;
    c.plasma_on = true; c.n0 = n0; c.ppc_x = 1; c.ppc_y = 1;
    const double Ne = nb * std::pow(2 * M_PI, 1.5) * sig * sig * sz;
    c.gamma = 1e5; c.sigx = c.sigy = sig; c.sigz = sz; c.nmacro = 400000; c.zeta_center = zc;
    c.charge = Ne * c.nc * std::pow(c.Lu, 3) * 1.602176634e-19;
    c.dz = dz; c.dz_push = dz; c.freeze_bunch = true;
    Config::IonSpecies s; s.mass = mass; c.ions.push_back(s);
    Simulation sim(c, comm);
    sim.init();
    for (int k = 0; k < steps; ++k) sim.step();
    IonCollapse r;
    const int L = c.nx + c.ny, i0 = c.nx / 2, N = c.nzeta;
    const std::vector<double> rl = sim.rho_lines_i.host();
    for (int m = 0; m + 1 < N; ++m) {                       // half-cell centres
        r.zeta.push_back(0.5 * (c.zeta[m] + c.zeta[m + 1]));
        r.ni.push_back(rl[size_t(m) * L + i0] / n0);
    }
    // test-ion model on the same zeta points
    const double dx = sim.dx, s2 = sig * sig + 0.5 * dx * dx, R = 6 * sig, dr0 = 0.01;
    const int K = int(R / dr0);
    std::vector<double> rr(K), vv(K), q(K);
    for (int k = 0; k < K; ++k) { rr[k] = (k + 0.5) * dr0; q[k] = n0 * rr[k] * dr0; }
    auto tsc1 = [&](double x) { const double a = std::fabs(x) / dx; return a < 0.5 ? 0.75 - a * a : a < 1.5 ? 0.5 * (1.5 - a) * (1.5 - a) : 0.0; };
    std::vector<int> idx(K);
    auto axis_density = [&]() {
        double n = 0;
        for (int k = 0; k < K; ++k) {
            const double a = std::fabs(rr[k]);
            if (a > 1.5 * std::sqrt(2.0) * dx) continue;
            double w = 0;
            for (int t = 0; t < 64; ++t) { const double th = 2 * M_PI * (t + 0.5) / 64; w += tsc1(a * std::cos(th)) * tsc1(a * std::sin(th)); }
            n += 2 * M_PI * q[k] * w / 64 / (dx * dx);
        }
        return n / n0;
    };
    const double dt = 0.05;
    double t = 0, zeta = c.Lwin;
    size_t out = r.zeta.size();                              // fill from the front
    r.model.assign(out, 0.0);
    while (out > 0) {
        // fields at zeta: ion charge inside each shell (sorted), bunch charge
        for (int k = 0; k < K; ++k) idx[k] = k;
        std::sort(idx.begin(), idx.end(), [&](int a, int b) { return std::fabs(rr[a]) < std::fabs(rr[b]); });
        const double nbz = nb * std::exp(-0.5 * (zeta - zc) * (zeta - zc) / (sz * sz));
        double Qi = 0;
        std::vector<double> E(K);
        for (int k : idx) {
            const double a = std::fabs(rr[k]);
            Qi += 0.5 * q[k];
            const double e = (Qi - nbz * s2 * (1 - std::exp(-a * a / (2 * s2)))) / std::max(a, 1e-9);
            E[k] = rr[k] < 0 ? -e : e;
            Qi += 0.5 * q[k];
        }
        while (out > 0 && r.zeta[out - 1] >= zeta - 0.5 * dt) r.model[--out] = axis_density();
        for (int k = 0; k < K; ++k) { vv[k] += E[k] / mass * dt; rr[k] += vv[k] * dt; }   // Z = 1
        t += dt; zeta -= dt;
    }
    return r;
}
void test_ion_collapse() {
    const double mH = 1.00784 * 1822.888486, mD = 2.01410 * 1822.888486;
    for (int sp = 0; sp < 2; ++sp) {
        const IonCollapse r = ion_collapse_run(sp ? mD : mH, 320, 80, 1.0, 3, 100, 0.002);
        double worst = 0, early = 0, peak = 0; int cnt = 0;
        bool compressed = false;
        for (size_t k = r.zeta.size(); k-- > 0;) {          // from the front of the window
            peak = std::max(peak, r.ni[k]);
            if (r.model[k] >= 30) compressed = true;
            if (compressed) continue;
            const double d = std::fabs(r.ni[k] / r.model[k] - 1);
            if (r.model[k] > 3) { worst = std::max(worst, d); ++cnt; }
            else if (r.model[k] > 1.5) early = std::max(early, d);
        }
        const char* nm = sp ? "D" : "H";
        check(cnt > 10 && worst < 0.03, strf("ions: collapse in a dense bunch (%s), on-axis Z n_i/n0 = test-ion model", nm), worst, 0, 0.03);
        info(strf("ions: collapse (%s), deviation before the compression (model 1.5 ... 3)", nm), early, "blowout forming");
        info(strf("ions: collapse (%s), peak on-axis Z n_i/n0", nm), peak, "in the window");
    }
}
