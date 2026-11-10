// LASY laser files (laser type = file, src/laser_file.*). HDF5 build only (make HDF5=1); without HDF5 the test
// checks that reading a file is a clear error.
#include "test_util.hpp"
#include <cstdio>
#include <memory>

#if defined(GEMPIC_HDF5)
namespace {
const double W = 12, TAU = 25, ZC = 100, PA = 0.3, CEP = 0.4;   // built-in Gaussian of the comparison
double gt(double t, double tau) { return std::exp(-2 * std::log(2.0) * t * t / (tau * tau)); }

// cartesian file on t0 + l dt, x0 + i dx (y the same), envelope f(x, y, t)
template <class F> LaserFileData cart(int nt, double t0, double dt, int n, double x0, double dx, F f) {
    LaserFileData d;
    d.cartesian = true; d.omegaL = 1; d.px = std::cos(PA); d.py = std::sin(PA);
    d.nt = nt; d.t0 = t0; d.dt = dt; d.nx = d.ny = n; d.x0 = d.y0 = x0; d.dx = d.dy = dx;
    d.env.resize(size_t(nt) * n * n);
    for (int l = 0; l < nt; ++l)
        for (int j = 0; j < n; ++j)
            for (int i = 0; i < n; ++i) d.env[(size_t(l) * n + j) * n + i] = f(x0 + i * dx, x0 + j * dx, t0 + l * dt);
    return d;
}
// thetaMode file with components C_0, C_1 (cos), S_1 (sin) of f(r, t) -> (c0, c1, s1)
template <class F> LaserFileData theta(int nt, double t0, double dt, int nr, double dr, F f) {
    LaserFileData d;
    d.cartesian = false; d.omegaL = 1; d.px = std::cos(PA); d.py = std::sin(PA);
    d.nt = nt; d.t0 = t0; d.dt = dt; d.nr = nr; d.r0 = 0; d.dr = dr; d.ncomp = 3;
    d.env.assign(size_t(3) * nt * nr, 0.0);
    for (int l = 0; l < nt; ++l)
        for (int j = 0; j < nr; ++j) {
            cd c[3];
            f(j * dr, t0 + l * dt, c);
            for (int k = 0; k < 3; ++k) d.env[(size_t(k) * nt + l) * nr + j] = c[k];
        }
    return d;
}
Config grid() {
    Config c = base(32, 32, 161, 80, 80, 160);        // dx = 2.5, dzeta = 1: on the nodes of the files below
    c.move_x = c.move_y = false; c.dz = 20; c.dz_push = 20; c.z_end = 1e9;
    return c;
}
std::vector<float> run(const Config::GridLaser& g, int steps) {
    Config c = grid();
    c.glasers.push_back(g);
    Simulation sim(c, comm);
    sim.init();
    for (int k = 0; k < steps; ++k) sim.step();
    return sim.F.host();
}
double reldiff(const std::vector<float>& a, const std::vector<float>& b) {
    double d = 0, m = 0;
    for (size_t i = 0; i < a.size(); ++i) { d = std::max(d, double(std::fabs(a[i] - b[i]))); m = std::max(m, double(std::fabs(a[i]))); }
    return m > 0 ? d / m : 1e300;
}
Config::GridLaser from_file(const std::string& path, double file_z, double a0 = -1) {
    Config u = Config::base_units(10e-6);
    const LaserFileData fd = read_laser_file(path, "", -1, u.Lu, u.Tu);
    LaserFileField::Params p;
    p.zeta_center = ZC; p.t_center = 0; p.file_z = file_z;
    if (a0 > 0) p.scale = a0 * fd.omegaL / fd.peak;
    Config::GridLaser g;
    g.file = std::make_shared<LaserFileField>(fd, p);
    g.zeta_c = ZC;
    return g;
}
}  // namespace
#endif

// 28. LASY files. (a) a cartesian file of the built-in Gaussian (a0 = 1, w = 12, tau = 25, pol_angle 0.3, cep 0.4) at
//     its own plane, on a finer grid whose nodes contain the code's points: the fields at the start = built-in
//     (float rounding); the same as normalized_vector_potential (E = i omega a - da/dt, central differences); a0
//     rescaling. (b) thetaMode file (C_0) = built-in (radial interpolation, dr = dx/4); with an m = 1 part
//     (C_1, S_1) = the cartesian file of the same field. (c) a file at the focus, file_z = 3 z_R (w = 6, k w = 6):
//     the envelope at z = 0 against an independent exact reference, the Hankel transform per frequency
//     (quadrature in k and omega); the paraxial built-in at its z_focus is reported. (d) 20 steps in vacuum: the
//     fields of the file laser = built-in.
void test_lasy() {
#if !defined(GEMPIC_HDF5)
    bool err = false;
    try { read_laser_file("none.h5", "", -1, 1, 1); } catch (const std::runtime_error& e) { err = std::string(e.what()).find("HDF5") != std::string::npos; }
    check(err, "LASY: without HDF5 reading a laser file is a clear error (make HDF5=1 for the checks)", err, 1, 0);
#else
    Config u = Config::base_units(10e-6);
    const double amp = 1.0;                                  // a0 kL
    Config::GridLaser gb;
    gb.a0 = 1; gb.kL = 1; gb.waist = W; gb.tau = TAU; gb.zeta_c = ZC; gb.pol_angle = PA; gb.cep = CEP;
    const std::vector<float> ref = run(gb, 0);
    // file time tau = ZC - zeta; envelope amp g(tau) exp(-r^2/w^2) e^{i cep}
    auto gauss = [&](double x, double y, double t) { return cd(amp * gt(t, TAU) * std::exp(-(x * x + y * y) / (W * W))) * std::polar(1.0, CEP); };
    const char* f1 = "lasy_test_cart.h5";
    write_laser_file(f1, cart(401, -100, 0.5, 145, -45, 0.625, gauss), u.Lu, u.Tu);
    const double da = reldiff(ref, run(from_file(f1, 0), 0));
    check(da < 2e-6, "LASY (a): cartesian file at its plane = built-in Gaussian (fields at the start)", da, 0, 2e-6);
    {   // normalized vector potential a with E = i a - da/dt = the Gaussian above: a = E / i up to the derivative term
        LaserFileData d = cart(401, -100, 0.5, 145, -45, 0.625, [&](double x, double y, double t) {
            // a(t) = A g(t) with E = (i - g'/g) a: choose a = E / (i - g'/g), g'/g = -4 ln2 t / tau^2
            return gauss(x, y, t) / cd(4 * std::log(2.0) * t / (TAU * TAU), 1.0); });
        // E of this a is the Gaussian only up to d/dt of the factor 1 / (i + 4 ln2 t / tau^2): compare against the
        // E file of the same a, differentiated analytically
        d.field = "normalized_vector_potential";
        write_laser_file("lasy_test_a.h5", d, u.Lu, u.Tu);
        LaserFileData e = d;
        e.field = "electric_field";
        const double q = 4 * std::log(2.0) / (TAU * TAU);
        for (int l = 0; l < d.nt; ++l) {
            const double t = d.t0 + l * d.dt;
            // a = E0 g / (i + q t) => da/dt = E0 [g' (i + q t) - g q] / (i + q t)^2, g' = -q t g
            const cd den(q * t, 1.0), fac = (cd(-q * t) * den - q) / (den * den);   // (da/dt) / (E0 g)
            for (size_t k = 0; k < size_t(d.nx) * d.ny; ++k) {
                cd& v = e.env[size_t(l) * d.nx * d.ny + k];
                const cd E0g = v * den;                    // E0 g
                v = cd(0, 1) * v - E0g * fac;              // i a - da/dt
            }
        }
        write_laser_file("lasy_test_ae.h5", e, u.Lu, u.Tu);
        const double dv = reldiff(run(from_file("lasy_test_ae.h5", 0), 0), run(from_file("lasy_test_a.h5", 0), 0));
        check(dv < 1e-4, "LASY (a): normalized_vector_potential = electric_field of the same pulse", dv, 0, 1e-4);
        const std::vector<float> r2 = run(from_file(f1, 0, 2.0), 0);
        double m1 = 0, m2 = 0;
        for (size_t i = 0; i < ref.size(); ++i) { m1 = std::max(m1, double(std::fabs(ref[i]))); m2 = std::max(m2, double(std::fabs(r2[i]))); }
        close("LASY (a): a0 = 2 doubles the field of the a0 = 1 file", m2 / m1, 2.0, 1e-5);
        std::remove("lasy_test_a.h5"); std::remove("lasy_test_ae.h5");
    }
    // (b) thetaMode
    write_laser_file("lasy_test_th.h5", theta(401, -100, 0.5, 73, 0.625, [&](double r, double t, cd* c) {
        c[0] = gauss(r, 0, t); c[1] = c[2] = 0; }), u.Lu, u.Tu);
    const double db = reldiff(ref, run(from_file("lasy_test_th.h5", 0), 0));
    check(db < 3e-3, "LASY (b): thetaMode file (C_0) = built-in Gaussian (radial interpolation, dr = dx/4)", db, 0, 3e-3);
    auto m1f = [&](double x, double y, double t) { return gauss(x, y, t) * (1.0 + 0.5 * x / W + cd(0, 0.3) * y / W); };
    write_laser_file("lasy_test_th1.h5", theta(401, -100, 0.5, 73, 0.625, [&](double r, double t, cd* c) {
        c[0] = gauss(r, 0, t); c[1] = gauss(r, 0, t) * 0.5 * r / W; c[2] = gauss(r, 0, t) * cd(0, 0.3) * r / W; }), u.Lu, u.Tu);
    write_laser_file("lasy_test_c1.h5", cart(401, -100, 0.5, 145, -45, 0.625, m1f), u.Lu, u.Tu);
    const std::vector<float> c1 = run(from_file("lasy_test_c1.h5", 0), 0);
    const double dm = reldiff(c1, run(from_file("lasy_test_th1.h5", 0), 0));
    check(dm < 3e-3, "LASY (b): thetaMode file with m = 1 (cos, sin) = cartesian file of the same field", dm, 0, 3e-3);
    std::remove("lasy_test_th.h5"); std::remove("lasy_test_th1.h5"); std::remove("lasy_test_c1.h5");

    // (c) exact propagation: file at the focus, w = 6, file_z = 3 z_R; reference: Hankel transform per frequency
    {
        const double w = 6, zR = 0.5 * w * w, fz = 3 * zR, tau = 25;
        LaserFileData d = cart(151, -75, 1.0, 81, -40, 1.0, [&](double x, double y, double t) {
            return cd(gt(t, tau) * std::exp(-(x * x + y * y) / (w * w))); });
        d.px = 1; d.py = 0;
        write_laser_file("lasy_test_f.h5", d, u.Lu, u.Tu);
        Config uu = Config::base_units(10e-6);
        LaserFileField::Params p; p.file_z = fz; p.padding = 2; p.padding_xy = 2;
        const LaserFileField F(read_laser_file("lasy_test_f.h5", "", -1, uu.Lu, uu.Tu), p);
        std::remove("lasy_test_f.h5");
        const double sig = tau / std::sqrt(4 * std::log(2.0));          // g = exp(-t^2 / (2 sig^2))
        const std::vector<double> rs = {0, 3, 6, 10, 15, 20};
        const int NV = 241, NK = 3000;
        std::vector<cd> H(size_t(NV) * rs.size());
        std::vector<double> nu(NV), S(NV);
        for (int q = 0; q < NV; ++q) {
            nu[q] = -7 / sig + 14 / sig * q / (NV - 1);
            S[q] = sig / std::sqrt(2 * M_PI) * std::exp(-0.5 * sig * sig * nu[q] * nu[q]) * 14 / sig / (NV - 1);
            const double om = 1 + nu[q], kmax = std::min(om, 12 / w);
            for (size_t ir = 0; ir < rs.size(); ++ir) {
                cd s = 0;
                for (int j = 0; j < NK; ++j) {                    // midpoint rule in k on [0, kmax]
                    const double k = (j + 0.5) * kmax / NK;
                    s += k * std::cyl_bessel_j(0.0, k * rs[ir]) * std::exp(-k * k * w * w / 4) *
                         std::polar(1.0, (std::sqrt(om * om - k * k) - om) * (-fz));
                }
                H[q * rs.size() + ir] = s * (kmax / NK) * (w * w / 2);
            }
        }
        double dmax = 0, pk = 0, dpar = 0;
        const double zRp = 0.5 * w * w;
        const cd qb(1.0, -fz / zRp);
        for (size_t ir = 0; ir < rs.size(); ++ir)
            for (int t = -40; t <= 40; t += 2) {
                cd U = 0;
                for (int q = 0; q < NV; ++q) U += S[q] * std::polar(1.0, -nu[q] * t) * H[q * rs.size() + ir];
                const cd V = F.env(rs[ir], 0, t);            // the nodes of the propagated grid: no interpolation
                const cd P = gt(t, tau) * std::exp(-rs[ir] * rs[ir] / (w * w) / qb) / qb;   // paraxial, one frequency
                dmax = std::max(dmax, std::abs(V - U)); dpar = std::max(dpar, std::abs(P - U)); pk = std::max(pk, std::abs(U));
            }
        check(dmax < 1e-3 * pk, strf("LASY (c): file at the focus, file_z = 3 z_R (k w = 6): envelope at z = 0 vs Hankel per omega"), dmax / pk, 0, 1e-3);
        info("LASY (c): paraxial one-frequency Gaussian (built-in z_focus) vs the exact field", dpar / pk, "difference of the models, not checked");
    }
    // (d) 20 steps in vacuum
    const double dd = reldiff(run(gb, 20), run(from_file(f1, 0), 20));
    check(dd < 1e-5, "LASY (d): file laser = built-in after 20 steps in vacuum (fields)", dd, 0, 1e-5);
    std::remove(f1);
#endif
}
