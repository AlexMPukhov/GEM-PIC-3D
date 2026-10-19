// Tests of GEM-PIC-3D against independent references.
#include <cmath>
#include <cstdio>
#include <random>
#include "sim.hpp"

static int failures = 0;
static void check(bool ok, const char* name, double val, double ref, double tol) {
    std::printf("%-58s %s  value %.6g  reference %.6g  tol %.2g\n", name, ok ? "PASS" : "FAIL", val, ref, tol);
    if (!ok) ++failures;
}
static void close(const char* name, double val, double ref, double rtol) {
    check(std::fabs(val - ref) <= rtol * std::fabs(ref), name, val, ref, rtol);
}

static Config base(int nx, int ny, int nz, double Lx, double Ly, double Lw) {
    Config c;
    c.lambda0 = 10e-6; c.Lu = c.lambda0 / (2 * M_PI); c.Tu = c.Lu / 299792458.0;
    c.nc = 8.8541878128e-12 * 9.1093837015e-31 * std::pow(299792458.0 / c.Lu, 2) / std::pow(1.602176634e-19, 2);
    c.nx = nx; c.ny = ny; c.nzeta = nz; c.Lx = Lx; c.Ly = Ly; c.Lwin = Lw; c.h = Lw / (nz - 1);
    c.sponge = 0; c.move_x = c.move_y = false;
    return c;
}

// 1. FFT: pair transforms of two real fields and back.
static void test_fft() {
    const int nx = 16, ny = 8; FFT2 f(nx, ny);
    std::mt19937 r(3); std::uniform_real_distribution<double> U(-1, 1);
    std::vector<double> a(nx * ny), b(nx * ny), a2(nx * ny), b2(nx * ny);
    for (auto& v : a) v = U(r);
    for (auto& v : b) v = U(r);
    std::vector<cplx> A(nx * ny), B(nx * ny);
    f.forward_pair(a.data(), b.data(), A.data(), B.data());
    // direct DFT of a at one wavenumber
    cplx d = 0; const int kx = 3, ky = 5;
    for (int ix = 0; ix < nx; ++ix) for (int iy = 0; iy < ny; ++iy)
        d += a[ix * ny + iy] * std::polar(1.0, -2 * M_PI * (double(kx * ix) / nx + double(ky * iy) / ny));
    check(std::abs(A[kx * ny + ky] - d) < 1e-12, "fft: pair transform = direct DFT", std::abs(A[kx * ny + ky]), std::abs(d), 1e-12);
    f.inverse_pair(A.data(), B.data(), a2.data(), b2.data());
    double e = 0; for (int i = 0; i < nx * ny; ++i) e = std::max(e, std::fabs(a2[i] - a[i]) + std::fabs(b2[i] - b[i]));
    check(e < 1e-13, "fft: inverse pair restores both fields", e, 0, 1e-13);
}

// 2. Vacuum: a tilted forward wave packet R ~ cos(kx x) cos(kappa zeta) evolves as exp(-i omega s) with the
//    dispersion of the reduced system, omega = 2 kappa k^2 / (4 kappa^2 - k^2), and keeps its amplitude.
static void test_dispersion() {
    const double kx = 0.1, kappa = 1.0;
    const int N = 1025; const double h = 2 * M_PI / kappa / 32, Lw = h * (N - 1);
    Config c = base(16, 4, N, 2 * M_PI / kx, 10, Lw);
    c.self_fields = true; c.dz = 10; c.nmacro = 0;
    Simulation sim(c);
    const int ik = 1 * c.ny + 0;                       // (kx = 2 pi / Lx, ky = 0)
    const double zc = 0.5 * Lw, w = 6 * 2 * M_PI;
    for (int n = 0; n < N; ++n) {
        const double z = n * h, env = std::exp(-std::pow((z - zc) / w, 2));
        sim.Rx[size_t(n) * sim.nk + ik] = 0.5 * env * std::cos(kappa * z);
        sim.Rx[size_t(n) * sim.nk + (c.nx - 1) * c.ny] = 0.5 * env * std::cos(kappa * z);
    }
    sim.march(0.0);
    auto demod = [&]() {
        cplx a = 0;
        for (int n = 0; n < N; ++n) a += sim.Rx[size_t(n) * sim.nk + ik] * std::polar(1.0, -kappa * n * h);
        return a;
    };
    const cplx a0 = demod();
    const int steps = 40;
    for (int k = 0; k < steps; ++k) { sim.march(0.5 * c.dz); }
    const cplx a1 = demod();
    const double om = -std::arg(a1 / a0) / (steps * c.dz);
    const double k2 = kx * kx, ref = 2 * kappa * k2 / (4 * kappa * kappa - k2);
    close("vacuum: omega of a tilted forward wave (0.1 rad)", om, ref, 0.01);
    close("vacuum: amplitude kept over 40 steps (trapezoidal rule)", std::abs(a1), std::abs(a0), 2e-3);
}

// 3. One electron in the analytic plane wave, no self-fields: light-front momentum h = gamma - p.khat
//    is conserved, the peak gamma follows from h and p_perp + q A, and the electron leaves the
//    pulse with its initial momentum (Lawson-Woodward).
static void test_plane_wave(const std::string& pol) {
    Config c = base(4, 4, 4, 100, 100, 1e7);
    c.self_fields = false; c.laser_on = true; c.a0 = 5; c.kL = 1; c.theta = 0.1; c.pol = pol;
    c.tau = 150; c.start_fwhm = 3.5; c.stop_fwhm = 3.5; c.push_phase = 0.025;
    c.gamma = 50; c.charge = 1e-15; c.nmacro = 1; c.zeta_center = 0.5e7;
    c.dz = 30; c.dz_push = 15; c.z_end = 1e9;
    Simulation sim(c);
    sim.init();
    auto& p = sim.parts[0];
    auto hlf = [&]() { const double g = std::sqrt(1 + p.px * p.px + p.py * p.py + p.pz * p.pz);
                       return g - p.px * sim.laser.kx - p.pz * sim.laser.kz; };
    const double h0 = hlf(), pz0 = p.pz, px0 = p.px;
    // p_perp relative to khat: along the p direction e_p = (kz, 0, -kx) and along y
    const double pp0 = px0 * sim.laser.kz - pz0 * sim.laser.kx;
    double dh = 0, gmax = 0;
    while (!sim.finished()) {
        sim.step();
        dh = std::max(dh, std::fabs(hlf() - h0));
        gmax = std::max(gmax, std::sqrt(1 + p.px * p.px + p.py * p.py + p.pz * p.pz));
    }
    // gamma + p_par = (1 + |p_perp|^2) / h, |p_perp|^2 largest where |p_perp0 + a e_pol| is largest
    const double a = c.a0;
    double pmax2 = pol == "s" ? pp0 * pp0 + a * a : (std::fabs(pp0) + a) * (std::fabs(pp0) + a);
    const double gref = 0.5 * ((1 + pmax2) / h0 + h0);
    const std::string tag = "plane wave (" + pol + "-pol): ";
    check(dh / h0 < 1e-5, (tag + "h = gamma - p.khat conserved").c_str(), dh / h0, 0, 1e-5);
    close((tag + "peak gamma from h and p_perp - A").c_str(), gmax, gref, 2e-3);
    close((tag + "p_z after the pulse (Lawson-Woodward)").c_str(), p.pz, pz0, 1e-5);
    check(std::fabs(p.px - px0) < 1e-4, (tag + "p_x after the pulse").c_str(), p.px, px0, 1e-4);
}

// 4. Field of a long round Gaussian bunch against Gauss's law: E_r = lambda (1 - exp(-r^2/2 sigma^2)) / (2 pi r).
static void test_gauss() {
    const double sig = 6.0, sz = 60.0, Lw = 8 * sz;
    Config c = base(128, 128, 129, 40 * sig, 40 * sig, Lw);
    c.gamma = 1000; c.sigx = c.sigy = sig; c.sigz = sz; c.charge = 1e-12; c.nmacro = 400000;
    c.zeta_center = 0.5 * Lw; c.dz = 10; c.k_filter = false;
    Simulation sim(c);
    sim.init();
    double Qcode = 0; for (auto& p : sim.parts) Qcode += -p.w;
    const double lam = Qcode / (std::sqrt(2 * M_PI) * sz);     // line density at the centre
    const int n = int(std::lround(c.zeta_center / c.h));
    for (double rr : {1.0, 2.0}) {
        const double r = rr * sig;
        // E_x at (x = Lx/2 + r, y = Ly/2) and the mirror point, averaged (odd in x)
        const int ix = int(std::lround((0.5 * c.Lx + r) / sim.dx)), im = int(std::lround((0.5 * c.Lx - r) / sim.dx));
        const int iy = c.ny / 2;
        const double rg = ix * sim.dx - 0.5 * c.Lx;
        const double Ex = 0.5 * (sim.F[(0 * size_t(sim.N) + n) * sim.nk + ix * c.ny + iy] -
                                 sim.F[(0 * size_t(sim.N) + n) * sim.nk + im * c.ny + iy]);
        const double ref = lam * (1 - std::exp(-rg * rg / (2 * sig * sig))) / (2 * M_PI * rg);
        char name[80]; std::snprintf(name, sizeof name, "Gauss: E_r of a long bunch at r = %.0f sigma", rr);
        close(name, Ex, ref, 0.03);
    }
    // B_y of the co-moving field: R = (1 + beta) E, L ~ 0 -> B_y = E_x (gamma = 1000)
    const int ix = int(std::lround((0.5 * c.Lx + sig) / sim.dx)), im = int(std::lround((0.5 * c.Lx - sig) / sim.dx)), iy = c.ny / 2;
    const double rg = ix * sim.dx - 0.5 * c.Lx;
    const double By = 0.5 * (sim.F[(4 * size_t(sim.N) + n) * sim.nk + ix * c.ny + iy] - sim.F[(4 * size_t(sim.N) + n) * sim.nk + im * c.ny + iy]);
    close("Gauss: B_y of the bunch at r = 1 sigma", By, lam * (1 - std::exp(-rg * rg / (2 * sig * sig))) / (2 * M_PI * rg), 0.03);
}

// 5. A relativistic bunch in its own field (no laser) keeps its energy and its field.
static void test_equilibrium() {
    const double sig = 6.0, sz = 6.0, Lw = 16 * sz;
    Config c = base(64, 64, 257, 30 * sig, 30 * sig, Lw);
    c.gamma = 1000; c.sigx = c.sigy = sig; c.sigz = sz; c.charge = 10e-12; c.nmacro = 50000;
    c.zeta_center = 0.6 * Lw; c.dz = 50; c.dz_push = 50;
    Simulation sim(c);
    sim.init();
    auto mean_g = [&]() { double s = 0; for (auto& p : sim.parts) s += std::sqrt(1 + p.px * p.px + p.py * p.py + p.pz * p.pz);
                          return s / sim.parts.size(); };
    const double g0 = mean_g(), W0 = sim.field_energy;
    for (int k = 0; k < 100; ++k) sim.step();
    const double g1 = mean_g();
    check(std::fabs(g1 - g0) / g0 < 1e-6, "equilibrium: <gamma> of a gamma = 1000 bunch over 100 steps", (g1 - g0) / g0, 0, 1e-6);
    close("equilibrium: field energy kept over 100 steps", sim.field_energy, W0, 0.02);
}


// 6. The absorbing sponge must not drain the bunch through its own bound field: a dense gamma = 50 bunch
//    evolves the same with and without the sponge (damping all of R instead loses ~1e-3 of p_z per mm).
static void test_sponge() {
    double pz[2];
    for (int k = 0; k < 2; ++k) {
        Config c = base(32, 32, 129, 100, 100, 64);
        c.gamma = 50; c.sigx = c.sigy = 6; c.sigz = 6; c.charge = 160e-12; c.nmacro = 20000;
        c.zeta_center = 40; c.dz = 30; c.dz_push = 30; c.sponge = k ? 15 : 0; c.sponge_rate = 0.1;
        Simulation sim(c);
        sim.init();
        for (int st = 0; st < 40; ++st) sim.step();
        double s = 0; for (auto& p : sim.parts) s += p.pz;
        pz[k] = s / sim.parts.size();
    }
    check(std::fabs(pz[1] - pz[0]) / pz[0] < 2e-6, "sponge: no drag on a dense bunch (with vs without)", (pz[1] - pz[0]) / pz[0], 0, 2e-6);
}

int main() {
    test_fft();
    test_dispersion();
    test_plane_wave("s");
    test_plane_wave("p");
    test_gauss();
    test_equilibrium();
    test_sponge();
    std::printf("%s: %d failure(s)\n", failures ? "FAILED" : "ALL PASSED", failures);
    return failures ? 1 : 0;
}
