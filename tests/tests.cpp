// Tests of GEM-PIC-3D against independent references. Device data are read through host copies, so the
// same tests run in the CPU and the GPU build.
#include <cmath>
#include <complex>
#include <cstdio>
#include <random>
#include "sim.hpp"

static int failures = 0;
static Comm comm;
static void check(bool ok, const char* name, double val, double ref, double tol) {
    std::printf("%-60s %s  value %.6g  reference %.6g  tol %.2g\n", name, ok ? "PASS" : "FAIL", val, ref, tol);
    std::fflush(stdout);
    if (!ok) ++failures;
}
static void close(const char* name, double val, double ref, double rtol) {
    check(std::fabs(val - ref) <= rtol * std::fabs(ref), name, val, ref, rtol);
}

static Config base(int nx, int ny, int nz, double Lx, double Ly, double Lw) {
    Config c = Config::base_units(10e-6);
    c.nx = nx; c.ny = ny; c.nzeta = nz; c.Lx = Lx; c.Ly = Ly; c.Lwin = Lw; c.h = Lw / (nz - 1);
    c.sponge = 0; c.move_x = c.move_y = false;
    return c;
}
static double mean_pz(Simulation& sim) {
    double s = 0; long n = 0;
    for (auto& p : sim.host_particles()) if (p.alive == 1) { s += p.pz; ++n; }
    return s / n;
}

// 1. FFT: pair transforms of two real fields and back; the batched transform against the plane-by-plane one.
static void test_fft() {
    const int nx = 16, ny = 8; FFT2 f(nx, ny);
    std::mt19937 r(3); std::uniform_real_distribution<double> U(-1, 1);
    std::vector<double> a(nx * ny), b(nx * ny), a2(nx * ny), b2(nx * ny);
    for (auto& v : a) v = U(r);
    for (auto& v : b) v = U(r);
    std::vector<std::complex<double>> A(nx * ny), B(nx * ny);
    f.forward_pair(a.data(), b.data(), A.data(), B.data());
    std::complex<double> d = 0; const int kx = 3, ky = 5;
    for (int i = 0; i < nx; ++i) for (int j = 0; j < ny; ++j)
        d += a[i * ny + j] * std::polar(1.0, -2 * M_PI * (double(kx * i) / nx + double(ky * j) / ny));
    check(std::abs(A[kx * ny + ky] - d) < 1e-12, "fft: pair transform = direct DFT", std::abs(A[kx * ny + ky]), std::abs(d), 1e-12);
    f.inverse_pair(A.data(), B.data(), a2.data(), b2.data());
    double e = 0; for (int i = 0; i < nx * ny; ++i) e = std::max(e, std::fabs(a2[i] - a[i]) + std::fabs(b2[i] - b[i]));
    check(e < 1e-13, "fft: inverse pair restores both fields", e, 0, 1e-13);
    // batched (device) transform of 5 planes against FFT2
    const int np = 5;
    std::vector<cpx> h(np * nx * ny);
    for (auto& v : h) v = C(U(r), U(r));
    Buf<cpx> dbuf; dbuf.upload(h);
    FftMany fm(nx, ny);
    fm.run(dbuf.data(), np, false);
    std::vector<cpx> got = dbuf.host();
    double err = 0;
    for (int p = 0; p < np; ++p) {
        std::vector<std::complex<double>> z(nx * ny);
        for (int i = 0; i < nx * ny; ++i) z[i] = {h[p * nx * ny + i].re, h[p * nx * ny + i].im};
        f.forward(z.data());
        for (int i = 0; i < nx * ny; ++i)
            err = std::max(err, std::abs(z[i] - std::complex<double>(got[p * nx * ny + i].re, got[p * nx * ny + i].im)));
    }
    check(err < 1e-12, "fft: batched transform = plane by plane", err, 0, 1e-12);
}

// 2. Vacuum: a tilted forward wave packet R ~ cos(kx x) cos(kappa zeta) evolves as exp(-i omega s) with the
//    dispersion of the reduced system, omega = 2 kappa k^2 / (4 kappa^2 - k^2), and keeps its amplitude.
static void test_dispersion() {
    const double kx = 0.1, kappa = 1.0;
    const int N = 1025; const double h = 2 * M_PI / kappa / 32, Lw = h * (N - 1);
    Config c = base(16, 4, N, 2 * M_PI / kx, 10, Lw);
    c.self_fields = true; c.dz = 10; c.nmacro = 0;
    Simulation sim(c, comm);
    const int ik = 1 * c.ny + 0, ikm = (c.nx - 1) * c.ny;
    const double zc = 0.5 * Lw, w = 6 * 2 * M_PI;
    std::vector<cpx> R(size_t(N) * sim.nk, C(0));
    for (int n = 0; n < N; ++n) {
        const double z = n * h, env = std::exp(-std::pow((z - zc) / w, 2));
        R[size_t(n) * sim.nk + ik] = R[size_t(n) * sim.nk + ikm] = C(0.5 * env * std::cos(kappa * z));
    }
    sim.Rx.upload(R);
    sim.solve_fields(0.0);
    auto demod = [&]() {
        const std::vector<cpx> r = sim.Rx.host();
        std::complex<double> a = 0;
        for (int n = 0; n < N; ++n)
            a += std::complex<double>(r[size_t(n) * sim.nk + ik].re, r[size_t(n) * sim.nk + ik].im) * std::polar(1.0, -kappa * n * h);
        return a;
    };
    const std::complex<double> a0 = demod();
    const int steps = 40;
    for (int k = 0; k < steps; ++k) sim.solve_fields(0.5 * c.dz);
    const std::complex<double> a1 = demod();
    const double om = -std::arg(a1 / a0) / (steps * c.dz);
    const double k2 = kx * kx, ref = 2 * kappa * k2 / (4 * kappa * kappa - k2);
    close("vacuum: omega of a tilted forward wave (0.1 rad)", om, ref, 0.01);
    close("vacuum: amplitude kept over 40 steps (trapezoidal rule)", std::abs(a1), std::abs(a0), 2e-3);
}

// 3. One electron in the analytic plane wave, no self-fields: h = gamma - p.khat is conserved, the peak gamma
//    follows from h and p_perp + q A, and the electron leaves the pulse with its initial momentum.
static void test_plane_wave(const std::string& pol) {
    Config c = base(4, 4, 4, 100, 100, 1e7);
    c.self_fields = false; c.laser_on = true; c.a0 = 5; c.kL = 1; c.theta = 0.1; c.pol = pol;
    c.tau = 150; c.start_fwhm = 3.5; c.stop_fwhm = 3.5; c.push_phase = 0.025;
    c.gamma = 50; c.charge = 1e-15; c.nmacro = 1; c.zeta_center = 0.5e7;
    c.dz = 30; c.dz_push = 15; c.z_end = 1e9;
    Simulation sim(c, comm);
    sim.init();
    auto P = [&]() { return sim.host_particles()[0]; };
    auto hlf = [&](const Particle& p) { const double g = std::sqrt(1 + p.px * p.px + p.py * p.py + p.pz * p.pz);
                                        return g - p.px * sim.laser.kx - p.pz * sim.laser.kz; };
    Particle p = P();
    const double h0 = hlf(p), pz0 = p.pz, px0 = p.px;
    const double pp0 = px0 * sim.laser.kz - pz0 * sim.laser.kx;
    double dh = 0, gmax = 0;
    for (;;) {
        sim.step();
        p = P();
        dh = std::max(dh, std::fabs(hlf(p) - h0));
        gmax = std::max(gmax, std::sqrt(1 + p.px * p.px + p.py * p.py + p.pz * p.pz));
        const double et = sim.laser.eta(p.x + sim.X0, sim.s, sim.s - (p.zeta - c.zeta_center));
        if (et - sim.laser.eta_c > c.stop_fwhm * c.tau) break;
    }
    const double a = c.a0;
    const double pmax2 = pol == "s" ? pp0 * pp0 + a * a : (std::fabs(pp0) + a) * (std::fabs(pp0) + a);
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
    Simulation sim(c, comm);
    sim.init();
    double Qcode = 0; for (auto& p : sim.host_particles()) Qcode += -p.w;
    const double lam = Qcode / (std::sqrt(2 * M_PI) * sz);
    const int n = int(std::lround(c.zeta_center / c.h));
    const std::vector<float> F = sim.F.host();
    auto at = [&](int comp, int ixx, int iyy) { return double(F[(size_t(comp) * (sim.ns + 2) + n) * sim.nk + ixx * c.ny + iyy]); };
    const int iy = c.ny / 2;
    for (double rr : {1.0, 2.0}) {
        const double r = rr * sig;
        const int ixp = int(std::lround((0.5 * c.Lx + r) / sim.dx)), ixm = int(std::lround((0.5 * c.Lx - r) / sim.dx));
        const double rg = ixp * sim.dx - 0.5 * c.Lx;
        const double Ex = 0.5 * (at(0, ixp, iy) - at(0, ixm, iy));
        const double ref = lam * (1 - std::exp(-rg * rg / (2 * sig * sig))) / (2 * M_PI * rg);
        char name[80]; std::snprintf(name, sizeof name, "Gauss: E_r of a long bunch at r = %.0f sigma", rr);
        close(name, Ex, ref, 0.03);
    }
    const int ixp = int(std::lround((0.5 * c.Lx + sig) / sim.dx)), ixm = int(std::lround((0.5 * c.Lx - sig) / sim.dx));
    const double rg = ixp * sim.dx - 0.5 * c.Lx;
    const double By = 0.5 * (at(4, ixp, iy) - at(4, ixm, iy));
    close("Gauss: B_y of the bunch at r = 1 sigma", By, lam * (1 - std::exp(-rg * rg / (2 * sig * sig))) / (2 * M_PI * rg), 0.03);
}

// 5. A relativistic bunch in its own field (no laser) keeps its energy and its field.
static void test_equilibrium() {
    const double sig = 6.0, sz = 6.0, Lw = 16 * sz;
    Config c = base(64, 64, 257, 30 * sig, 30 * sig, Lw);
    c.gamma = 1000; c.sigx = c.sigy = sig; c.sigz = sz; c.charge = 10e-12; c.nmacro = 50000;
    c.zeta_center = 0.6 * Lw; c.dz = 50; c.dz_push = 50;
    Simulation sim(c, comm);
    sim.init();
    auto mean_g = [&]() { double s = 0; long n = 0;
        for (auto& p : sim.host_particles()) { s += std::sqrt(1 + p.px * p.px + p.py * p.py + p.pz * p.pz); ++n; }
        return s / n; };
    const double g0 = mean_g(), W0 = sim.total_field_energy();
    for (int k = 0; k < 100; ++k) sim.step();
    const double g1 = mean_g();
    check(std::fabs(g1 - g0) / g0 < 1e-6, "equilibrium: <gamma> of a gamma = 1000 bunch over 100 steps", (g1 - g0) / g0, 0, 1e-6);
    close("equilibrium: field energy kept over 100 steps", sim.total_field_energy(), W0, 0.02);
}

// 6. The absorbing sponge must not drain the bunch through its own bound field: a dense gamma = 50 bunch
//    evolves the same with and without the sponge (damping all of R instead loses ~1e-3 of p_z per mm).
static void test_sponge() {
    double pz[2];
    for (int k = 0; k < 2; ++k) {
        Config c = base(32, 32, 129, 100, 100, 64);
        c.gamma = 50; c.sigx = c.sigy = 6; c.sigz = 6; c.charge = 160e-12; c.nmacro = 20000;
        c.zeta_center = 40; c.dz = 30; c.dz_push = 30; c.sponge = k ? 15 : 0; c.sponge_rate = 0.1;
        Simulation sim(c, comm);
        sim.init();
        for (int st = 0; st < 40; ++st) sim.step();
        pz[k] = mean_pz(sim);
    }
    check(std::fabs(pz[1] - pz[0]) / pz[0] < 2e-6, "sponge: no drag on a dense bunch (with vs without)", (pz[1] - pz[0]) / pz[0], 0, 2e-6);
}

int main(int argc, char** argv) {
    comm.init(&argc, &argv);
    std::printf("GEM-PIC-3D tests (%s build)\n", GEMPIC_GPU ? "CUDA" : "CPU");
    test_fft();
    test_dispersion();
    test_plane_wave("s");
    test_plane_wave("p");
    test_gauss();
    test_equilibrium();
    test_sponge();
    std::printf("%s: %d failure(s)\n", failures ? "FAILED" : "ALL PASSED", failures);
    comm.finalize();
    return failures ? 1 : 0;
}
