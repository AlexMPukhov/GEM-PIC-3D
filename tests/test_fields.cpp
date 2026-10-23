// Tests: FFTs, grids, vacuum dispersion and diffraction.
#include "test_util.hpp"

// 1. FFT: pair transforms of two real fields and back; any size against the direct DFT; the batched
//    transforms (2D for Fourier y, x only for the free-hand y) against FFT2 and FFT1, and their round trips.
void test_fft_pair() {
    const int nx = 16, ny = 8;
    FFT2 f(nx, ny);
    std::mt19937 r(3); std::uniform_real_distribution<double> U(-1, 1);
    std::vector<double> a(nx * ny), b(nx * ny), a2(nx * ny), b2(nx * ny);
    for (int i = 0; i < nx * ny; ++i) { a[i] = U(r); b[i] = U(r); }
    std::vector<cd> A(nx * ny), B(nx * ny);
    f.forward_pair(a.data(), b.data(), A.data(), B.data());
    cd ref = 0;
    for (int i = 0; i < nx; ++i) for (int j = 0; j < ny; ++j)
        ref += a[i * ny + j] * std::polar(1.0, -2 * M_PI * (3.0 * i / nx + 5.0 * j / ny));
    close("fft: pair transform = direct DFT", std::abs(A[3 * ny + 5]), std::abs(ref), 1e-12);
    f.inverse_pair(A.data(), B.data(), a2.data(), b2.data());
    double e = 0; for (int i = 0; i < nx * ny; ++i) e = std::max(e, std::fabs(a2[i] - a[i]) + std::fabs(b2[i] - b[i]));
    check(e < 1e-13, "fft: inverse pair restores both fields", e, 0, 1e-13);
}
void test_fft_sizes() {   // mixed radix and Bluestein
    std::mt19937 r(4); std::uniform_real_distribution<double> U(-1, 1);
    double err = 0;
    for (int n : {6, 15, 37, 60, 97, 120}) {
        FFT1 g(n);
        std::vector<cd> z(n), ref(n);
        for (auto& v : z) v = {U(r), U(r)};
        for (int k = 0; k < n; ++k)
            for (int j = 0; j < n; ++j) ref[k] += z[j] * std::polar(1.0, -2 * M_PI * double((long(j) * k) % n) / n);
        g.run(z.data(), false);
        for (int k = 0; k < n; ++k) err = std::max(err, std::abs(z[k] - ref[k]));
    }
    check(err < 1e-12, "fft: sizes 6, 15, 37, 60, 97, 120 = direct DFT", err, 0, 1e-12);
}
void test_fft_batched(int nx, int ny) {
    const int np = 5;
    std::mt19937 r(5); std::uniform_real_distribution<double> U(-1, 1);
    std::vector<cpx> h(np * nx * ny);
    for (auto& v : h) v = C(U(r), U(r));
    for (bool two_d : {true, false}) {
        Buf<cpx> dbuf; dbuf.upload(h);
        FftMany fm(nx, ny, two_d);
        fm.run(dbuf.data(), np, false);
        std::vector<cpx> got = dbuf.host();
        double err = 0;
        if (two_d) {
            FFT2 f(nx, ny);
            for (int p = 0; p < np; ++p) {
                std::vector<cd> z(nx * ny);                       // FFT2 layout ix*ny + iy
                for (int j = 0; j < ny; ++j) for (int i = 0; i < nx; ++i) z[i * ny + j] = zc(h[(p * ny + j) * nx + i]);
                f.forward(z.data());
                for (int j = 0; j < ny; ++j) for (int i = 0; i < nx; ++i)
                    err = std::max(err, std::abs(z[i * ny + j] - zc(got[(p * ny + j) * nx + i])));
            }
        } else {
            FFT1 f1(nx);
            for (int row = 0; row < np * ny; ++row) {
                std::vector<cd> z(nx);
                for (int i = 0; i < nx; ++i) z[i] = zc(h[row * nx + i]);
                f1.run(z.data(), false);
                for (int i = 0; i < nx; ++i) err = std::max(err, std::abs(z[i] - zc(got[row * nx + i])));
            }
        }
        fm.run(dbuf.data(), np, true);
        got = dbuf.host();
        double rt = 0;
        for (size_t i = 0; i < h.size(); ++i) rt = std::max(rt, std::fabs(got[i].re - h[i].re) + std::fabs(got[i].im - h[i].im));
        check(err < 1e-12 && rt < 1e-13, strf("fft: batched %s %dx%d, round trip", two_d ? "2D transform = FFT2" : "x-transform = FFT1", nx, ny),
              std::max(err, rt), 0, 1e-12);
    }
}

// 2. Free-hand grids: the mapping and the eigenvectors of the y Laplacian.
void test_grids() {
    StepMapping m;
    m.set({0, 10, 30}, {1, 1, 3});
    const double hstep = m.coord(20.5) - m.coord(19.5), href = m.step(m.coord(20));
    double inv = 0;
    for (double xi : {0.0, 3.3, 10.0, 17.5, 25.0, 40.0}) inv = std::max(inv, std::fabs(m.index(m.coord(xi)) - xi));
    check(inv < 1e-12 && std::fabs(hstep / href - 1) < 1e-3, "grid: mapping inverse, local step = h(x)", inv, 0, 1e-12);
    Config c = Config::base_units(10e-6);
    c.y_table({0, 10, 30}, {1, 1, 3}, 60);
    check(std::fabs(c.ynode[c.ny / 2]) < 1e-15 && std::fabs(c.ynode[0] + c.ynode[c.ny]) < 1e-12 && c.ynode[c.ny] >= 60,
          "grid: y table mirrored about the axis, walls beyond y_max", c.ynode[c.ny], 60, 0);
    // eigenvectors: (k^2 - Lap_y) applied to an eigenvector v gives (k^2 + mu) v
    c.nx = 4; c.Lx = 10; c.zeta_uniform(8, 7); c.self_fields = true;
    Simulation sim(c, comm);
    const int ny = c.ny;
    const std::vector<double> evb = sim.evb.host(), evl = sim.evl.host();
    double res = 0;
    for (int s : {1, 7, ny / 2, ny - 1}) {
        std::vector<double> v(ny + 1, 0.0);
        double vmax = 0;
        for (int j = 1; j < ny; ++j) { v[j] = evb[size_t(j) * ny + s]; vmax = std::max(vmax, std::fabs(v[j])); }
        for (int j = 1; j < ny; ++j) {
            const double lap = ((v[j + 1] - v[j]) / sim.hDel[j] - (v[j] - v[j - 1]) / sim.hDel[j - 1]) / sim.hA[j];
            res = std::max(res, std::fabs(-lap - evl[s] * v[j]) / (evl[s] * vmax));
        }
    }
    check(res < 1e-9, "grid: eigenvectors of the free-hand y Laplacian", res, 0, 1e-9);
}

// 3. Vacuum: a wave tilted in x, R_y ~ cos(k_x x) cos(kappa zeta), uniform in y, evolves as exp(-i omega s) with
//    the dispersion of the reduced system, omega = 2 kappa k^2 / (4 kappa^2 - k^2), and keeps its amplitude;
//    on the default grids, on a free-hand zeta grid (fine over the packet, 4x coarser at both ends), with
//    the free-hand y (finite differences between walls), and with mode numbers that are not powers of two.
void test_dispersion(const std::string& kind) {
    const double kx = 0.1, kappa = 1.0, h = 2 * M_PI / kappa / 32;
    Config c = kind == "sizes" ? base(15, 6, 1025, 2 * M_PI / kx, 10, h * 1024) : base(16, 4, 1025, 2 * M_PI / kx, 10, h * 1024);
    const double w = 6 * 2 * M_PI;
    double zc0 = 0.5 * c.Lwin;
    if (kind == "zeta") {
        const double a1 = 40, a2 = a1 + 30 * h * 4, b1 = a2 + 8 * w, b2 = b1 + 30 * h * 4;
        c.zeta_table({0, a1, a2, b1, b2}, {4 * h, 4 * h, h, h, 4 * h}, b2 + 60);
        zc0 = 0.5 * (a2 + b1);
    }
    if (kind == "y") c.y_table({0}, {2.5}, 5);
    c.self_fields = true; c.dz = 10; c.nmacro = 0;
    Simulation sim(c, comm);
    const int N = c.nzeta, nx = c.nx, ny = c.ny;
    std::vector<cpx> R(size_t(N) * sim.nk, C(0));
    for (int n = 0; n < N; ++n) {
        const double z = c.zeta[n], env = std::exp(-std::pow((z - zc0) / w, 2));
        for (int j = 0; j < (c.y_fourier ? 1 : ny); ++j)      // Fourier y: the k_y = 0 mode; free-hand: every row
            R[size_t(n) * sim.nk + j * nx + 1] = R[size_t(n) * sim.nk + j * nx + nx - 1] = C(0.5 * env * std::cos(kappa * z));
    }
    sim.Ry.upload(R);
    sim.solve_fields(0.0);
    const std::vector<double> ws = sim.wsl.host();
    const int jm = c.y_fourier ? 0 : 1;
    auto demod = [&]() {
        const std::vector<cpx> r = sim.Ry.host();
        cd a = 0;
        for (int n = 0; n < N; ++n) a += ws[n] * zc(r[size_t(n) * sim.nk + jm * nx + 1]) * std::polar(1.0, -kappa * c.zeta[n]);
        return a;
    };
    const cd a0 = demod();
    const int steps = 40;
    for (int k = 0; k < steps; ++k) sim.solve_fields(0.5 * c.dz);
    const cd a1 = demod();
    const double om = -std::arg(a1 / a0) / (steps * c.dz);
    const double k2 = kx * kx, ref = 2 * kappa * k2 / (4 * kappa * kappa - k2);
    const std::string tag = kind == "zeta" ? "vacuum (free-hand zeta): " : kind == "y" ? "vacuum (free-hand y): "
                          : kind == "sizes" ? "vacuum (15 x 6 modes): " : "vacuum: ";
    close(tag + "omega of a tilted forward wave (0.1 rad)", om, ref, 0.01);
    close(tag + "amplitude kept over 40 steps", std::abs(a1), std::abs(a0), 2e-3);
}

// 4. Diffraction in y: a forward Gaussian beam R_y = exp(-y^2/w0^2) cos(kappa zeta) (k_x = 0) spreads with
//    Fourier y, with the free-hand y at a constant step, and out of the fine core of a free-hand y grid into
//    its coarse cells. Its rms width follows from the phase phi(k_y) of the reduced system after n steps of
//    the trapezoidal rule in s, phi = 2 n atan(omega dz/2), omega = 2 kappa k^2/(4 kappa^2 - k^2):
//    <y^2> = w0^2/4 + <phi'(k)^2> over |u(k)|^2 ~ exp(-k^2 w0^2/2). The free-hand 1 -> 4 grid is checked
//    against the constant step (ref_fd, its returned width).
double test_diffraction(const std::string& kind, double ref_fd) {
    const double kappa = 1.0, h = 2 * M_PI / kappa / 32, w0 = 10, wz = 8 * 2 * M_PI;
    const int N = 1281;
    Config c = base(4, 256, N, 40, 256, h * (N - 1));
    if (kind == "constant step") c.y_table({0}, {1}, 150);
    if (kind == "free-hand 1 -> 4") c.y_table({0, 30, 60}, {1, 1, 4}, 150);
    c.dz = 10; c.nmacro = 0;
    Simulation sim(c, comm);
    const int nx = c.nx, ny = c.ny;
    const double zc0 = 0.5 * c.Lwin;
    // y of the points that carry R_y: Fourier nodes (centred on node 0, periodic) or the half-nodes
    std::vector<double> yp(ny);
    for (int j = 0; j < ny; ++j) yp[j] = c.y_fourier ? (j <= ny / 2 ? j : j - ny) * sim.dy : 0.5 * (c.ynode[j] + c.ynode[j + 1]);
    std::vector<cd> prof(ny);
    for (int j = 0; j < ny; ++j) prof[j] = std::exp(-yp[j] * yp[j] / (w0 * w0));
    FFT1 fy(c.y_fourier ? ny : 1);
    if (c.y_fourier) fy.run(prof.data(), false);              // the fields are spectra in y
    std::vector<cpx> R(size_t(N) * sim.nk, C(0));
    for (int n = 0; n < N; ++n) {
        const double z = c.zeta[n], env = std::exp(-std::pow((z - zc0) / wz, 2)) * std::cos(kappa * z);
        for (int j = 0; j < ny; ++j) R[size_t(n) * sim.nk + j * nx] = C(env * prof[j].real(), env * prof[j].imag());
    }
    sim.Ry.upload(R);
    sim.solve_fields(0.0);
    const int steps = 15;
    for (int k = 0; k < steps; ++k) sim.solve_fields(0.5 * c.dz);
    const std::vector<cpx> r = sim.Ry.host();
    std::vector<cd> u(ny, 0.0);
    for (int j = 0; j < ny; ++j)
        for (int n = 0; n < N; ++n) u[j] += zc(r[size_t(n) * sim.nk + j * nx]) * std::polar(1.0, -kappa * c.zeta[n]);
    if (c.y_fourier) fy.run(u.data(), true);                  // back to y (normalization drops out)
    double s0 = 0, s2 = 0;
    for (int j = 0; j < ny; ++j) { s0 += sim.hDel[j] * std::norm(u[j]); s2 += sim.hDel[j] * std::norm(u[j]) * yp[j] * yp[j]; }
    const double s = steps * c.dz;
    double q0 = 0, q2 = 0;
    for (int i = -4000; i <= 4000; ++i) {
        const double k = i * 1e-4, g = std::exp(-k * k * w0 * w0 / 2);
        const double d = 4 * kappa * kappa - k * k, om = 2 * kappa * k * k / d;
        const double om1 = (4 * kappa * k * d + 4 * kappa * k * k * k) / (d * d);
        const double ph1 = s * om1 / (1 + 0.25 * om * om * c.dz * c.dz);
        q0 += g; q2 += g * ph1 * ph1;
    }
    const double ref = std::sqrt(w0 * w0 / 4 + q2 / q0), width = std::sqrt(s2 / s0);
    const std::string name = strf("diffraction in y (%s): rms width after %.1f Rayleigh lengths", kind.c_str(), s / (0.5 * kappa * w0 * w0));
    if (kind == "Fourier") close(name, width, ref, 3e-3);
    else if (kind == "constant step") close(name, width, ref, 1e-2);
    else close(name + " = constant step", width, ref_fd, 2e-3);
    return width;
}
