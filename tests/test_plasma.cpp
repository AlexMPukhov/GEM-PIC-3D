// Tests: plasma wakes (1D laser, 3D beam), smoothing, streaming kinematics.
#include "test_util.hpp"

// 10. Plasma: 1D linear wake of a Gaussian laser pulse (a0 = 0.05, n0 = 0.01 n_c, transversely uniform) against
//     the cold-fluid theory E'' + kp^2 E = -kp^2 d/dzeta <a^2>/2, <a^2> = a0^2 g^2/2: amplitude and plasma
//     wavelength behind the pulse, after 3 steps; with the extrapolated predictor and the previous-level one.
void test_plasma_wake_1d(bool extrap) {
    Config c = Config::base_units(10e-6);
    const double h = 2 * M_PI / 32;
    const int N = int(300 / h) + 1;
    c.nx = 4; c.Lx = 40; c.y_uniform(4, 40); c.zeta_uniform(N, h * (N - 1));
    c.sponge = c.sponge_y = 0; c.move_x = c.move_y = false;
    c.plasma_on = true; c.n0 = 0.01; c.ppc_x = 1; c.ppc_y = 1; c.predictor_extrapolate = extrap;
    c.nmacro = 0; c.dz = 25; c.dz_push = 25; c.zeta_center = 0.75 * c.Lwin;
    Config::GridLaser g; g.a0 = 0.05; g.kL = 1; g.waist = 1e7; g.tau = 2 * M_PI / std::sqrt(c.n0) * 0.4; g.zeta_c = 240;
    c.glasers.push_back(g);
    Simulation sim(c, comm);
    sim.init();
    for (int k = 0; k < 3; ++k) sim.step();
    const std::vector<float> F = sim.F.host();
    const double kp = std::sqrt(c.n0);
    auto a2 = [&](double z) { const double d = z - g.zeta_c; const double e = std::exp(-2 * std::log(2.0) * d * d / (g.tau * g.tau)); return 0.5 * g.a0 * g.a0 * e * e; };
    double emax = 0, rmax = 0;
    std::vector<double> ez(N);
    for (int n = 0; n < N; ++n) {
        double v = 0;
        for (int i = 0; i < sim.nk; ++i) v += F[(size_t(2) * (sim.ns + 2) + n) * sim.nk + i] / sim.nk;
        ez[n] = v;
        if (c.zeta[n] < 200) emax = std::max(emax, std::fabs(v));
    }
    for (int n = 0; n < N; n += 4) {
        const double z = c.zeta[n];
        if (z >= 200) break;
        double E = 0; const double d2 = h / 4;
        for (double zp = z; zp < c.Lwin; zp += d2)
            E += std::sin(kp * (zp - z)) / kp * (-kp * kp * (a2(zp + 1e-3) - a2(zp - 1e-3)) / 2e-3 / 2) * d2;
        rmax = std::max(rmax, std::fabs(E));
    }
    // plasma wavelength from the first and last zero crossings behind the pulse
    double z1 = -1, z2 = -1; int cnt = 0;
    for (int n = N - 2; n >= 0; --n) {
        if (c.zeta[n] > 200) continue;
        if ((ez[n] > 0) != (ez[n + 1] > 0)) {
            const double zc = c.zeta[n] + (c.zeta[n + 1] - c.zeta[n]) * ez[n] / (ez[n] - ez[n + 1]);
            if (z1 < 0) z1 = zc;
            z2 = zc; ++cnt;
        }
    }
    const double lam = 2 * (z1 - z2) / (cnt - 1);
    const std::string tag = extrap ? "plasma: 1D laser wake: " : "plasma: 1D laser wake (predictor = previous): ";
    close((tag + "amplitude against linear theory").c_str(), emax, rmax, extrap ? 0.02 : 0.03);
    close((tag + "plasma wavelength 2 pi / k_p").c_str(), lam, 2 * M_PI / kp, 0.005);
}

// 11. Plasma: 3D linear wake of a narrow electron bunch (gamma = 1000, k_p sigma_r = k_p sigma_z = 0.5,
//     n_b = 0.01 n0), rigid (numerics.freeze_bunch), on the axis after 30 steps (the start-up transient decays over the first tens of steps).
//     Reference: the linear theory of the discrete scheme. The plasma current is deposited and the fields are
//     gathered with TSC, so a transverse mode k sees the plasma frequency k_p S(k), S = sinc^3(k_x dx/2)
//     sinc^3(k_y dy/2); the bunch is deposited with S(k) too. Per mode
//       E_z(k, zeta) = k_p^2 S^2 / (k^2 + k_p^2 S^2) int_zeta dzeta' cos(k_p S (zeta' - zeta)) S n_b(k, zeta'),
//     summed over the modes of the box (Nyquist modes excluded). With S = 1 this is the continuum theory, from
//     which the scheme deviates by about 9 % (the wake modes oscillate ~1.4 % too slowly at sigma/dx = 2.7).
//     The free-hand y run is compared with the Fourier y run.
std::vector<double> beam_wake_axis(bool freehand, Config& c, double& Ne, Simulation*& keep) {
    c = Config::base_units(10e-6);
    const double sig = 5, sz = 5, Lw = 150, h = 0.5;
    c.nx = 64; c.Lx = 120;
    if (freehand) c.y_table({0}, {1.875}, 60); else c.y_uniform(64, 120);
    c.zeta_uniform(int(Lw / h) + 1, Lw);
    c.sponge = c.sponge_y = 0; c.move_x = c.move_y = false; c.k_filter = false;
    c.plasma_on = true; c.n0 = 0.01; c.ppc_x = 1; c.ppc_y = 1;
    const double nb = 0.01 * c.n0;
    Ne = nb * std::pow(2 * M_PI, 1.5) * sig * sig * sz;                        // electrons (code units)
    c.gamma = 1000; c.sigx = c.sigy = sig; c.sigz = sz; c.nmacro = 200000; c.zeta_center = 120;
    c.charge = Ne * c.nc * std::pow(c.Lu, 3) * 1.602176634e-19;
    c.dz = 10; c.dz_push = 10; c.freeze_bunch = true;
    keep = new Simulation(c, comm);
    Simulation& sim = *keep;
    sim.init();
    for (int k = 0; k < 30; ++k) sim.step();
    const std::vector<float> F = sim.F.host();
    const int ic = c.nx / 2, jc = c.ny / 2;                                      // the axis: x = 0, y = 0
    std::vector<double> ez(c.nzeta);
    for (int n = 0; n < c.nzeta; ++n) ez[n] = F[(size_t(2) * (sim.ns + 2) + n) * sim.nk + size_t(jc) * c.nx + ic];
    return ez;
}
void test_plasma_wake_beam() {
    Config c, cf; double Ne, Nf; Simulation *s1 = nullptr, *s2 = nullptr;
    const std::vector<double> ez = beam_wake_axis(false, c, Ne, s1);
    const std::vector<double> ef = beam_wake_axis(true, cf, Nf, s2);
    const double kp = std::sqrt(c.n0), sig = c.sigx, sz = c.sigz, dx = s1->dx, dy = s1->dy, Ly = c.Ly;
    delete s1; delete s2;
    const double lam0 = Ne / (std::sqrt(2 * M_PI) * sz);
    auto sinc = [](double x) { return x == 0 ? 1.0 : std::sin(x) / x; };
    // per mode: A(k) and the cosine/sine moments of the bunch profile at the frequency K = k_p S(k)
    std::vector<double> zs;
    for (double zp = 0; zp < c.Lwin; zp += 0.05) zs.push_back(zp);
    std::vector<double> E(c.nzeta, 0.0), E0(c.nzeta, 0.0);
    for (int m = -c.nx / 2 + 1; m < c.nx / 2; ++m)
        for (int l = -c.ny / 2 + 1; l < c.ny / 2; ++l) {
            const double kx = 2 * M_PI * m / c.Lx, ky = 2 * M_PI * l / Ly, k2 = kx * kx + ky * ky;
            const double S = std::pow(sinc(0.5 * kx * dx) * sinc(0.5 * ky * dy), 3);
            for (int cont = 0; cont < 2; ++cont) {                  // 0: the discrete scheme, 1: the continuum
                const double Sk = cont ? 1.0 : S, K = kp * Sk;
                const double A = std::exp(-0.5 * k2 * sig * sig) * Sk * K * K / (k2 + K * K) / (c.Lx * Ly);
                double Cc = 0, Ss = 0;
                for (double zp : zs) {
                    const double g = lam0 * std::exp(-std::pow(zp - c.zeta_center, 2) / (2 * sz * sz)) * 0.05;
                    Cc += std::cos(K * zp) * g; Ss += std::sin(K * zp) * g;
                }
                std::vector<double>& out = cont ? E0 : E;
                for (int n = 0; n < c.nzeta; ++n) {
                    const double z = c.zeta[n];
                    // int_zeta^inf cos(K (zeta' - zeta)) g: only zeta' > zeta contributes; the bunch is ahead of the
                    // points used (z < 110), so the full moments are the integral
                    out[n] += A * (std::cos(K * z) * Cc + std::sin(K * z) * Ss);
                }
            }
        }
    double num = 0, den = 0, n0 = 0, nf = 0, df = 0;
    for (int n = 0; n < c.nzeta; ++n) {
        const double z = c.zeta[n];
        if (z > 100 || z < 20) continue;                                        // behind the bunch
        num += (ez[n] - E[n]) * (ez[n] - E[n]); den += E[n] * E[n];
        n0 += (ez[n] - E0[n]) * (ez[n] - E0[n]);
        nf += (ef[n] - ez[n]) * (ef[n] - ez[n]); df += ez[n] * ez[n];
    }
    std::printf("%-66s INFO  value %.6g  (scheme vs continuum theory)\n", "plasma: 3D beam wake on the axis: rms deviation from continuum",
                std::sqrt(n0 / den));
    check(std::sqrt(num / den) < 0.03, "plasma: 3D beam wake on the axis = linear theory of the scheme (rms)", std::sqrt(num / den), 0, 0.03);
    check(std::sqrt(nf / df) < 0.03, "plasma: 3D beam wake on the axis, free-hand y = Fourier y (rms)", std::sqrt(nf / df), 0, 0.03);
}

// 12. Plasma: the smoothing (1 - a^2 lap_perp)^-1 of the streaming current commutes with the divergence on the
//     free-hand y grid: N div(J_a) = div(J_0) with N = 1 + a^2 (k_x^2 - b-b+) on the nodes.
void test_smoothing_divergence() {
    Config c = Config::base_units(10e-6);
    c.nx = 8; c.Lx = 20; c.y_table({0, 5, 15}, {1, 1, 3}, 20); c.zeta_uniform(8, 7);
    c.plasma_on = true; c.n0 = 0.01; c.k_filter = false; c.sponge = c.sponge_y = 0; c.move_x = false;
    c.nmacro = 0; c.dz = 10;
    std::vector<cpx> J[2];
    std::mt19937 r(5); std::uniform_real_distribution<double> U(-1, 1);
    const int nk = c.nx * c.ny;
    std::vector<double> src(3 * nk);
    for (auto& v : src) v = U(r);
    for (int k = 0; k < 2; ++k) {
        Config ck = c; ck.smooth_length = k ? 2.0 : 0.0;
        Simulation sim(ck, comm);
        sim.jsr.upload(src);
        sim.stream_spectra();
        J[k] = sim.jsn.host();
    }
    Simulation sim(c, comm);
    const std::vector<double> kx = sim.kxs.host();
    const int nx = c.nx, ny = c.ny;
    std::vector<double> ia(ny), id(ny);
    for (int j = 0; j < ny; ++j) { id[j] = 1 / sim.hDel[j]; ia[j] = 1 / sim.hA[j]; }
    double res = 0, scale = 0;
    const double a2 = 4.0;
    for (int ix_ = 0; ix_ < nx; ++ix_) {
        auto divv = [&](const std::vector<cpx>& A, int j) {   // node j: i k_x J_x + (J_y(j) - J_y(j-1)) / A_j
            const cpx jx = A[size_t(j) * nx + ix_], ju = A[nk + size_t(j) * nx + ix_], jd = A[nk + size_t(j - 1) * nx + ix_];
            return std::complex<double>(-kx[ix_] * jx.im, kx[ix_] * jx.re) + std::complex<double>(ju.re - jd.re, ju.im - jd.im) * ia[j];
        };
        std::vector<std::complex<double>> d0(ny + 1, 0.0), da(ny + 1, 0.0);
        for (int j = 1; j < ny; ++j) { d0[j] = divv(J[0], j); da[j] = divv(J[1], j); }
        for (int j = 1; j < ny; ++j) {
            const std::complex<double> lap = ((da[j + 1] - da[j]) * id[j] - (da[j] - da[j - 1]) * id[j - 1]) * ia[j];
            const std::complex<double> Nd = da[j] + a2 * (kx[ix_] * kx[ix_] * da[j] - lap);
            res = std::max(res, std::abs(Nd - d0[j])); scale = std::max(scale, std::abs(d0[j]));
        }
    }
    check(res < 1e-12 * scale, "plasma: smoothing commutes with the divergence (free-hand y)", res / scale, 0, 1e-12);
}

// 13. Streaming and trapping kinematics in vacuum (plasma block, no plasma within r_max = 0): an electron with
//     v_z = 0.995 joining the march at zeta0 is trapped when it reaches z = s1, at zeta0 - (1 - v_z) dz / v_z, and
//     moves on in the beam list; one with v_z = 0.707 < vz_min leaves the beam list and keeps streaming.
void test_streaming_kinematics() {
    Config c = Config::base_units(10e-6);
    c.nx = 8; c.Lx = 40; c.y_uniform(8, 40); c.zeta_uniform(201, 200);
    c.plasma_on = true; c.n0 = 0.01; c.r_max = 0; c.sponge = c.sponge_y = 0; c.move_x = c.move_y = false;
    c.nmacro = 0; c.dz = 50; c.dz_push = 50;
    Simulation sim(c, comm);
    sim.init();
    Particle f{}; f.x = 20; f.y = 20; f.zeta = 150.3; f.pz = 10; f.w = 1e-12; f.q = -1; f.qm = -1; f.sp = SP_PLASMA; f.z = 0; f.alive = 6;
    Particle sl = f; sl.pz = 1; sl.zeta = 120.7; sl.alive = 1; sl.sp = SP_BUNCH;
    sim.joining_.push_back(f);
    sim.set_particles({sl});
    sim.step();
    const double vf = 10 / std::sqrt(101.0), vs = 1 / std::sqrt(2.0);
    std::vector<Particle> b = sim.host_particles();
    const bool ok1 = b.size() == 1 && b[0].sp == SP_TRAPPED;
    const double zt = ok1 ? b[0].zeta : 0, zref = 150.3 - (1 - vf) * 50 / vf;
    check(ok1 && std::fabs(zt - zref) < 1e-9, "streaming: fast electron trapped at zeta0 - (1 - v_z) dz / v_z", zt, zref, 1e-9);
    const bool ok2 = sim.joining_.size() == 1;
    const double zs = ok2 ? sim.joining_[0].zeta : 0, zsref = 120.7 - (1 - vs) * 50 / vs;
    check(ok2 && std::fabs(zs - zsref) < 1e-9, "streaming: slow electron leaves the beam list and keeps streaming", zs, zsref, 1e-9);
    sim.step();
    b = sim.host_particles();
    const double zt2 = b.empty() ? 0 : b[0].zeta;
    check(!b.empty() && std::fabs(zt2 - (zref - (1 - vf) * 50 / vf)) < 1e-6, "streaming: trapped electron moves on in the beam list", zt2,
          zref - (1 - vf) * 50 / vf, 1e-6);
}
