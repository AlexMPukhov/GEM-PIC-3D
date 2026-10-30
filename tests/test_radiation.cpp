// Physics check (make physics): radiation energy balance of coherent undulator emission.
#include "test_util.hpp"
#include <complex>

// 27. Coherent spontaneous emission of a short bunch in a planar undulator against the far-field radiation integral
//     (Jackson 14.65), as tests/undulator.sh of GEM-PIC-RZ: 1 fC, gamma = 20, sigma_z = 0.785 k0^-1, sigma_r =
//     200 k0^-1, K = 0.5, lambda_r = lambda0 (lambda_u = 4468 k0^-1), 4 periods with ramps of one period, no
//     natural focusing. Reference:
//       d2W/dw dOmega = e^2/(16 pi^3 eps0 c) |int n x ((n - b) x db/dt)/(1 - n.b)^2 e^{iw(t - n.r/c)} dt|^2
//     for the trajectory in the same field, times N_e^2 and the form factor exp(-k^2 sigma_z^2 - k^2 theta^2
//     sigma_r^2). On the grid the radiation is W_x - W_y = (1/2) int (E_x^2 + B_y^2 - E_y^2 - B_x^2) after the
//     undulator: the bound field of the round bunch contributes equally to both, and the y-polarized emission is
//     4e-4 of the total here (sigma_r >> the orbit offset K / (gamma k_u) = 18 k0^-1). The grid converges at second
//     order in dzeta and as 1 / L_x^2 in the box (the periodic images sample the narrow angular spectrum); converged
//     it agrees with the reference to 0.6 %. Here (dzeta = 0.2, L_x = 6400, dx = 50) it is about 2 % low.
static double radiation_integral(double gamma, double K, double lu, int periods, double sz, double sr, double Ne, double thmax) {
    const double ku = 2 * M_PI / lu, L = periods * lu, ramp = lu, B0 = K * ku;
    const int nz = periods * 96 + 1;
    std::vector<double> z(nz), ux(nz), t(nz), x(nz), bx(nz), bz(nz), dbx(nz), dbz(nz), dtdz(nz), wt(nz);
    for (int i = 0; i < nz; ++i) z[i] = L * i / (nz - 1);
    auto By = [&](double zz) { return B0 * std::min(1.0, std::min(zz / ramp, (L - zz) / ramp)) * std::cos(ku * zz); };
    for (int i = 1; i < nz; ++i) ux[i] = ux[i - 1] + 0.5 * (By(z[i]) + By(z[i - 1])) * (z[i] - z[i - 1]);
    for (int i = 0; i < nz; ++i) {
        const double uz = std::sqrt(gamma * gamma - 1 - ux[i] * ux[i]);
        bx[i] = ux[i] / gamma; bz[i] = uz / gamma; dtdz[i] = gamma / uz;
    }
    for (int i = 1; i < nz; ++i) {
        const double h = z[i] - z[i - 1];
        t[i] = t[i - 1] + 0.5 * h * (dtdz[i] + dtdz[i - 1]);
        x[i] = x[i - 1] + 0.5 * h * (bx[i] * dtdz[i] + bx[i - 1] * dtdz[i - 1]);
    }
    for (int i = 0; i < nz; ++i) {
        const int a = std::max(i - 1, 0), b = std::min(i + 1, nz - 1);
        dbx[i] = (bx[b] - bx[a]) / (z[b] - z[a]) / dtdz[i]; dbz[i] = (bz[b] - bz[a]) / (z[b] - z[a]) / dtdz[i];
        wt[i] = dtdz[i] * (z[b] - z[a]) / (b - a);
    }
    const int nnu = 601, nth = 121, nph = 16;               // phi in [0, pi]: the pattern is even in y
    const double nu0 = 0.4, nu1 = 1.6, dnu = (nu1 - nu0) / (nnu - 1), dth = thmax / (nth - 1), dph = M_PI / nph;
    double tot = 0;
#pragma omp parallel for reduction(+ : tot) schedule(dynamic)
    for (int it = 0; it < nth; ++it) {
        const double th = dth * it;
        std::vector<double> A0(nz), A1(nz), A2(nz), w(nz), arg(nz);
        for (int ip = 0; ip < nph; ++ip) {
            const double ph = dph * (ip + 0.5), n0 = std::sin(th) * std::cos(ph), n1 = std::sin(th) * std::sin(ph), n2 = std::cos(th);
            for (int i = 0; i < nz; ++i) {
                const double a0 = n1 * dbz[i], a1 = (n2 - bz[i]) * dbx[i] - (n0 - bx[i]) * dbz[i], a2 = -n1 * dbx[i];
                A0[i] = n1 * a2 - n2 * a1; A1[i] = n2 * a0 - n0 * a2; A2[i] = n0 * a1 - n1 * a0;
                w[i] = wt[i] / std::pow(1 - n0 * bx[i] - n2 * bz[i], 2);
                arg[i] = t[i] - (n0 * x[i] + n2 * z[i]);
            }
            for (int k = 0; k < nnu; ++k) {
                const double nu = nu0 + dnu * k;
                std::complex<double> s0 = 0, s1 = 0, s2 = 0;
                for (int i = 0; i < nz; ++i) { const std::complex<double> e = std::polar(w[i], nu * arg[i]); s0 += A0[i] * e; s1 += A1[i] * e; s2 += A2[i] * e; }
                const double F2 = std::exp(-nu * nu * (sz * sz + std::pow(std::sin(th) * sr, 2)));
                tot += (std::norm(s0) + std::norm(s1) + std::norm(s2)) * F2 * std::sin(th) * dth * 2 * dph * dnu;
            }
        }
    }
    // code energy unit eps0 E0^2 / k0^3 (E0 = m c^2 k0 / e): factor (4 pi r_e k0)^2 / (16 pi^3)
    const double re = 2.8179403262e-15, k0 = 2 * M_PI / 0.8e-6;
    return Ne * Ne * std::pow(4 * M_PI * re * k0, 2) / (16 * M_PI * M_PI * M_PI) * tot;
}

void test_undulator_radiation() {
    const double gamma = 20, K = 0.5, lu = 2 * gamma * gamma * 2 * M_PI / (1 + K * K / 2), sz = 0.785, sr = 200, Q = 1e-15;
    const int periods = 4;
    Config c = Config::base_units(0.8e-6);
    c.nx = 128; c.Lx = 6400; c.y_uniform(128, 6400);
    const double hz = 0.2, Lw = 200;
    c.zeta_uniform(int(Lw / hz) + 1, Lw);
    c.sponge = c.sponge_y = 0; c.move_x = c.move_y = false; c.k_filter = false; c.zeta_filter = false;
    c.dz = lu / 16; c.dz_push = c.dz; c.z_end = (periods + 0.5) * lu;
    Undulator u{};
    u.ku = 2 * M_PI / lu; u.B0 = K * u.ku; u.z0 = 0; u.z1 = periods * lu; u.ramp = lu; u.focusing = 0;
    c.undulators = {u};
    Config::Bunch b;
    b.charge = Q; b.gamma = gamma; b.sigx = b.sigy = sr; b.sigz = sz; b.zeta_center = 160; b.nmacro = 200000;
    c.bunches = {b};
    Simulation sim(c, comm);
    sim.init();
    while (!sim.finished()) sim.step();
    const std::vector<float> F = sim.F.host();
    const size_t fpl = size_t(sim.ns + 2) * sim.nk;
    double w = 0;
    for (int l = 0; l < sim.ns; ++l)
        for (int i = 0; i < sim.nk; ++i) {
            const size_t k = size_t(l) * sim.nk + i;
            const double ex = F[k], ey = F[fpl + k], bx = F[3 * fpl + k], by = F[4 * fpl + k];
            w += ex * ex + by * by - ey * ey - bx * bx;
        }
    w *= 0.5 * sim.dx * sim.dy * hz;
    const double ref = radiation_integral(gamma, K, lu, periods, sz, sr, Q / 1.602176634e-19, 5 / sr);
    close("undulator radiation: coherent energy (W_x - W_y) = far-field radiation integral", w, ref, 0.04);
}
