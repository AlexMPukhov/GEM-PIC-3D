#include "sim.hpp"
#include <algorithm>
#include <cmath>
#include <fstream>
#include <random>
#include <stdexcept>

namespace {
constexpr double c_SI = 299792458.0, e_SI = 1.602176634e-19, me_SI = 9.1093837015e-31,
                 eps0 = 8.8541878128e-12;
constexpr double q = -1.0;                       // electron charge in units of e
const cplx I(0, 1);

inline void cross(const double a[3], const double b[3], double r[3]) {
    r[0] = a[1] * b[2] - a[2] * b[1]; r[1] = a[2] * b[0] - a[0] * b[2]; r[2] = a[0] * b[1] - a[1] * b[0];
}
// Quadratic (TSC) weights on a periodic grid with nodes at i*d.
inline void tsc(double x, double d, int n, int idx[3], double w[3]) {
    const double xi = x / d;
    const int i0 = int(std::floor(xi + 0.5));
    const double f = xi - i0;
    w[0] = 0.5 * (0.5 - f) * (0.5 - f); w[1] = 0.75 - f * f; w[2] = 0.5 * (0.5 + f) * (0.5 + f);
    for (int k = 0; k < 3; ++k) idx[k] = ((i0 - 1 + k) % n + n) % n;
}
}  // namespace

// ---------------------------------------------------------------- configuration
Config Config::from_deck(const Deck& d) {
    Config c;
    c.lambda0 = d.num("grid", "lambda0", Kind::Length,
                      d.has("laser", "wavelength") ? d.num("laser", "wavelength", Kind::Length) : 1e-6);
    c.Lu = c.lambda0 / (2 * M_PI); c.Tu = c.Lu / c_SI;
    const double w0 = c_SI / c.Lu;
    c.nc = eps0 * me_SI * w0 * w0 / (e_SI * e_SI);
    auto L = [&](const std::string& b, const std::string& k) { return d.num(b, k, Kind::Length) / c.Lu; };
    auto Ld = [&](const std::string& b, const std::string& k, double def_code) {
        return d.has(b, k) ? d.num(b, k, Kind::Length) / c.Lu : def_code; };
    c.nx = int(d.integer("grid", "nx", 0)); c.ny = int(d.integer("grid", "ny", 0));
    c.nzeta = int(d.integer("grid", "nzeta", 0));
    c.Lx = L("grid", "Lx"); c.Ly = Ld("grid", "Ly", c.Lx);
    c.Lwin = L("grid", "window_length");
    c.dz = L("grid", "dz"); c.z_end = L("grid", "z_end");
    c.dz_push = Ld("grid", "dz_push", c.dz);
    c.sponge = Ld("grid", "sponge", 0.1 * std::min(c.Lx, c.Ly));
    c.sponge_rate = d.num("grid", "sponge_rate", Kind::None, 0.1);
    c.move_x = d.flag("grid", "move_x", true); c.move_y = d.flag("grid", "move_y", true);
    c.self_fields = d.flag("numerics", "self_fields", true);
    c.zeta_filter = d.flag("numerics", "zeta_filter", true);
    c.k_filter = d.flag("numerics", "k_filter", true);
    c.push_phase = d.num("numerics", "push_phase", Kind::Angle, 0.05);
    if (c.nx < 4 || c.ny < 4 || c.nzeta < 4) throw std::runtime_error("grid: nx, ny, nzeta must be >= 4");
    c.h = c.Lwin / (c.nzeta - 1);

    if (d.has_block("laser")) {
        c.laser_on = d.flag("laser", "enabled", true);
        c.a0 = d.num("laser", "a0", Kind::None);
        c.kL = c.lambda0 / d.num("laser", "wavelength", Kind::Length, c.lambda0);
        c.theta = d.num("laser", "angle", Kind::Angle, 0.0);
        c.tau = d.num("laser", "tau_fwhm", Kind::Time) / c.Tu;
        c.cep = d.num("laser", "cep", Kind::Angle, 0.0);
        c.pol = d.str("laser", "polarization", "s");
        if (c.pol != "s" && c.pol != "p" && c.pol != "circular")
            throw std::runtime_error("laser.polarization must be s, p or circular");
        c.start_fwhm = d.num("laser", "start_fwhm", Kind::None, 3.5);
        c.stop_fwhm = d.num("laser", "stop_fwhm", Kind::None, c.start_fwhm);   // 0: run to z_end
    }
    c.charge = d.num("bunch", "charge", Kind::Charge);
    c.gamma = d.num("bunch", "gamma", Kind::None);
    c.sigx = L("bunch", "sigma_x"); c.sigy = Ld("bunch", "sigma_y", c.sigx); c.sigz = L("bunch", "sigma_z");
    c.emitx = Ld("bunch", "emittance_x", 0); c.emity = Ld("bunch", "emittance_y", c.emitx);
    c.espread = d.num("bunch", "energy_spread", Kind::None, 0.0);
    c.zeta_center = Ld("bunch", "zeta_center", 0.75 * c.Lwin);
    c.nmacro = d.integer("bunch", "macroparticles", 100000);
    c.noise = d.str("bunch", "shot_noise", "random");
    if (c.noise != "random" && c.noise != "fawley") throw std::runtime_error("bunch.shot_noise must be random or fawley");
    c.noise_lambda = Ld("bunch", "noise_wavelength", 0);
    c.beamlet = int(d.integer("bunch", "beamlet", 4));
    c.seed = (unsigned long)d.integer("bunch", "seed", 1);
    if (c.noise == "fawley" && (c.noise_lambda <= 0 || c.beamlet < 3 || c.nmacro % c.beamlet))
        throw std::runtime_error("fawley noise: noise_wavelength > 0, beamlet >= 3, macroparticles a multiple of beamlet");
    c.dir = d.str("output", "dir", "out");
    c.every = int(d.integer("output", "every", 10));
    c.dump_particles = d.flag("output", "particles", true);
    d.check_unused();
    return c;
}

void Config::summary(FILE* f) const {
    const double um = Lu * 1e6;
    std::fprintf(f, "GEM-PIC-3D\n  lambda0 = %.4g um, code length = %.4g um, n_c = %.4g cm^-3\n", lambda0 * 1e6, um, nc * 1e-6);
    std::fprintf(f, "  box %d x %d x %d: Lx = %.4g um, Ly = %.4g um, window = %.4g um\n", nx, ny, nzeta, Lx * um, Ly * um, Lwin * um);
    std::fprintf(f, "  dx = %.4g um, dy = %.4g um, dzeta = %.4g um, dz = %.4g um, dz_push = %.4g um, z_end = %.4g mm\n",
                 Lx / nx * um, Ly / ny * um, h * um, dz * um, dz_push * um, z_end * Lu * 1e3);
    std::fprintf(f, "  sponge %.4g um, rate %.3g; transverse window follows bunch: x %s, y %s\n", sponge * um, sponge_rate,
                 move_x ? "yes" : "no", move_y ? "yes" : "no");
    if (laser_on)
        std::fprintf(f, "  laser: a0 = %g, lambda = %.4g um, angle = %.4g rad, %s, tau_FWHM = %.4g ps, start/stop %.3g/%.3g FWHM\n",
                     a0, lambda0 / kL * 1e6, theta, pol.c_str(), tau * Tu * 1e12, start_fwhm, stop_fwhm);
    else std::fprintf(f, "  laser: off\n");
    std::fprintf(f, "  bunch: Q = %.4g pC, gamma = %g, sigma = %.3g x %.3g x %.3g um, %ld macroparticles (%s noise)\n",
                 charge * 1e12, gamma, sigx * um, sigy * um, sigz * um, nmacro, noise.c_str());
    std::fprintf(f, "  push: RK4 in z, sub-step <= %.3g rad of laser phase\n", push_phase);
    std::fprintf(f, "  self fields %s, zeta filter %s, k filter %s\n", self_fields ? "on" : "off", zeta_filter ? "on" : "off",
                 k_filter ? "on" : "off");
}

// ---------------------------------------------------------------- laser
double Laser::envelope(double et) const {
    const double x = et - eta_c;
    return std::exp(-2 * std::log(2.0) * x * x / (tau * tau));
}

void Laser::field(double x, double z, double t, double E[3], double B[3]) const {
    E[0] = E[1] = E[2] = B[0] = B[1] = B[2] = 0;
    if (!on) return;
    const double xi = eta(x, z, t) - eta_c;
    if (std::fabs(xi) > 5 * tau) return;
    const double g = std::exp(-2 * std::log(2.0) * xi * xi / (tau * tau));
    const double gp = -4 * std::log(2.0) * xi / (tau * tau) * g;
    const double ph = kL * xi + cep, c = std::cos(ph), s = std::sin(ph);
    // dA/deta for A = a0 g cos(ph) along e1, and a0 g sin(ph) along e2 for circular polarization
    const double d1 = a0 * (gp * c - kL * g * s), d2 = a0 * (gp * s + kL * g * c);
    const double es[3] = {0, 1, 0}, ep[3] = {kz, 0, -kx};      // s and p unit vectors, both normal to khat
    for (int k = 0; k < 3; ++k) {
        if (pol == 0) E[k] = -d1 * es[k];
        else if (pol == 1) E[k] = -d1 * ep[k];
        else E[k] = -d1 * ep[k] - d2 * es[k];
    }
    const double kh[3] = {kx, 0, kz};
    cross(kh, E, B);
}

// ---------------------------------------------------------------- setup
Simulation::Simulation(const Config& c) : cfg(c), fft_(c.nx, c.ny) {
    N = cfg.nzeta; NH = N - 1; nk = cfg.nx * cfg.ny;
    dx = cfg.Lx / cfg.nx; dy = cfg.Ly / cfg.ny;
    kxd.resize(nk); kyd.resize(nk); kxs.resize(nk); kys.resize(nk); filt.resize(nk); mask.resize(nk);
    for (int ix = 0; ix < cfg.nx; ++ix)
        for (int iy = 0; iy < cfg.ny; ++iy) {
            const int i = ix * cfg.ny + iy;
            const int mx = ix <= cfg.nx / 2 ? ix : ix - cfg.nx, my = iy <= cfg.ny / 2 ? iy : iy - cfg.ny;
            kxs[i] = 2 * M_PI * mx / cfg.Lx; kys[i] = 2 * M_PI * my / cfg.Ly;
            kxd[i] = (2 * ix == cfg.nx) ? 0 : kxs[i];     // odd derivatives vanish at Nyquist
            kyd[i] = (2 * iy == cfg.ny) ? 0 : kys[i];
            double f = 1;
            if (cfg.k_filter) f = std::pow(std::cos(0.5 * kxs[i] * dx), 2) * std::pow(std::cos(0.5 * kys[i] * dy), 2);
            if (ix == 0 && iy == 0) f = 0;                    // no net current in a periodic box
            if (2 * ix == cfg.nx || 2 * iy == cfg.ny) f = 0;
            filt[i] = f;
            // sponge: graded damping in a layer of width `sponge` at the box edges
            auto prof = [&](double xpos, double Lb) {
                const double dd = std::min(xpos, Lb - xpos);
                if (cfg.sponge <= 0 || dd >= cfg.sponge) return 1.0;
                const double r = 1 - dd / cfg.sponge;
                return 1 - cfg.sponge_rate * r * r;
            };
            mask[i] = prof(ix * dx, cfg.Lx) * prof(iy * dy, cfg.Ly);
        }
    const size_t nf = size_t(N) * nk, nh = size_t(NH) * nk;
    Rx.assign(nf, 0); Ry.assign(nf, 0); Ez0.assign(nf, 0); Bz0.assign(nf, 0);
    j0x.assign(nh, 0); j0y.assign(nh, 0);
    F.assign(6 * nf, 0.f); jr.assign(3 * nh, 0.0);
    laser.on = cfg.laser_on; laser.a0 = cfg.a0; laser.kL = cfg.kL; laser.tau = cfg.tau; laser.cep = cfg.cep;
    laser.kx = std::sin(cfg.theta); laser.kz = std::cos(cfg.theta);
    laser.pol = cfg.pol == "s" ? 0 : cfg.pol == "p" ? 1 : 2;
    laser.eta_c = cfg.start_fwhm * cfg.tau;   // the bunch centre starts at eta = 0
}

void Simulation::load_bunch() {
    std::mt19937_64 rng(cfg.seed);
    std::normal_distribution<double> G(0, 1);
    const double Ne = cfg.charge / e_SI;
    W_macro_ne = Ne / cfg.nmacro;
    const double W = W_macro_ne / (cfg.nc * std::pow(cfg.Lu, 3));     // code weight
    const double p0 = std::sqrt(cfg.gamma * cfg.gamma - 1);
    const int M = cfg.noise == "fawley" ? cfg.beamlet : 1;
    const long nb = cfg.nmacro / M;
    parts.clear(); parts.reserve(cfg.nmacro);
    for (long b = 0; b < nb; ++b) {
        const double x = cfg.sigx * G(rng), y = cfg.sigy * G(rng), z = cfg.sigz * G(rng);
        const double px = cfg.sigx > 0 ? cfg.emitx / cfg.sigx * G(rng) : 0;
        const double py = cfg.sigy > 0 ? cfg.emity / cfg.sigy * G(rng) : 0;
        const double pz = p0 * (1 + cfg.espread * G(rng));
        std::vector<double> dz(M, 0.0);
        if (M > 1) {
            // beamlet: M particles spread evenly over noise_wavelength, displaced so that the bunching at the
            // harmonics h < M/2 equals that of the real electrons (Fawley, PRST-AB 5, 070701 (2002))
            const double Neb = Ne / nb, lam = cfg.noise_lambda;
            const double th0 = 2 * M_PI * std::uniform_real_distribution<double>(0, 1)(rng);
            std::vector<double> dth(M, 0.0);
            for (int hh = 1; 2 * hh < M; ++hh) {
                const cplx bh = std::sqrt(0.5 / Neb) * cplx(G(rng), G(rng));
                const cplx ch = -2.0 * I * bh / double(hh);
                for (int j = 0; j < M; ++j) dth[j] += std::real(ch * std::exp(-I * double(hh) * (th0 + 2 * M_PI * j / M)));
            }
            for (int j = 0; j < M; ++j) dz[j] = lam * ((th0 + 2 * M_PI * j / M + dth[j]) / (2 * M_PI) - 0.5);
        }
        for (int j = 0; j < M; ++j) {
            const double g = std::sqrt(1 + px * px + py * py + pz * pz);
            const double vx = px / g, vy = py / g, vz = pz / g;
            const double zz = z + dz[j];
            // particle at (x, y, zz) at t = 0 crosses the plane z = 0 at t = -zz / vz
            Particle p{};
            p.x = x - vx * zz / vz + 0.5 * cfg.Lx; p.y = y - vy * zz / vz + 0.5 * cfg.Ly;
            p.zeta = cfg.zeta_center + zz / vz;
            p.px = px; p.py = py; p.pz = pz; p.w = W; p.alive = 1;
            parts.push_back(p);
        }
    }
    X0 = -0.5 * cfg.Lx; Y0 = -0.5 * cfg.Ly;
}

void Simulation::init() {
    load_bunch();
    s = 0; istep = 0;
    if (!cfg.self_fields) return;
    deposit();
    comoving_field();
    march(0.0);        // level s = 0: E_z, B_z, L from R, and j0 <- deposited currents
}

// Co-moving field of the bunch: -(lap_perp + gamma^-2 d_zeta^2) psi = rho, R = (1 + beta) E_perp, E_perp = -grad psi.
void Simulation::comoving_field() {
    double sb = 0, sw = 0;
    for (auto& p : parts) if (p.alive) { sb += p.w * p.pz / std::sqrt(1 + p.px * p.px + p.py * p.py + p.pz * p.pz); sw += p.w; }
    const double beta = sw > 0 ? sb / sw : 1, g2 = 1 - beta * beta;   // gamma^-2
    // rho per half-cell (jz / beta) in spectral space, stored temporarily in Ez0[m]
    std::vector<cplx> tmp(nk);
    for (int m = 0; m < NH; ++m) {
        fft_.forward_pair(&jr[(2 * NH + m) * size_t(nk)], nullptr, &Ez0[size_t(m) * nk], nullptr);
        for (int i = 0; i < nk; ++i) Ez0[size_t(m) * nk + i] *= filt[i] / beta;
    }
    std::vector<cplx> rhs(N), cp(N), psi(N);
    const double off = g2 / (cfg.h * cfg.h);
    for (int i = 0; i < nk; ++i) {
        const double k2 = kxd[i] * kxd[i] + kyd[i] * kyd[i];
        for (int n = 0; n < N; ++n) {
            cplx r = 0; int c = 0;
            if (n >= 1) { r += Ez0[size_t(n - 1) * nk + i]; ++c; }
            if (n <= N - 2) { r += Ez0[size_t(n) * nk + i]; ++c; }
            rhs[n] = 0.5 * r;
        }
        // Thomas algorithm: diag = k2 + 2 off, off-diagonals -off, psi = 0 beyond the window
        const double dg = k2 + 2 * off;
        if (dg == 0) { for (int n = 0; n < N; ++n) psi[n] = 0; }
        else {
            cp[0] = -off / dg; psi[0] = rhs[0] / dg;
            for (int n = 1; n < N; ++n) {
                const cplx den = dg + off * cp[n - 1];
                cp[n] = -off / den; psi[n] = (rhs[n] + off * psi[n - 1]) / den;
            }
            for (int n = N - 2; n >= 0; --n) psi[n] -= cp[n] * psi[n + 1];
        }
        for (int n = 0; n < N; ++n) {
            Rx[size_t(n) * nk + i] = -(1 + beta) * I * kxd[i] * psi[n];
            Ry[size_t(n) * nk + i] = -(1 + beta) * I * kyd[i] * psi[n];
        }
    }
    std::fill(Ez0.begin(), Ez0.end(), cplx(0));
}

// ---------------------------------------------------------------- particles
void Simulation::gather(double x, double y, double zeta, double E[3], double B[3]) const {
    E[0] = E[1] = E[2] = B[0] = B[1] = B[2] = 0;
    if (!cfg.self_fields) return;
    double fz = zeta / cfg.h; int n = int(std::floor(fz));
    if (n < 0 || n >= N - 1) return;
    fz -= n;
    int ix[3], iy[3]; double wx[3], wy[3];
    tsc(x, dx, cfg.nx, ix, wx); tsc(y, dy, cfg.ny, iy, wy);
    for (int c = 0; c < 6; ++c) {
        const float* a = &F[(size_t(c) * N + n) * nk];
        const float* b = a + nk;
        double v = 0;
        for (int i = 0; i < 3; ++i)
            for (int j = 0; j < 3; ++j) {
                const int k = ix[i] * cfg.ny + iy[j];
                v += wx[i] * wy[j] * ((1 - fz) * a[k] + fz * b[k]);
            }
        (c < 3 ? E[c] : B[c - 3]) = v;
    }
}

double Simulation::particle_eta(const Particle& p) const {
    const double t = s - (p.zeta - cfg.zeta_center);
    return laser.eta(p.x + X0, s, t);
}

// Advance all particles from the plane z = s0 to z = s0 + ds along their characteristics with z as the
// independent variable (GT2): dr_perp/dz = p_perp/p_z, dzeta/dz = -(gamma - p_z)/p_z,
// dp/dz = q (gamma E + p x B)/p_z, by RK4 in sub-steps. The analytic laser is evaluated at the exact
// position and time t = z - zeta, the grid fields are those of level s0. Integrating the total force
// along z avoids the splitting error of Boris/Vay rotations when E and v x B of a co-propagating
// wave nearly cancel (|q dt B| >> gamma for steps on the slow scale).
namespace {
struct State { double x, y, zeta, px, py, pz; };
}
void Simulation::push(double s0, double ds_total) {
    const double zmax = (N - 1) * cfg.h, s1 = s0 + ds_total;
    long nlost = 0;
    auto rhs = [&](const State& Y, double z, State& d) -> bool {
        const double g = std::sqrt(1 + Y.px * Y.px + Y.py * Y.py + Y.pz * Y.pz);
        if (Y.pz < 0.3 * g) return false;                     // the z-advance needs v_z well above 0
        const double gm = (1 + Y.px * Y.px + Y.py * Y.py) / (g + Y.pz);   // gamma - p_z without cancellation
        double E[3], B[3], Ee[3], Be[3];
        gather(std::fmod(std::fmod(Y.x, cfg.Lx) + cfg.Lx, cfg.Lx), std::fmod(std::fmod(Y.y, cfg.Ly) + cfg.Ly, cfg.Ly), Y.zeta, E, B);
        laser.field(Y.x + X0, z, z - (Y.zeta - cfg.zeta_center), Ee, Be);
        for (int c = 0; c < 3; ++c) { E[c] += Ee[c]; B[c] += Be[c]; }
        const double ip = 1 / Y.pz;
        d.x = Y.px * ip; d.y = Y.py * ip; d.zeta = -gm * ip;
        d.px = q * (g * E[0] + Y.py * B[2] - Y.pz * B[1]) * ip;
        d.py = q * (g * E[1] + Y.pz * B[0] - Y.px * B[2]) * ip;
        d.pz = q * (g * E[2] + Y.px * B[1] - Y.py * B[0]) * ip;
        return true;
    };
    auto axpy = [](const State& a, double h, const State& b) {
        return State{a.x + h * b.x, a.y + h * b.y, a.zeta + h * b.zeta, a.px + h * b.px, a.py + h * b.py, a.pz + h * b.pz};
    };
#pragma omp parallel for reduction(+ : nlost) schedule(static)
    for (long ip = 0; ip < long(parts.size()); ++ip) {
        Particle& p = parts[ip];
        if (!p.alive) continue;
        State Y{p.x, p.y, p.zeta, p.px, p.py, p.pz};
        double z = s0;
        bool ok = true;
        while (ok && z < s1 - 1e-12 * std::fabs(s1)) {
            // sub-step: at most dz_push, and at most push_phase of the laser phase kL (gamma - p.khat)/p_z dz
            double ds = std::min(cfg.dz_push, s1 - z);
            if (laser.on) {
                const double g = std::sqrt(1 + Y.px * Y.px + Y.py * Y.py + Y.pz * Y.pz);
                const double hl = g - Y.px * laser.kx - Y.pz * laser.kz;
                const double lim = cfg.push_phase * Y.pz / (laser.kL * std::max(hl, 1e-12));
                if (lim < ds) ds = (s1 - z) / std::ceil((s1 - z) / lim);
            }
            State k1, k2, k3, k4;
            ok = rhs(Y, z, k1) && rhs(axpy(Y, 0.5 * ds, k1), z + 0.5 * ds, k2) &&
                 rhs(axpy(Y, 0.5 * ds, k2), z + 0.5 * ds, k3) && rhs(axpy(Y, ds, k3), z + ds, k4);
            if (!ok) break;
            const double w = ds / 6;
            Y.x += w * (k1.x + 2 * k2.x + 2 * k3.x + k4.x); Y.y += w * (k1.y + 2 * k2.y + 2 * k3.y + k4.y);
            Y.zeta += w * (k1.zeta + 2 * k2.zeta + 2 * k3.zeta + k4.zeta);
            Y.px += w * (k1.px + 2 * k2.px + 2 * k3.px + k4.px); Y.py += w * (k1.py + 2 * k2.py + 2 * k3.py + k4.py);
            Y.pz += w * (k1.pz + 2 * k2.pz + 2 * k3.pz + k4.pz);
            z += ds;
        }
        if (!ok) { p.alive = 0; ++nlost; continue; }
        p.x = Y.x; p.y = Y.y;          // continuous coordinates: the laser is not periodic in the box
        p.zeta = Y.zeta; p.px = Y.px; p.py = Y.py; p.pz = Y.pz;
        if (p.zeta < 0 || p.zeta >= zmax) { p.alive = 0; ++nlost; }
    }
    lost += nlost;
}

// Beam current of the plane z = s: a particle carries the charge q w / v_z per unit zeta, j = rho v.
// Half-cell m = [zeta_m, zeta_m+1] (NGP in zeta), TSC transverse; (1,2,1)/4 filter along zeta.
void Simulation::deposit() {
    std::fill(jr.begin(), jr.end(), 0.0);
    const double vol = cfg.h * dx * dy;
    for (auto& p : parts) {
        if (!p.alive) continue;
        const int m = int(std::floor(p.zeta / cfg.h));
        if (m < 0 || m >= NH) continue;
        const double g = std::sqrt(1 + p.px * p.px + p.py * p.py + p.pz * p.pz);
        const double vx = p.px / g, vy = p.py / g, vz = p.pz / g;
        const double f = q * p.w / vol;
        int ix[3], iy[3]; double wx[3], wy[3];
        tsc(p.x, dx, cfg.nx, ix, wx); tsc(p.y, dy, cfg.ny, iy, wy);
        double* jx = &jr[(size_t(0) * NH + m) * nk];
        double* jy = &jr[(size_t(1) * NH + m) * nk];
        double* jz = &jr[(size_t(2) * NH + m) * nk];
        for (int a = 0; a < 3; ++a)
            for (int b = 0; b < 3; ++b) {
                const int k = ix[a] * cfg.ny + iy[b];
                const double w = f * wx[a] * wy[b];
                jx[k] += w * vx / vz; jy[k] += w * vy / vz; jz[k] += w;
            }
    }
    if (cfg.zeta_filter) {
        std::vector<double> prev(nk), cur(nk);
        for (int c = 0; c < 3; ++c) {
            std::fill(prev.begin(), prev.end(), 0.0);
            for (int m = 0; m < NH; ++m) {
                double* a = &jr[(size_t(c) * NH + m) * nk];
                const double* nx = m + 1 < NH ? a + nk : nullptr;
                for (int i = 0; i < nk; ++i) {
                    cur[i] = a[i];
                    a[i] = 0.25 * prev[i] + 0.5 * a[i] + 0.25 * (nx ? nx[i] : 0.0);
                }
                std::swap(prev, cur);
            }
        }
    }
}

// ---------------------------------------------------------------- field march
// Level s1 from level s0, from the front slice (n = N-1, L = E_z = B_z = 0) to the rear.
// D = dz/2 (trapezoidal rule in s for R); D = 0 only evaluates E_z, B_z, L of the stored R.
// The transverse operators are multiplications by a = i kx, b = i ky, so every (kx, ky) is a scalar
// problem: B_z and E_z decouple exactly and follow in closed form, then R and L.
void Simulation::march(double D) {
    const double h = cfg.h;
    std::vector<cplx> upx(nk, 0), upy(nk, 0), upz(nk, 0), dnx(nk), dny(nk), dnz(nk);
    std::vector<cplx> pEz(nk, 0), pBz(nk, 0), pLx(nk, 0), pLy(nk, 0), pRx(nk, 0), pRy(nk, 0);
    std::vector<cplx> cEz(nk), cBz(nk), cLx(nk), cLy(nk), cRx(nk), cRy(nk), rbx(nk), rby(nk);
    std::vector<double> r1(nk), r2(nk), l1(nk), l2(nk), ez(nk), bz(nk);
    double energy = 0;
    auto spectrum = [&](int m, std::vector<cplx>& X, std::vector<cplx>& Y, std::vector<cplx>& Z) {
        fft_.forward_pair(&jr[(size_t(0) * NH + m) * nk], &jr[(size_t(1) * NH + m) * nk], X.data(), Y.data());
        fft_.forward_pair(&jr[(size_t(2) * NH + m) * nk], nullptr, Z.data(), nullptr);
        for (int i = 0; i < nk; ++i) { X[i] *= filt[i]; Y[i] *= filt[i]; Z[i] *= filt[i]; }
    };
    for (int n = N - 1; n >= 0; --n) {
        if (n >= 1) spectrum(n - 1, dnx, dny, dnz);
        else { std::fill(dnx.begin(), dnx.end(), 0); std::fill(dny.begin(), dny.end(), 0); std::fill(dnz.begin(), dnz.end(), 0); }
        const size_t o = size_t(n) * nk;
#pragma omp parallel for schedule(static)
        for (int i = 0; i < nk; ++i) {
            const cplx a = I * kxd[i], b = I * kyd[i];
            const double k2 = kxd[i] * kxd[i] + kyd[i] * kyd[i];
            // current at the slice, both half-cells, both levels (trapezoidal rule in s)
            cplx Jx = dnx[i] + upx[i], Jy = dny[i] + upy[i];
            if (n >= 1) { Jx += j0x[size_t(n - 1) * nk + i]; Jy += j0y[size_t(n - 1) * nk + i]; }
            if (n <= N - 2) { Jx += j0x[o + i]; Jy += j0y[o + i]; }
            const cplx Rsx = Rx[o + i] + D * (a * Ez0[o + i] + b * Bz0[o + i]) - 0.5 * D * Jx;
            const cplx Rsy = Ry[o + i] + D * (b * Ez0[o + i] - a * Bz0[o + i]) - 0.5 * D * Jy;
            if (n == N - 1) {
                cEz[i] = cBz[i] = cLx[i] = cLy[i] = 0; cRx[i] = Rsx; cRy[i] = Rsy;
                continue;
            }
            const cplx Lsx = pLx[i] - 0.25 * h * (a * pEz[i] - b * pBz[i]) - 0.5 * h * upx[i];
            const cplx Lsy = pLy[i] - 0.25 * h * (a * pBz[i] + b * pEz[i]) - 0.5 * h * upy[i];
            const cplx Cp = 0.5 * (a * (pRy[i] + pLy[i]) - b * (pRx[i] + pLx[i]));
            const cplx Qp = -0.5 * a * (pRx[i] - pLx[i]) + 0.5 * b * (pLy[i] - pRy[i]);
            const cplx SB = a * Rsy + a * Lsy - b * Rsx - b * Lsx;
            const cplx SE = -a * Rsx + a * Lsx + b * Lsy - b * Rsy;
            const double den = 1 + 0.25 * h * (D + 0.25 * h) * k2;
            const cplx Bz = (pBz[i] - 0.5 * h * Cp - 0.25 * h * SB) / den;
            const cplx Ez = (pEz[i] - 0.5 * h * Qp - 0.25 * h * SE - h * upz[i]) / den;
            cEz[i] = Ez; cBz[i] = Bz;
            cRx[i] = Rsx + D * (a * Ez + b * Bz); cRy[i] = Rsy + D * (b * Ez - a * Bz);
            cLx[i] = Lsx - 0.25 * h * (a * Ez - b * Bz); cLy[i] = Lsy - 0.25 * h * (a * Bz + b * Ez);
        }
        // real space: sponge on R, then E and B of the slice for the particle push
        if (cfg.sponge > 0) {
            // The sponge acts on R - R_b only, where R_b = 2 E_perp, k^2 psi = rho, is the bound field of the
            // charge of this slice (v = c limit, local in the slice). Damping the full R would drain the
            // bunch through its own Coulomb field at the box edges, like a resistive wall.
            for (int i = 0; i < nk; ++i) {
                const double k2 = kxd[i] * kxd[i] + kyd[i] * kyd[i];
                const cplx rho = 0.5 * (upz[i] + dnz[i]);
                const cplx f = k2 > 0 ? -2.0 * rho / k2 : cplx(0);
                rbx[i] = I * kxd[i] * f; rby[i] = I * kyd[i] * f;
                cRx[i] -= rbx[i]; cRy[i] -= rby[i];
            }
            fft_.inverse_pair(cRx.data(), cRy.data(), r1.data(), r2.data());
            for (int i = 0; i < nk; ++i) { r1[i] *= mask[i]; r2[i] *= mask[i]; }
            fft_.forward_pair(r1.data(), r2.data(), cRx.data(), cRy.data());
            for (int i = 0; i < nk; ++i) { cRx[i] += rbx[i]; cRy[i] += rby[i]; }
        }
        fft_.inverse_pair(cRx.data(), cRy.data(), r1.data(), r2.data());
        fft_.inverse_pair(cLx.data(), cLy.data(), l1.data(), l2.data());
        fft_.inverse_pair(cEz.data(), cBz.data(), ez.data(), bz.data());
        float* Fe = F.data();
        for (int i = 0; i < nk; ++i) {
            const double Ex = 0.5 * (r1[i] + l1[i]), Ey = 0.5 * (r2[i] + l2[i]);
            const double By = 0.5 * (r1[i] - l1[i]), Bx = 0.5 * (l2[i] - r2[i]);
            Fe[(0 * size_t(N) + n) * nk + i] = float(Ex); Fe[(1 * size_t(N) + n) * nk + i] = float(Ey);
            Fe[(2 * size_t(N) + n) * nk + i] = float(ez[i]);
            Fe[(3 * size_t(N) + n) * nk + i] = float(Bx); Fe[(4 * size_t(N) + n) * nk + i] = float(By);
            Fe[(5 * size_t(N) + n) * nk + i] = float(bz[i]);
            const double w = (n == 0 || n == N - 1) ? 0.5 : 1.0;
            energy += w * 0.5 * (Ex * Ex + Ey * Ey + ez[i] * ez[i] + Bx * Bx + By * By + bz[i] * bz[i]);
        }
        std::copy(cRx.begin(), cRx.end(), Rx.begin() + o); std::copy(cRy.begin(), cRy.end(), Ry.begin() + o);
        std::copy(cEz.begin(), cEz.end(), Ez0.begin() + o); std::copy(cBz.begin(), cBz.end(), Bz0.begin() + o);
        if (n <= N - 2) {   // half-cell n is no longer needed at level s0
            std::copy(upx.begin(), upx.end(), j0x.begin() + o); std::copy(upy.begin(), upy.end(), j0y.begin() + o);
        }
        std::swap(pEz, cEz); std::swap(pBz, cBz); std::swap(pLx, cLx); std::swap(pLy, cLy);
        std::swap(pRx, cRx); std::swap(pRy, cRy);
        std::swap(upx, dnx); std::swap(upy, dny); std::swap(upz, dnz);
    }
    field_energy = energy * dx * dy * h;
}

// ---------------------------------------------------------------- time step
void Simulation::step() {
    push(s, cfg.dz);
    s += cfg.dz; ++istep;
    if (cfg.self_fields) { deposit(); march(0.5 * cfg.dz); }
    shift_window();
}

// The transverse window follows the bunch centroid in whole cells; a shift is exact in Fourier space.
void Simulation::shift_window() {
    double sx = 0, sy = 0, sw = 0;
    for (auto& p : parts) if (p.alive) { sx += p.w * p.x; sy += p.w * p.y; sw += p.w; }
    if (sw == 0) return;
    const int nxs = cfg.move_x ? int(std::lround((sx / sw - 0.5 * cfg.Lx) / dx)) : 0;
    const int nys = cfg.move_y ? int(std::lround((sy / sw - 0.5 * cfg.Ly) / dy)) : 0;
    if (std::abs(nxs) < 2 && std::abs(nys) < 2) return;
    X0 += nxs * dx; Y0 += nys * dy;
    for (auto& p : parts) {
        p.x -= nxs * dx; p.y -= nys * dy;
    }
    if (!cfg.self_fields) return;
    std::vector<cplx> ph(nk);
    for (int i = 0; i < nk; ++i) ph[i] = std::exp(I * (kxs[i] * nxs * dx + kys[i] * nys * dy));
    auto rot = [&](std::vector<cplx>& A) { for (size_t j = 0; j < A.size(); ++j) A[j] *= ph[j % nk]; };
    rot(Rx); rot(Ry); rot(Ez0); rot(Bz0); rot(j0x); rot(j0y);
    std::vector<float> tmp(nk);
    for (size_t pl = 0; pl < 6 * size_t(N); ++pl) {
        float* a = &F[pl * nk];
        for (int ix = 0; ix < cfg.nx; ++ix)
            for (int iy = 0; iy < cfg.ny; ++iy)
                tmp[ix * cfg.ny + iy] = a[((ix + nxs) % cfg.nx + cfg.nx) % cfg.nx * cfg.ny + ((iy + nys) % cfg.ny + cfg.ny) % cfg.ny];
        std::copy(tmp.begin(), tmp.end(), a);
    }
}

bool Simulation::finished() const {
    if (s >= cfg.z_end - 1e-9) return true;
    if (!laser.on || cfg.stop_fwhm <= 0) return false;
    double se = 0, sw = 0;
    for (auto& p : parts) if (p.alive) { se += p.w * particle_eta(p); sw += p.w; }
    return sw > 0 && se / sw - laser.eta_c > cfg.stop_fwhm * laser.tau;
}

// ---------------------------------------------------------------- output
void Simulation::diagnostics(FILE* f, bool header) {
    if (header)
        std::fprintf(f, "# step  z[mm]  N_alive  <x>[um] <y>[um] <zeta>[um]  rms_x rms_y rms_zeta[um]  <px> <py> <pz> <gamma> "
                        "<p.khat>  rms_pz  eta_rel[FWHM] a_env  W_field[J] W_kin[J]\n");
    double S[16] = {0};
    for (auto& p : parts) {
        if (!p.alive) continue;
        const double g = std::sqrt(1 + p.px * p.px + p.py * p.py + p.pz * p.pz);
        const double X = p.x + X0, Y = p.y + Y0;
        const double v[] = {1, X, Y, p.zeta, X * X, Y * Y, p.zeta * p.zeta, p.px, p.py, p.pz, g,
                            p.px * laser.kx + p.pz * laser.kz, p.pz * p.pz, particle_eta(p)};
        for (int k = 0; k < 14; ++k) S[k] += p.w * v[k];
    }
    const double w = S[0] > 0 ? S[0] : 1;
    auto m = [&](int k) { return S[k] / w; };
    auto rms = [&](int k, int k2) { return std::sqrt(std::max(0.0, m(k2) - m(k) * m(k))); };
    const double um = cfg.Lu * 1e6;
    long alive = 0; for (auto& p : parts) alive += p.alive;
    const double eta_rel = laser.on ? (m(13) - laser.eta_c) / laser.tau : 0;
    const double aenv = laser.on ? laser.a0 * laser.envelope(m(13)) : 0;
    const double Efield = 1.0 * me_SI * c_SI * c_SI / (cfg.Lu * e_SI);     // field unit [V/m] (omega0/c = 1/Lu)
    const double Wf = field_energy * eps0 * Efield * Efield * std::pow(cfg.Lu, 3);
    const double Wk = alive * W_macro_ne * (m(10) - 1) * me_SI * c_SI * c_SI;
    std::fprintf(f, "%6ld %10.5f %8ld %10.4f %10.4f %10.4f %9.4f %9.4f %9.4f %12.6e %12.6e %14.8e %14.8e %14.8e %11.4e %8.4f %8.4f %11.4e %11.4e\n",
                 istep, s * cfg.Lu * 1e3, alive, m(1) * um, m(2) * um, m(3) * um, rms(1, 4) * um, rms(2, 5) * um,
                 rms(3, 6) * um, m(7), m(8), m(9), m(10), m(11), rms(9, 12), eta_rel, aenv, Wf, Wk);
    std::fflush(f);
}

void Simulation::dump_particles(const std::string& file) const {
    std::ofstream o(file);
    o.precision(10);
    o << "# x[um] y[um] zeta[um] px py pz (m c), z = " << s * cfg.Lu * 1e3 << " mm\n";
    const double um = cfg.Lu * 1e6;
    for (auto& p : parts)
        if (p.alive) o << (p.x + X0) * um << ' ' << (p.y + Y0) * um << ' ' << p.zeta * um << ' ' << p.px << ' ' << p.py << ' ' << p.pz << '\n';
}
