#include "sim.hpp"
#include <algorithm>
#include <cmath>
#include <complex>
#include <fstream>
#include <random>
#include <stdexcept>

namespace {
constexpr double c_SI = 299792458.0, e_SI = 1.602176634e-19, me_SI = 9.1093837015e-31, eps0 = 8.8541878128e-12;
constexpr double q = -1.0;                       // electron charge in units of e
enum Tag { TAG_HAND = 11, TAG_CUR = 12, TAG_BND = 13, TAG_GHOST = 14, TAG_TRIF = 15, TAG_TRIB = 16 };
}  // namespace

// ---------------------------------------------------------------- configuration
Config Config::base_units(double lambda0) {
    Config c;
    c.lambda0 = lambda0; c.Lu = lambda0 / (2 * M_PI); c.Tu = c.Lu / c_SI;
    const double w0 = c_SI / c.Lu;
    c.nc = eps0 * me_SI * w0 * w0 / (e_SI * e_SI);
    return c;
}

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
    c.window_lag = int(d.integer("grid", "window_lag", 4));
    if (c.window_lag < 1) throw std::runtime_error("grid.window_lag must be >= 1");
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

void Config::summary(FILE* f, int ranks) const {
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
    std::fprintf(f, "  parallel: %s, %d MPI rank(s), transverse window lag %d steps\n", GEMPIC_GPU ? "CUDA" : "OpenMP", ranks, window_lag);
    std::fprintf(f, "  self fields %s, zeta filter %s, k filter %s\n", self_fields ? "on" : "off", zeta_filter ? "on" : "off",
                 k_filter ? "on" : "off");
}

// ---------------------------------------------------------------- setup
Simulation::Simulation(const Config& c, Comm& cm) : cfg(c), comm(cm) {
    N = cfg.nzeta; nk = cfg.nx * cfg.ny;
    dx = cfg.Lx / cfg.nx; dy = cfg.Ly / cfg.ny;
    // blocks of consecutive slices, rank P-1 at the front of the window
    const int P = comm.size, base = N / P, rem = N % P;
    a = 0;
    for (int r = 0; r < comm.rank; ++r) a += base + (r < rem);
    ns = base + (comm.rank < rem);
    b = a + ns;
    if (P > 1 && base < 4) throw std::runtime_error("MPI: at least 4 slices per rank are needed");

    std::vector<double> hkx(nk), hky(nk), hsx(nk), hsy(nk), hf(nk), hm(nk);
    for (int ixx = 0; ixx < cfg.nx; ++ixx)
        for (int iyy = 0; iyy < cfg.ny; ++iyy) {
            const int i = ixx * cfg.ny + iyy;
            const int mx = ixx <= cfg.nx / 2 ? ixx : ixx - cfg.nx, my = iyy <= cfg.ny / 2 ? iyy : iyy - cfg.ny;
            hsx[i] = 2 * M_PI * mx / cfg.Lx; hsy[i] = 2 * M_PI * my / cfg.Ly;
            hkx[i] = (2 * ixx == cfg.nx) ? 0 : hsx[i];     // odd derivatives vanish at Nyquist
            hky[i] = (2 * iyy == cfg.ny) ? 0 : hsy[i];
            double f = 1;
            if (cfg.k_filter) f = std::pow(std::cos(0.5 * hsx[i] * dx), 2) * std::pow(std::cos(0.5 * hsy[i] * dy), 2);
            if (ixx == 0 && iyy == 0) f = 0;                // no net current in a periodic box
            if (2 * ixx == cfg.nx || 2 * iyy == cfg.ny) f = 0;
            hf[i] = f;
            auto prof = [&](double xpos, double Lb) {
                const double dd = std::min(xpos, Lb - xpos);
                if (cfg.sponge <= 0 || dd >= cfg.sponge) return 1.0;
                const double r = 1 - dd / cfg.sponge;
                return 1 - cfg.sponge_rate * r * r;
            };
            hm[i] = prof(ixx * dx, cfg.Lx) * prof(iyy * dy, cfg.Ly);
        }
    kxd.upload(hkx); kyd.upload(hky); kxs.upload(hsx); kys.upload(hsy); filt.upload(hf); mask.upload(hm);

    const size_t pl = size_t(nk);
    for (Buf<cpx>* p : {&Rx, &Ry, &Ez, &Bz, &Lx, &Ly}) p->resize(ns * pl);
    bnd.resize(6 * pl);
    for (Buf<cpx>* p : {&j0x, &j0y, &jkx, &jky, &jkz, &S1, &S2}) p->resize((ns + 1) * pl);
    jw.resize(3 * (ns + 3) * pl);
    F.resize(6 * (ns + 2) * pl);
    fft_ = std::make_unique<FftMany>(cfg.nx, cfg.ny);

    laser.on = cfg.laser_on; laser.a0 = cfg.a0; laser.kL = cfg.kL; laser.tau = cfg.tau; laser.cep = cfg.cep;
    laser.kx = std::sin(cfg.theta); laser.kz = std::cos(cfg.theta);
    laser.pol = cfg.pol == "s" ? 0 : cfg.pol == "p" ? 1 : 2;
    laser.eta_c = cfg.start_fwhm * cfg.tau;   // the bunch centre starts at eta = 0
}

std::vector<Particle> Simulation::host_particles() const {
    std::vector<Particle> v(static_cast<size_t>(np));
    parts.to_host(v.data(), 0, size_t(np));
    return v;
}
void Simulation::set_particles(const std::vector<Particle>& v) {
    np = long(v.size());
    if (parts.size() < v.size() || parts.size() > 2 * v.size() + 1024) parts.resize(v.size() + v.size() / 4 + 64);
    parts.from_host(v.data(), 0, v.size());
}

// All ranks draw the same bunch; each keeps the particles of its half-cells.
void Simulation::load_bunch() {
    using cplx = std::complex<double>;
    const cplx I(0, 1);
    std::mt19937_64 rng(cfg.seed);
    std::normal_distribution<double> G(0, 1);
    const double Ne = cfg.charge / e_SI;
    W_macro_ne = cfg.nmacro > 0 ? Ne / cfg.nmacro : 0;
    const double W = W_macro_ne / (cfg.nc * std::pow(cfg.Lu, 3));     // code weight
    const double p0 = std::sqrt(cfg.gamma * cfg.gamma - 1);
    const int M = cfg.noise == "fawley" ? cfg.beamlet : 1;
    const long nb = cfg.nmacro / M;
    const int hlo = comm.rank == 0 ? 0 : a + 1, hhi = comm.rank == comm.size - 1 ? N - 2 : b;
    std::vector<Particle> mine;
    for (long ib = 0; ib < nb; ++ib) {
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
            // the particle at (x, y, zz) at t = 0 crosses the plane z = 0 at t = -zz / vz
            Particle p{};
            p.x = x - vx * zz / vz + 0.5 * cfg.Lx; p.y = y - vy * zz / vz + 0.5 * cfg.Ly;
            p.zeta = cfg.zeta_center + zz / vz;
            p.px = px; p.py = py; p.pz = pz; p.w = W; p.alive = 1;
            const int hc = int(std::floor(p.zeta / cfg.h));
            if (p.zeta < 0 || hc > N - 2) { if (comm.rank == 0) ++lost_total; continue; }
            if (hc >= hlo && hc <= hhi) mine.push_back(p);
        }
    }
    set_particles(mine);
    X0 = -0.5 * cfg.Lx; Y0 = -0.5 * cfg.Ly;
}

// ---------------------------------------------------------------- particles (device code)
namespace {
struct PushCtx {
    const float* F; int a, ns, N, nx, ny, nk, self; double dx, dy, h, Lx, Ly;
    Laser laser; double X0, zc, dz_push, push_phase, zhand, zmax, s1;
};
struct State { double x, y, zeta, px, py, pz; };

// Quadratic (TSC) weights on a periodic grid with nodes at i*d.
HD inline void tsc(double x, double d, int n, int idx[3], double w[3]) {
    const double xi = x / d;
    const int i0 = int(::floor(xi + 0.5));
    const double f = xi - i0;
    w[0] = 0.5 * (0.5 - f) * (0.5 - f); w[1] = 0.75 - f * f; w[2] = 0.5 * (0.5 + f) * (0.5 + f);
    for (int k = 0; k < 3; ++k) idx[k] = ((i0 - 1 + k) % n + n) % n;
}
// Grid fields of level s0 at (x, y, zeta): TSC transverse, linear in zeta between the slices of this rank
// (own slices and the two ghost slices of the rank above).
HD inline void gather(const PushCtx& c, double x, double y, double zeta, double E[3], double B[3]) {
    E[0] = E[1] = E[2] = B[0] = B[1] = B[2] = 0;
    if (!c.self) return;
    double fz = zeta / c.h;
    const int n = int(::floor(fz));
    const int fl = n - c.a;
    if (fl < 0 || fl > c.ns) return;
    fz -= n;
    int ix[3], iy[3]; double wx[3], wy[3];
    tsc(x, c.dx, c.nx, ix, wx); tsc(y, c.dy, c.ny, iy, wy);
    for (int comp = 0; comp < 6; ++comp) {
        const float* p0 = c.F + (size_t(comp) * (c.ns + 2) + fl) * c.nk;
        const float* p1 = p0 + c.nk;
        double v = 0;
        for (int i = 0; i < 3; ++i)
            for (int j = 0; j < 3; ++j) {
                const int k = ix[i] * c.ny + iy[j];
                v += wx[i] * wy[j] * ((1 - fz) * p0[k] + fz * p1[k]);
            }
        if (comp < 3) E[comp] = v; else B[comp - 3] = v;
    }
}
// Characteristics with z as the independent variable: dr/dz = p/p_z, dzeta/dz = -(gamma - p_z)/p_z,
// dp/dz = q (gamma E + p x B)/p_z, with gamma - p_z = (1 + p_perp^2)/(gamma + p_z).
HD inline bool rhs(const PushCtx& c, const State& Y, double z, State& d) {
    const double g = ::sqrt(1 + Y.px * Y.px + Y.py * Y.py + Y.pz * Y.pz);
    if (Y.pz < 0.3 * g) return false;
    const double gm = (1 + Y.px * Y.px + Y.py * Y.py) / (g + Y.pz);
    double E[3], B[3], Ee[3], Be[3];
    gather(c, ::fmod(::fmod(Y.x, c.Lx) + c.Lx, c.Lx), ::fmod(::fmod(Y.y, c.Ly) + c.Ly, c.Ly), Y.zeta, E, B);
    c.laser.field(Y.x + c.X0, z, z - (Y.zeta - c.zc), Ee, Be);
    for (int k = 0; k < 3; ++k) { E[k] += Ee[k]; B[k] += Be[k]; }
    const double ip = 1 / Y.pz;
    d.x = Y.px * ip; d.y = Y.py * ip; d.zeta = -gm * ip;
    d.px = q * (g * E[0] + Y.py * B[2] - Y.pz * B[1]) * ip;
    d.py = q * (g * E[1] + Y.pz * B[0] - Y.px * B[2]) * ip;
    d.pz = q * (g * E[2] + Y.px * B[1] - Y.py * B[0]) * ip;
    return true;
}
HD inline State axpy(const State& a, double h, const State& b) {
    return State{a.x + h * b.x, a.y + h * b.y, a.zeta + h * b.zeta, a.px + h * b.px, a.py + h * b.py, a.pz + h * b.pz};
}
// RK4 from z0 to the plane s1. A sub-step is at most dz_push and at most push_phase of the laser phase.
// A particle that enters the half-cells of the rank below (zeta < zhand) at the start of a sub-step is
// handed over with the rest of the step.
HD inline void push_one(const PushCtx& c, Particle& p, double z0) {
    State Y{p.x, p.y, p.zeta, p.px, p.py, p.pz};
    double z = z0;
    while (z < c.s1 - 1e-12 * ::fabs(c.s1)) {
        if (Y.zeta < c.zhand) {
            p.x = Y.x; p.y = Y.y; p.zeta = Y.zeta; p.px = Y.px; p.py = Y.py; p.pz = Y.pz;
            p.alive = 2; p.rem = c.s1 - z;
            return;
        }
        double ds = c.dz_push < c.s1 - z ? c.dz_push : c.s1 - z;
        if (c.laser.on) {
            const double g = ::sqrt(1 + Y.px * Y.px + Y.py * Y.py + Y.pz * Y.pz);
            const double hl = g - Y.px * c.laser.kx - Y.pz * c.laser.kz;
            const double lim = c.push_phase * Y.pz / (c.laser.kL * (hl > 1e-12 ? hl : 1e-12));
            if (lim < ds) ds = (c.s1 - z) / ::ceil((c.s1 - z) / lim);
        }
        State k1, k2, k3, k4;
        if (!(rhs(c, Y, z, k1) && rhs(c, axpy(Y, 0.5 * ds, k1), z + 0.5 * ds, k2) &&
              rhs(c, axpy(Y, 0.5 * ds, k2), z + 0.5 * ds, k3) && rhs(c, axpy(Y, ds, k3), z + ds, k4))) {
            p.alive = 0;
            return;
        }
        const double w = ds / 6;
        Y.x += w * (k1.x + 2 * k2.x + 2 * k3.x + k4.x); Y.y += w * (k1.y + 2 * k2.y + 2 * k3.y + k4.y);
        Y.zeta += w * (k1.zeta + 2 * k2.zeta + 2 * k3.zeta + k4.zeta);
        Y.px += w * (k1.px + 2 * k2.px + 2 * k3.px + k4.px); Y.py += w * (k1.py + 2 * k2.py + 2 * k3.py + k4.py);
        Y.pz += w * (k1.pz + 2 * k2.pz + 2 * k3.pz + k4.pz);
        z += ds;
    }
    p.x = Y.x; p.y = Y.y; p.zeta = Y.zeta; p.px = Y.px; p.py = Y.py; p.pz = Y.pz;   // continuous coordinates
    p.alive = (p.zeta < 0 || p.zeta >= c.zmax) ? 0 : 1;
    // a particle that ends the step in the half-cells of the rank below belongs to that rank
    if (p.alive == 1 && p.zeta < c.zhand) { p.alive = 2; p.rem = 0; }
}
}  // namespace

// Push the particles [i0, i1) to the plane s1: own particles from s0, adopted ones (alive = 3) for their rest.
void Simulation::push(double s0, double s1, long i0, long i1) {
    PushCtx c;
    c.F = F.data(); c.a = a; c.ns = ns; c.N = N; c.nx = cfg.nx; c.ny = cfg.ny; c.nk = nk; c.self = cfg.self_fields;
    c.dx = dx; c.dy = dy; c.h = cfg.h; c.Lx = cfg.Lx; c.Ly = cfg.Ly; c.laser = laser; c.X0 = X0;
    c.zc = cfg.zeta_center; c.dz_push = cfg.dz_push; c.push_phase = cfg.push_phase;
    c.zhand = comm.rank == 0 ? -1e300 : (a + 1) * cfg.h;
    c.zmax = (N - 1) * cfg.h; c.s1 = s1;
    Particle* P = parts.data();
    par_for(i1 - i0, [=] HD(long k) {
        Particle& p = P[i0 + k];
        if (p.alive == 1) push_one(c, p, s0);
        else if (p.alive == 3) push_one(c, p, s1 - p.rem);
    });
    device_sync();
}

// Hand the particles that entered the half-cells of the rank below to it, adopt those of the rank above
// and finish their push. Lost particles are removed here.
void Simulation::exchange_particles(double s1) {
    if (comm.size == 1) return;
    std::vector<Particle> all = host_particles(), keep, hand;
    keep.reserve(all.size());
    for (auto& p : all) {
        if (p.alive == 1) keep.push_back(p);
        else if (p.alive == 2) hand.push_back(p);
        else ++lost_total;
    }
    if (comm.rank > 0) comm.isend(comm.rank - 1, TAG_HAND, hand.data(), hand.size());
    else if (!hand.empty()) throw std::runtime_error("handover on rank 0");
    const long n_own = long(keep.size());
    if (comm.rank < comm.size - 1) {
        std::vector<Particle> got = comm.recv_vec<Particle>(comm.rank + 1, TAG_HAND);
        for (auto& p : got) { p.alive = 3; keep.push_back(p); }
    }
    set_particles(keep);
    if (np > n_own) push(s1, s1, n_own, np);
    for (auto& p : host_particles())
        if (p.alive == 2) throw std::runtime_error("an adopted particle crossed a whole block in one step (blocks too small)");
}

// ---------------------------------------------------------------- currents
// Deposit (a particle on the plane z = s carries q w / v_z per unit zeta, j = q w v / v_z: NGP over the
// half-cell in zeta, TSC transverse), exchange of the half-cells at the block edges, (1,2,1)/4 filter
// along zeta, spectra with the transverse filter and without the k = 0 component.
void Simulation::currents() {
    jw.zero();
    const int nh = ns + 3, a_ = a, nx = cfg.nx, ny = cfg.ny, nkk = nk;
    const double h = cfg.h, dx_ = dx, dy_ = dy, vol = cfg.h * dx * dy;
    double* J = jw.data();
    const Particle* P = parts.data();
    par_for(np, [=] HD(long ip) {
        const Particle& p = P[ip];
        if (p.alive != 1) return;
        const int hw = int(::floor(p.zeta / h)) - (a_ - 2);
        if (hw < 0 || hw >= nh) return;
        const double g = ::sqrt(1 + p.px * p.px + p.py * p.py + p.pz * p.pz);
        const double vx = p.px / g, vy = p.py / g, vz = p.pz / g;
        const double f = q * p.w / vol;
        int ix[3], iy[3]; double wx[3], wy[3];
        tsc(p.x, dx_, nx, ix, wx); tsc(p.y, dy_, ny, iy, wy);
        for (int u = 0; u < 3; ++u)
            for (int v = 0; v < 3; ++v) {
                const long k = ix[u] * ny + iy[v];
                const double w = f * wx[u] * wy[v];
                atomic_add(&J[(0L * nh + hw) * nkk + k], w * vx / vz);
                atomic_add(&J[(1L * nh + hw) * nkk + k], w * vy / vz);
                atomic_add(&J[(2L * nh + hw) * nkk + k], w);
            }
    });
    device_sync();
    // the top three half-cells of this block are the bottom three ghosts of the block above
    const size_t pl = size_t(nk);
    if (comm.rank < comm.size - 1) {
        std::vector<double> buf(9 * pl);
        for (int comp = 0; comp < 3; ++comp) jw.to_host(&buf[comp * 3 * pl], (size_t(comp) * nh + ns) * pl, 3 * pl);
        comm.isend(comm.rank + 1, TAG_CUR, buf.data(), buf.size());
    }
    if (comm.rank > 0) {
        std::vector<double> buf(9 * pl);
        comm.recv(comm.rank - 1, TAG_CUR, buf.data(), buf.size());
        for (int comp = 0; comp < 3; ++comp) jw.from_host(&buf[comp * 3 * pl], (size_t(comp) * nh + 0) * pl, 3 * pl);
    }
    // filter along zeta into the half-cells a-1 .. b-1; jx + i jy packed into S1, jz into jkz
    const bool zf = cfg.zeta_filter;
    const int Ng = N;
    cpx* s1 = S1.data(); cpx* jz = jkz.data();
    par_for(long(ns + 1) * nk, [=] HD(long idx) {
        const int hk = int(idx / nkk);
        const long i = idx % nkk;
        const int hc = a_ - 1 + hk, hw = hk + 1;
        double v[3] = {0, 0, 0};
        if (hc >= 0 && hc <= Ng - 2)
            for (int comp = 0; comp < 3; ++comp) {
                const double* Jc = J + size_t(comp) * nh * nkk;
                v[comp] = zf ? 0.25 * Jc[(hw - 1) * nkk + i] + 0.5 * Jc[hw * nkk + i] + 0.25 * Jc[(hw + 1) * nkk + i]
                             : Jc[hw * nkk + i];
            }
        s1[idx] = C(v[0], v[1]);
        jz[idx] = C(v[2], 0);
    });
    fft_->run(S1.data(), ns + 1, false);
    fft_->run(jkz.data(), ns + 1, false);
    // unpack the pair, apply the transverse filter (zero at k = 0 and Nyquist)
    cpx* jx = jkx.data(); cpx* jy = jky.data();
    const double* fl = filt.data();
    par_for(long(ns + 1) * nk, [=] HD(long idx) {
        const long base = idx - idx % nkk;
        const long i = idx % nkk;
        const int ixx = int(i / ny), iyy = int(i % ny);
        const long m = base + long((nx - ixx) % nx) * ny + (ny - iyy) % ny;
        const cpx z = s1[idx], zm = conj(s1[m]);
        const cpx A = 0.5 * (z + zm), Bv = 0.5 * C((z - zm).im, -(z - zm).re);
        jx[idx] = fl[i] * A; jy[idx] = fl[i] * Bv; jz[idx] = fl[i] * jz[idx];
    });
    device_sync();
}

// ---------------------------------------------------------------- field march (device code)
namespace {
struct MarchCtx {
    cpx *Rx, *Ry, *Ez, *Bz, *Lx, *Ly;
    const cpx *bnd, *j0x, *j0y, *jkx, *jky, *jkz;
    const double *kx, *ky;
    double D, h; int a, b, N; long nk;
};
// Slice n of level s1 for the mode i: R by the trapezoidal rule in s, L, E_z, B_z by the trapezoidal rule
// over the half-cell [n, n+1]; B_z and E_z decouple and follow in closed form, then R and L.
// The values of slice n+1 (level s1) come from the arrays or, at the top of the block, from `bnd`.
HD inline void march_point(const MarchCtx& c, long i, int n) {
    const int l = n - c.a;
    const long o = long(l) * c.nk + i;
    const cpx a = ix(c.kx[i]), b = ix(c.ky[i]);
    const double k2 = c.kx[i] * c.kx[i] + c.ky[i] * c.ky[i], h = c.h, D = c.D;
    const long od = long(l) * c.nk + i, ou = long(l + 1) * c.nk + i;     // half-cells n-1 and n
    const cpx Jx = c.j0x[od] + c.j0x[ou] + c.jkx[od] + c.jkx[ou];
    const cpx Jy = c.j0y[od] + c.j0y[ou] + c.jky[od] + c.jky[ou];
    const cpx Rsx = c.Rx[o] + D * (a * c.Ez[o] + b * c.Bz[o]) - 0.5 * D * Jx;
    const cpx Rsy = c.Ry[o] + D * (b * c.Ez[o] - a * c.Bz[o]) - 0.5 * D * Jy;
    if (n == c.N - 1) {             // front of the window: L = E_z = B_z = 0
        c.Rx[o] = Rsx; c.Ry[o] = Rsy; c.Ez[o] = c.Bz[o] = c.Lx[o] = c.Ly[o] = C(0);
        return;
    }
    cpx pRx, pRy, pEz, pBz, pLx, pLy;
    if (n == c.b - 1) {
        pRx = c.bnd[0 * c.nk + i]; pRy = c.bnd[1 * c.nk + i]; pEz = c.bnd[2 * c.nk + i];
        pBz = c.bnd[3 * c.nk + i]; pLx = c.bnd[4 * c.nk + i]; pLy = c.bnd[5 * c.nk + i];
    } else {
        const long p = o + c.nk;
        pRx = c.Rx[p]; pRy = c.Ry[p]; pEz = c.Ez[p]; pBz = c.Bz[p]; pLx = c.Lx[p]; pLy = c.Ly[p];
    }
    const cpx upx = c.jkx[ou], upy = c.jky[ou], upz = c.jkz[ou];
    const cpx Lsx = pLx - 0.25 * h * (a * pEz - b * pBz) - 0.5 * h * upx;
    const cpx Lsy = pLy - 0.25 * h * (a * pBz + b * pEz) - 0.5 * h * upy;
    const cpx Cp = 0.5 * (a * (pRy + pLy) - b * (pRx + pLx));
    const cpx Qp = -0.5 * (a * (pRx - pLx)) + 0.5 * (b * (pLy - pRy));
    const cpx SB = a * Rsy + a * Lsy - b * Rsx - b * Lsx;
    const cpx SE = -(a * Rsx) + a * Lsx + b * Lsy - b * Rsy;
    const double den = 1 + 0.25 * h * (D + 0.25 * h) * k2;
    const cpx Bzn = (pBz - 0.5 * h * Cp - 0.25 * h * SB) / den;
    const cpx Ezn = (pEz - 0.5 * h * Qp - 0.25 * h * SE - h * upz) / den;
    c.Ez[o] = Ezn; c.Bz[o] = Bzn;
    c.Rx[o] = Rsx + D * (a * Ezn + b * Bzn); c.Ry[o] = Rsy + D * (b * Ezn - a * Bzn);
    c.Lx[o] = Lsx - 0.25 * h * (a * Ezn - b * Bzn); c.Ly[o] = Lsy - 0.25 * h * (a * Bzn + b * Ezn);
}
}  // namespace

// The march of this block: every mode is an independent recurrence over the slices, from b-1 down to a.
// GPU: one thread per mode (coalesced across modes); CPU: slice by slice, OpenMP over the modes.
void Simulation::march(double D) {
    const size_t pl = size_t(nk);
    if (comm.rank < comm.size - 1) {
        std::vector<cpx> buf(6 * pl);
        comm.recv(comm.rank + 1, TAG_BND, buf.data(), buf.size());
        bnd.from_host(buf.data(), 0, buf.size());
    }
    MarchCtx c{Rx.data(), Ry.data(), Ez.data(), Bz.data(), Lx.data(), Ly.data(), bnd.data(), j0x.data(), j0y.data(),
               jkx.data(), jky.data(), jkz.data(), kxd.data(), kyd.data(), D, cfg.h, a, b, N, long(nk)};
    const int lo = a, hi = b;
#if GEMPIC_GPU
    par_for(nk, [=] HD(long i) { for (int n = hi - 1; n >= lo; --n) march_point(c, i, n); });
#else
    for (int n = hi - 1; n >= lo; --n) par_for(nk, [=](long i) { march_point(c, i, n); });
#endif
    device_sync();
    if (comm.rank > 0) {           // slice a of level s1 is the boundary of the block below
        std::vector<cpx> buf(6 * pl);
        int k = 0;
        for (Buf<cpx>* p : {&Rx, &Ry, &Ez, &Bz, &Lx, &Ly}) p->to_host(&buf[pl * k++], 0, pl);
        comm.isend(comm.rank - 1, TAG_BND, buf.data(), buf.size());
    }
}

// Co-moving field of the bunch: -(lap_perp + gamma^-2 d_zeta^2) psi = rho, R = (1 + beta) E_perp,
// E_perp = -grad psi; one tridiagonal (Thomas) solve along zeta per mode, pipelined over the ranks.
void Simulation::comoving_field() {
    double sums[2] = {0, 0};
    for (auto& p : host_particles())
        if (p.alive == 1) { sums[0] += p.w * p.pz / std::sqrt(1 + p.px * p.px + p.py * p.py + p.pz * p.pz); sums[1] += p.w; }
    comm.allreduce_sum(sums, 2);
    const double beta = sums[1] > 0 ? sums[0] / sums[1] : 1;
    const double off = (1 - beta * beta) / (cfg.h * cfg.h);
    const size_t pl = size_t(nk);
    // forward elimination: cp in S1, dp in S2; the last row of the block below arrives in bnd[0], bnd[1]
    if (comm.rank > 0) {
        std::vector<cpx> buf(2 * pl);
        comm.recv(comm.rank - 1, TAG_TRIF, buf.data(), buf.size());
        bnd.from_host(buf.data(), 0, buf.size());
    }
    cpx* cp = S1.data(); cpx* dp = S2.data();
    const cpx* jz = jkz.data(); const cpx* bd = bnd.data();
    const double* kx = kxd.data(); const double* ky = kyd.data();
    const long nkk = nk; const int a_ = a, ns_ = ns, first = comm.rank == 0;
    par_for(nk, [=] HD(long i) {
        const double dg = kx[i] * kx[i] + ky[i] * ky[i] + 2 * off;
        cpx cprev = first ? C(0) : bd[i], dprev = first ? C(0) : bd[nkk + i];
        for (int l = 0; l < ns_; ++l) {
            const cpx rhs = (0.5 / beta) * (jz[l * nkk + i] + jz[(l + 1) * nkk + i]);
            if (dg == 0) { cp[l * nkk + i] = C(0); dp[l * nkk + i] = C(0); continue; }
            if (a_ + l == 0) { cp[l * nkk + i] = C(-off / dg); dp[l * nkk + i] = rhs / dg; }
            else {
                const cpx den = C(dg) + off * cprev;
                cp[l * nkk + i] = C(-off) / den; dp[l * nkk + i] = (rhs + off * dprev) / den;
            }
            cprev = cp[l * nkk + i]; dprev = dp[l * nkk + i];
        }
    });
    device_sync();
    if (comm.rank < comm.size - 1) {
        std::vector<cpx> buf(2 * pl);
        S1.to_host(buf.data(), (ns - 1) * pl, pl); S2.to_host(&buf[pl], (ns - 1) * pl, pl);
        comm.isend(comm.rank + 1, TAG_TRIF, buf.data(), buf.size());
    }
    // back substitution, psi overwrites dp; psi of slice b arrives from the block above
    std::vector<cpx> top(pl, C(0));
    if (comm.rank < comm.size - 1) comm.recv(comm.rank + 1, TAG_TRIB, top.data(), pl);
    bnd.from_host(top.data(), 0, pl);
    par_for(nk, [=] HD(long i) {
        cpx next = bd[i];
        for (int l = ns_ - 1; l >= 0; --l) {
            next = dp[l * nkk + i] - cp[l * nkk + i] * next;
            dp[l * nkk + i] = next;
        }
    });
    device_sync();
    if (comm.rank > 0) {
        std::vector<cpx> buf(pl);
        S2.to_host(buf.data(), 0, pl);
        comm.isend(comm.rank - 1, TAG_TRIB, buf.data(), pl);
    }
    cpx* rx = Rx.data(); cpx* ry = Ry.data(); cpx* ez = Ez.data(); cpx* bz = Bz.data();
    par_for(long(ns) * nk, [=] HD(long idx) {
        const long i = idx % nkk;
        rx[idx] = -(1 + beta) * (ix(kx[i]) * dp[idx]);
        ry[idx] = -(1 + beta) * (ix(ky[i]) * dp[idx]);
        ez[idx] = C(0); bz[idx] = C(0);
    });
    device_sync();
}

// Sponge on the radiation part R - R_b of the forward field (R_b = 2 E_perp of the charge of the slice,
// k^2 psi = rho, v = c limit), then E and B of the own slices in real space, then the ghost slices.
void Simulation::sponge_and_real_fields() {
    const long nkk = nk;
    const int nx = cfg.nx, ny = cfg.ny, ns_ = ns;
    cpx* rx = Rx.data(); cpx* ry = Ry.data();
    cpx* s1 = S1.data(); cpx* s2 = S2.data();
    const cpx* jz = jkz.data();
    const double* kx = kxd.data(); const double* ky = kyd.data(); const double* mk = mask.data();
    auto bound = [=] HD(long idx, cpx& bx, cpx& by) {
        const long i = idx % nkk; const long l = idx / nkk;
        const double k2 = kx[i] * kx[i] + ky[i] * ky[i];
        const cpx rho = 0.5 * (jz[l * nkk + i] + jz[(l + 1) * nkk + i]);
        const cpx f = k2 > 0 ? (-2.0 / k2) * rho : C(0);
        bx = ix(kx[i]) * f; by = ix(ky[i]) * f;
    };
    if (cfg.sponge > 0) {
        par_for(long(ns) * nk, [=] HD(long idx) {
            cpx bx, by; bound(idx, bx, by);
            const cpx u = rx[idx] - bx, v = ry[idx] - by;
            s1[idx] = C(u.re - v.im, u.im + v.re);           // u + i v
        });
        fft_->run(s1, ns, true);
        par_for(long(ns) * nk, [=] HD(long idx) { s1[idx] = mk[idx % nkk] * s1[idx]; });
        fft_->run(s1, ns, false);
        par_for(long(ns) * nk, [=] HD(long idx) {
            const long base = idx - idx % nkk, i = idx % nkk;
            const int ixx = int(i / ny), iyy = int(i % ny);
            const long m = base + long((nx - ixx) % nx) * ny + (ny - iyy) % ny;
            const cpx z = s1[idx], zm = conj(s1[m]);
            cpx bx, by; bound(idx, bx, by);
            s2[idx] = 0.5 * (z + zm) + bx;                                 // new R_x, kept in S2
            rx[idx] = 0.5 * C((z - zm).im, -(z - zm).re) + by;            // new R_y, written into R_x for now
        });
        par_for(long(ns) * nk, [=] HD(long idx) { ry[idx] = rx[idx]; rx[idx] = s2[idx]; });
    }
    // real fields: S1 <- inverse(R_x + i R_y), S2 <- inverse(L_x + i L_y)
    const cpx* lx = Lx.data(); const cpx* ly = Ly.data();
    par_for(long(ns) * nk, [=] HD(long idx) {
        s1[idx] = C(rx[idx].re - ry[idx].im, rx[idx].im + ry[idx].re);
        s2[idx] = C(lx[idx].re - ly[idx].im, lx[idx].im + ly[idx].re);
    });
    fft_->run(s1, ns, true);
    fft_->run(s2, ns, true);
    float* Fp = F.data();
    const long fpl = long(ns + 2) * nkk;      // stride between the components of F
    par_for(long(ns) * nk, [=] HD(long idx) {
        const double r1 = s1[idx].re, r2 = s1[idx].im, l1 = s2[idx].re, l2 = s2[idx].im;
        Fp[0 * fpl + idx] = float(0.5 * (r1 + l1));      // E_x
        Fp[1 * fpl + idx] = float(0.5 * (r2 + l2));      // E_y
        Fp[3 * fpl + idx] = float(0.5 * (l2 - r2));      // B_x
        Fp[4 * fpl + idx] = float(0.5 * (r1 - l1));      // B_y
    });
    const cpx* ez = Ez.data(); const cpx* bz = Bz.data();
    par_for(long(ns) * nk, [=] HD(long idx) { s1[idx] = C(ez[idx].re - bz[idx].im, ez[idx].im + bz[idx].re); });
    fft_->run(s1, ns, true);
    par_for(long(ns) * nk, [=] HD(long idx) {
        Fp[2 * fpl + idx] = float(s1[idx].re);           // E_z
        Fp[5 * fpl + idx] = float(s1[idx].im);           // B_z
    });
    device_sync();
    (void)ns_;
    // ghost slices: the two lowest slices of this block go to the block below; receive b, b+1 from above
    const size_t pl = size_t(nk);
    if (comm.rank > 0) {
        std::vector<float> buf(12 * pl);
        for (int comp = 0; comp < 6; ++comp) F.to_host(&buf[comp * 2 * pl], size_t(comp) * (ns + 2) * pl, 2 * pl);
        comm.isend(comm.rank - 1, TAG_GHOST, buf.data(), buf.size());
    }
    if (comm.rank < comm.size - 1) {
        std::vector<float> buf(12 * pl);
        comm.recv(comm.rank + 1, TAG_GHOST, buf.data(), buf.size());
        for (int comp = 0; comp < 6; ++comp) F.from_host(&buf[comp * 2 * pl], (size_t(comp) * (ns + 2) + ns) * pl, 2 * pl);
    }
}

void Simulation::solve_fields(double D) {
    currents();
    march(D);
    sponge_and_real_fields();
    const long n = long(ns + 1) * nk;
    cpx* a0 = j0x.data(); cpx* b0 = j0y.data(); const cpx* a1 = jkx.data(); const cpx* b1 = jky.data();
    par_for(n, [=] HD(long i) { a0[i] = a1[i]; b0[i] = b1[i]; });
    device_sync();
}

// ---------------------------------------------------------------- time step
void Simulation::init() {
    load_bunch();
    s = 0; istep = 0;
    if (cfg.self_fields) {
        currents();
        comoving_field();
        march(0.0);                 // E_z, B_z, L of the co-moving R; the sponge acts as in every step
        sponge_and_real_fields();
        const long n = long(ns + 1) * nk;
        cpx* a0 = j0x.data(); cpx* b0 = j0y.data(); const cpx* a1 = jkx.data(); const cpx* b1 = jky.data();
        par_for(n, [=] HD(long i) { a0[i] = a1[i]; b0[i] = b1[i]; });
        device_sync();
    }
    window_shift_post();
    comm.wait_sends();
}

void Simulation::step() {
    window_shift_start();
    push(s, s + cfg.dz, 0, np);
    exchange_particles(s + cfg.dz);
    s += cfg.dz; ++istep;
    if (cfg.self_fields) solve_fields(0.5 * cfg.dz);
    window_shift_post();
    comm.wait_sends();
}

// The transverse window follows the bunch centroid in whole cells. The centroid of step k is reduced over
// the ranks without blocking and used at step k + window_lag, so the ranks need not wait for each other.
void Simulation::window_shift_post() {
    if (!cfg.move_x && !cfg.move_y) return;
    Buf<double> acc(3);
    double* A = acc.data();
    const Particle* P = parts.data();
    const double x0 = X0, y0 = Y0;
    par_for(np, [=] HD(long i) {
        if (P[i].alive != 1) return;
        atomic_add(&A[0], P[i].w * (P[i].x + x0)); atomic_add(&A[1], P[i].w * (P[i].y + y0)); atomic_add(&A[2], P[i].w);
    });
    device_sync();
    cent_.emplace_back(0, acc.host());
    cent_.back().first = comm.iallreduce_sum(cent_.back().second.data(), 3);
}
void Simulation::window_shift_start() {
    if (!cfg.move_x && !cfg.move_y) return;
    if (long(cent_.size()) < cfg.window_lag) return;       // centroid of step istep + 1 - lag not yet due
    comm.wait(cent_.front().first);
    const std::vector<double> c = cent_.front().second;
    cent_.pop_front();
    if (c[2] <= 0) return;
    const int nxs = cfg.move_x ? int(std::lround((c[0] / c[2] - (X0 + 0.5 * cfg.Lx)) / dx)) : 0;
    const int nys = cfg.move_y ? int(std::lround((c[1] / c[2] - (Y0 + 0.5 * cfg.Ly)) / dy)) : 0;
    if (std::abs(nxs) >= 2 || std::abs(nys) >= 2) shift_window(nxs, nys);
}

// A shift is an exact phase factor in Fourier space and an index roll in real space.
void Simulation::shift_window(int nxs, int nys) {
    X0 += nxs * dx; Y0 += nys * dy;
    const double sx = nxs * dx, sy = nys * dy;
    Particle* P = parts.data();
    par_for(np, [=] HD(long i) { P[i].x -= sx; P[i].y -= sy; });
    if (cfg.self_fields) {
        const long nkk = nk;
        const double* kx = kxs.data(); const double* ky = kys.data();
        auto rot = [&](Buf<cpx>& A) {
            cpx* p = A.data();
            par_for(long(A.size()), [=] HD(long idx) {
                const long i = idx % nkk;
                const double ph = kx[i] * sx + ky[i] * sy;
                p[idx] = p[idx] * C(::cos(ph), ::sin(ph));
            });
        };
        rot(Rx); rot(Ry); rot(Ez); rot(Bz); rot(j0x); rot(j0y);
        Buf<float> tmp(F.size());
        float* t = tmp.data(); const float* f = F.data();
        const int nx = cfg.nx, ny = cfg.ny;
        par_for(long(F.size()), [=] HD(long idx) {
            const long base = idx - idx % nkk, i = idx % nkk;
            const int ixx = int(i / ny), iyy = int(i % ny);
            const int sxx = ((ixx + nxs) % nx + nx) % nx, syy = ((iyy + nys) % ny + ny) % ny;
            t[idx] = f[base + long(sxx) * ny + syy];
        });
        device_sync();
        F = std::move(tmp);
    }
    device_sync();
}

// ---------------------------------------------------------------- output
void Simulation::diagnostics(FILE* f, bool header) {
    if (header && comm.rank == 0)
        for (FILE* out : {f, stdout}) if (out) std::fprintf(out, "# step  z[mm]  N_alive  <x>[um] <y>[um] <zeta>[um]  rms_x rms_y rms_zeta[um]  <px> <py> <pz> <gamma> "
                        "<p.khat>  rms_pz  eta_rel[FWHM] a_env  W_field[J] W_kin[J]\n");
    double S[17] = {0};
    for (auto& p : host_particles()) {
        if (p.alive != 1) continue;
        const double g = std::sqrt(1 + p.px * p.px + p.py * p.py + p.pz * p.pz);
        const double X = p.x + X0, Y = p.y + Y0;
        const double et = laser.eta(X, s, s - (p.zeta - cfg.zeta_center));
        const double v[] = {1, X, Y, p.zeta, X * X, Y * Y, p.zeta * p.zeta, p.px, p.py, p.pz, g,
                            p.px * laser.kx + p.pz * laser.kz, p.pz * p.pz, et};
        for (int k = 0; k < 14; ++k) S[k] += p.w * v[k];
        S[14] += 1;
    }
    S[15] = cfg.self_fields ? local_field_energy() : 0;
    S[16] = double(lost_total);
    comm.allreduce_sum(S, 17);
    field_energy = S[15];
    const double w = S[0] > 0 ? S[0] : 1;
    auto m = [&](int k) { return S[k] / w; };
    auto rms = [&](int k, int k2) { return std::sqrt(std::max(0.0, m(k2) - m(k) * m(k))); };
    const double eta_rel = laser.on ? (m(13) - laser.eta_c) / laser.tau : 0;
    if (laser.on && cfg.stop_fwhm > 0 && eta_rel > cfg.stop_fwhm) stop_ = true;
    if (comm.rank != 0) return;
    const double um = cfg.Lu * 1e6;
    const double aenv = laser.on ? laser.a0 * laser.envelope(m(13)) : 0;
    const double Efield = me_SI * c_SI * c_SI / (cfg.Lu * e_SI);
    const double Wf = field_energy * eps0 * Efield * Efield * std::pow(cfg.Lu, 3);
    const double Wk = S[14] * W_macro_ne * (m(10) - 1) * me_SI * c_SI * c_SI;
    char line[512];
    std::snprintf(line, sizeof line, "%6ld %10.5f %8ld %10.4f %10.4f %10.4f %9.4f %9.4f %9.4f %12.6e %12.6e %14.8e %14.8e %14.8e %11.4e %8.4f %8.4f %11.4e %11.4e\n",
                 istep, s * cfg.Lu * 1e3, long(S[14]), m(1) * um, m(2) * um, m(3) * um, rms(1, 4) * um, rms(2, 5) * um,
                 rms(3, 6) * um, m(7), m(8), m(9), m(10), m(11), rms(9, 12), eta_rel, aenv, Wf, Wk);
    for (FILE* out : {f, stdout}) if (out) { std::fputs(line, out); std::fflush(out); }
}

// field energy of the own slices in code units (trapezoidal weights at the window ends)
double Simulation::local_field_energy() {
    Buf<double> acc(1);
    double* A = acc.data();
    const float* Fp = F.data();
    const long nkk = nk, fpl = long(ns + 2) * nk;
    const int a_ = a, ns_ = ns, Ng = N;
    par_for(nk, [=] HD(long i) {
        double e = 0;
        for (int l = 0; l < ns_; ++l) {
            const double w = (a_ + l == 0 || a_ + l == Ng - 1) ? 0.5 : 1.0;
            for (int comp = 0; comp < 6; ++comp) { const double v = Fp[comp * fpl + l * nkk + i]; e += 0.5 * w * v * v; }
        }
        atomic_add(A, e);
    });
    device_sync();
    return acc.host()[0] * dx * dy * cfg.h;
}
double Simulation::total_field_energy() {
    double e = cfg.self_fields ? local_field_energy() : 0;
    comm.allreduce_sum(&e, 1);
    return e;
}

void Simulation::dump_particles(const std::string& file) {
    std::vector<double> mine;
    for (auto& p : host_particles())
        if (p.alive == 1) for (double v : {p.x + X0, p.y + Y0, p.zeta, p.px, p.py, p.pz}) mine.push_back(v);
    std::vector<char> all = comm.gather_bytes(mine.data(), mine.size() * sizeof(double));
    if (comm.rank != 0) return;
    const double* d = reinterpret_cast<const double*>(all.data());
    const size_t n = all.size() / sizeof(double) / 6;
    std::ofstream o(file);
    o.precision(10);
    o << "# x[um] y[um] zeta[um] px py pz (m c), z = " << s * cfg.Lu * 1e3 << " mm\n";
    const double um = cfg.Lu * 1e6;
    for (size_t k = 0; k < n; ++k)
        o << d[6 * k] * um << ' ' << d[6 * k + 1] * um << ' ' << d[6 * k + 2] * um << ' ' << d[6 * k + 3] << ' '
          << d[6 * k + 4] << ' ' << d[6 * k + 5] << '\n';
}
