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
enum Tag { TAG_HAND = 11, TAG_CUR = 12, TAG_BND = 13, TAG_GHOST = 14, TAG_TRIF = 15, TAG_TRIB = 16, TAG_JZU = 17, TAG_JZD = 18 };
}  // namespace

// ---------------------------------------------------------------- configuration
Config Config::base_units(double lambda0) {
    Config c;
    c.lambda0 = lambda0; c.Lu = lambda0 / (2 * M_PI); c.Tu = c.Lu / c_SI;
    const double w0 = c_SI / c.Lu;
    c.nc = eps0 * me_SI * w0 * w0 / (e_SI * e_SI);
    return c;
}

// ---------------------------------------------------------------- grids
void Config::zeta_uniform(int n, double Lw) {
    if (n < 4) throw std::runtime_error("grid: nzeta must be >= 4");
    const double h = Lw / (n - 1);
    nzeta = n; zeta.resize(n);
    for (int i = 0; i < n; ++i) zeta[i] = i * h;
    Lwin = zeta.back(); zeta_change = 0;
}
void Config::zeta_table(const std::vector<double>& pts, const std::vector<double>& steps, double Lw) {
    StepMapping m;
    const std::string err = m.set(pts, steps);
    if (!err.empty()) throw std::runtime_error("grid.zeta_points/dzeta_points: " + err);
    const int cells = int(std::ceil(m.index(Lw) - 1e-9));
    if (cells < 3) throw std::runtime_error("grid: the zeta window needs at least 3 cells");
    nzeta = cells + 1; zeta.resize(nzeta);
    for (int i = 0; i < nzeta; ++i) zeta[i] = m.coord(i);
    Lwin = zeta.back(); zeta_change = m.max_change();
}
void Config::y_uniform(int n, double Ly_) {
    if (n < 4) throw std::runtime_error("grid: ny must be >= 4");
    const double h = Ly_ / n;
    ny = n; ynode.resize(n + 1); y_fourier = true;
    for (int j = 0; j <= n; ++j) ynode[j] = (j - n / 2) * h;
    Ly = Ly_; y_change = 0;
}
void Config::y_table(const std::vector<double>& pts, const std::vector<double>& steps, double ymax) {
    StepMapping m;
    const std::string err = m.set(pts, steps);
    if (!err.empty()) throw std::runtime_error("grid.y_points/dy_points: " + err);
    const int half = int(std::ceil(m.index(ymax) - 1e-9));
    if (half < 2) throw std::runtime_error("grid: the y grid needs at least 2 cells on each side");
    ny = 2 * half; ynode.resize(ny + 1); y_fourier = false; move_y = false;
    for (int k = 0; k <= half; ++k) { ynode[half + k] = m.coord(k); ynode[half - k] = -m.coord(k); }
    Ly = ynode[ny] - ynode[0]; y_change = m.max_change();
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
    auto LL = [&](const std::string& k) {
        std::vector<double> v = d.list("grid", k, Kind::Length);
        for (double& x : v) x /= c.Lu;
        return v;
    };
    c.nx = int(d.integer("grid", "nx", 0));
    if (c.nx < 4) throw std::runtime_error("grid: nx must be >= 4");
    c.Lx = L("grid", "Lx");
    const std::string yg = d.str("grid", "y_grid", "fourier");
    if (yg == "fourier") c.y_uniform(int(d.integer("grid", "ny", 0)), Ld("grid", "Ly", c.Lx));
    else if (yg == "table") {
        if (d.has("grid", "ny") || d.has("grid", "Ly"))
            throw std::runtime_error("grid: with y_grid = table, ny and Ly follow from y_points, dy_points and y_max");
        if (d.flag("grid", "move_y", false))
            throw std::runtime_error("grid: move_y needs y_grid = fourier (the free-hand y grid is fixed between walls)");
        c.y_table(LL("y_points"), LL("dy_points"), L("grid", "y_max"));
    } else throw std::runtime_error("grid.y_grid must be fourier or table");
    const std::string zg = d.str("grid", "zeta_grid", "uniform");
    if (zg == "uniform") c.zeta_uniform(int(d.integer("grid", "nzeta", 0)), L("grid", "window_length"));
    else if (zg == "table") {
        if (d.has("grid", "nzeta"))
            throw std::runtime_error("grid: with zeta_grid = table, nzeta follows from the table and window_length");
        c.zeta_table(LL("zeta_points"), LL("dzeta_points"), L("grid", "window_length"));
    } else throw std::runtime_error("grid.zeta_grid must be uniform or table");
    c.dz = L("grid", "dz"); c.z_end = L("grid", "z_end");
    c.dz_push = Ld("grid", "dz_push", c.dz);
    c.sponge = Ld("grid", "sponge", 0.1 * std::min(c.Lx, c.Ly));
    c.sponge_y = Ld("grid", "sponge_y", c.sponge);
    c.sponge_rate = d.num("grid", "sponge_rate", Kind::None, 0.1);
    c.move_x = d.flag("grid", "move_x", true);
    if (c.y_fourier) c.move_y = d.flag("grid", "move_y", true);
    c.window_lag = int(d.integer("grid", "window_lag", 4));
    if (c.window_lag < 1) throw std::runtime_error("grid.window_lag must be >= 1");
    c.self_fields = d.flag("numerics", "self_fields", true);
    c.zeta_filter = d.flag("numerics", "zeta_filter", true);
    c.k_filter = d.flag("numerics", "k_filter", true);
    c.push_phase = d.num("numerics", "push_phase", Kind::Angle, 0.05);

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
    double dzmin = 1e300, dzmax = 0, dymin = 1e300, dymax = 0;
    for (int m = 0; m + 1 < nzeta; ++m) { dzmin = std::min(dzmin, zeta[m + 1] - zeta[m]); dzmax = std::max(dzmax, zeta[m + 1] - zeta[m]); }
    for (int j = 0; j < ny; ++j) { dymin = std::min(dymin, ynode[j + 1] - ynode[j]); dymax = std::max(dymax, ynode[j + 1] - ynode[j]); }
    std::fprintf(f, "GEM-PIC-3D\n  lambda0 = %.4g um, code length = %.4g um, n_c = %.4g cm^-3\n", lambda0 * 1e6, um, nc * 1e-6);
    std::fprintf(f, "  x: %d modes, Lx = %.4g um (periodic), dx = %.4g um\n", nx, Lx * um, Lx / nx * um);
    if (y_fourier) std::fprintf(f, "  y: %d modes, Ly = %.4g um (periodic), dy = %.4g um\n", ny, Ly * um, Ly / ny * um);
    else std::fprintf(f, "  y: free-hand, %d cells between walls at +-%.4g um, dy = %.4g ... %.4g um, largest change %.1f %% per cell\n",
                      ny, 0.5 * Ly * um, dymin * um, dymax * um, 100 * y_change);
    std::fprintf(f, "  zeta: %d slices, window = %.4g um, dzeta = %.4g ... %.4g um, largest change %.1f %% per cell\n",
                 nzeta, Lwin * um, dzmin * um, dzmax * um, 100 * zeta_change);
    if (y_change > 0.25 || zeta_change > 0.25)
        std::fprintf(f, "  WARNING: a grid step changes by more than 25 %% from one cell to the next (keep it below ~20 %%)\n");
    std::fprintf(f, "  dz = %.4g um, dz_push = %.4g um, z_end = %.4g mm\n", dz * um, dz_push * um, z_end * Lu * 1e3);
    std::fprintf(f, "  sponge x %.4g um, y %.4g um, rate %.3g; transverse window follows bunch: x %s, y %s\n", sponge * um,
                 sponge_y * um, sponge_rate, move_x ? "yes" : "no", move_y ? "yes" : "no");
    if (laser_on)
        std::fprintf(f, "  laser: a0 = %g, lambda = %.4g um, angle = %.4g rad, %s, tau_FWHM = %.4g ps, start/stop %.3g/%.3g FWHM\n",
                     a0, lambda0 / kL * 1e6, theta, pol.c_str(), tau * Tu * 1e12, start_fwhm, stop_fwhm);
    else std::fprintf(f, "  laser: off\n");
    std::fprintf(f, "  bunch: Q = %.4g pC, gamma = %g, sigma = %.3g x %.3g x %.3g um, %ld macroparticles (%s noise)\n",
                 charge * 1e12, gamma, sigx * um, sigy * um, sigz * um, nmacro, noise.c_str());
    std::fprintf(f, "  push: RK4 in z, sub-step <= %.3g rad of laser phase\n", push_phase);
    std::fprintf(f, "  parallel: %s, %d MPI rank(s), transverse window lag %d steps\n", GEMPIC_GPU ? "CUDA" : "OpenMP", ranks, window_lag);
    std::fprintf(f, "  self fields %s, zeta filter %s, transverse filter %s\n", self_fields ? "on" : "off", zeta_filter ? "on" : "off",
                 k_filter ? "on" : "off");
}

// ---------------------------------------------------------------- setup
Simulation::Simulation(const Config& c, Comm& cm) : cfg(c), comm(cm) {
    N = cfg.nzeta; nk = cfg.nx * cfg.ny;
    dx = cfg.Lx / cfg.nx;
    if (int(cfg.zeta.size()) != N || int(cfg.ynode.size()) != cfg.ny + 1) throw std::runtime_error("grids not set");
    // blocks of consecutive slices, rank P-1 at the front of the window
    const int P = comm.size, base = N / P, rem = N % P;
    a = 0;
    for (int r = 0; r < comm.rank; ++r) a += base + (r < rem);
    ns = base + (comm.rank < rem);
    b = a + ns;
    if (P > 1 && base < 4) throw std::runtime_error("MPI: at least 4 slices per rank are needed");

    // x: Fourier modes
    const int nx = cfg.nx;
    std::vector<double> hkx(nx), hsx(nx), hf(nx), hm(nx);
    auto prof = [&](double dd, double w) {          // dd: distance from the box edge
        if (w <= 0 || dd >= w) return 1.0;
        const double r = 1 - dd / w;
        return 1 - cfg.sponge_rate * r * r;
    };
    for (int ixx = 0; ixx < nx; ++ixx) {
        const int mx = ixx <= nx / 2 ? ixx : ixx - nx;
        hsx[ixx] = 2 * M_PI * mx / cfg.Lx;
        hkx[ixx] = (2 * ixx == nx) ? 0 : hsx[ixx];          // odd derivatives vanish at Nyquist
        hf[ixx] = cfg.k_filter ? std::pow(std::cos(0.5 * hsx[ixx] * dx), 2) : 1.0;
        if (2 * ixx == nx) hf[ixx] = 0;
        hm[ixx] = prof(std::min(ixx * dx, cfg.Lx - ixx * dx), cfg.sponge);
    }
    kxd.upload(hkx); kxs.upload(hsx); filt.upload(hf); maskx.upload(hm);
    yf = cfg.y_fourier;
    if (yf) setup_y_fourier(); else setup_y();
    setup_zeta();

    const size_t pl = size_t(nk);
    for (Buf<cpx>* p : {&Rx, &Ry, &Ez, &Bz, &Lx, &Ly}) p->resize(ns * pl);
    bnd.resize(6 * pl);
    jzg.resize(2 * pl);
    for (Buf<cpx>* p : {&j0x, &j0y, &jkx, &jky, &jkz, &S1, &S2}) p->resize((ns + 1) * pl);
    if (!yf) {
        for (Buf<cpx>* p : {&sRx, &sRy, &sLx, &sLy, &srhs}) p->resize(pl);
        scp.resize(pl);
    }
    jw.resize(3 * (ns + 3) * pl);
    F.resize(6 * (ns + 2) * pl);
    fft_ = std::make_unique<FftMany>(cfg.nx, cfg.ny, yf);

    laser.on = cfg.laser_on; laser.a0 = cfg.a0; laser.kL = cfg.kL; laser.tau = cfg.tau; laser.cep = cfg.cep;
    laser.kx = std::sin(cfg.theta); laser.kz = std::cos(cfg.theta);
    laser.pol = cfg.pol == "s" ? 0 : cfg.pol == "p" ? 1 : 2;
    laser.eta_c = cfg.start_fwhm * cfg.tau;   // the bunch centre starts at eta = 0
}

// Fourier y: k_y per iy as for x, all components on the nodes j*dy of the box (physical y = j*dy + Y0).
void Simulation::setup_y_fourier() {
    const int ny = cfg.ny;
    dy = cfg.Ly / ny;
    std::vector<double> hky(ny), hsy(ny), hf(ny), hm(ny), hl(ny), hi(ny, 1 / dy);
    for (int j = 0; j < ny; ++j) {
        const int my = j <= ny / 2 ? j : j - ny;
        hsy[j] = 2 * M_PI * my / cfg.Ly;
        hky[j] = (2 * j == ny) ? 0 : hsy[j];
        hf[j] = cfg.k_filter ? std::pow(std::cos(0.5 * hsy[j] * dy), 2) : 1.0;
        if (2 * j == ny) hf[j] = 0;
        const double dd = std::min(j * dy, cfg.Ly - j * dy), w = cfg.sponge_y;
        double m = 1;
        if (w > 0 && dd < w) { const double r = 1 - dd / w; m = 1 - cfg.sponge_rate * r * r; }
        hm[j] = m; hl[j] = hky[j] * hky[j];
    }
    kyd.upload(hky); kys.upload(hsy); filty.upload(hf); masky_n.upload(hm); masky_h.upload(hm); evl.upload(hl);
    iD.upload(hi); iA.upload(hi);
    hDel.assign(ny, dy); hA.assign(ny, dy);
    yax.X = nullptr; yax.n = ny + 1; yax.uniform = 1; yax.x0 = 0; yax.ih = 1 / dy;
}

// Free-hand y: nodes j = 0..ny (0 and ny are the conducting walls), half-nodes j+1/2 = midpoints, j = 0..ny-1.
// E_x, B_y (R_x, L_x), E_z, j_x, j_z live on the nodes; E_y, B_x (R_y, L_y), B_z, j_y on the half-nodes.
void Simulation::setup_y() {
    const int ny = cfg.ny, nx = cfg.nx;
    const std::vector<double>& Y = cfg.ynode;
    hDel.assign(ny, 0); hA.assign(ny, 0);
    for (int j = 0; j < ny; ++j) hDel[j] = Y[j + 1] - Y[j];
    hA[0] = 0.5 * hDel[0];
    for (int j = 1; j < ny; ++j) hA[j] = 0.5 * (hDel[j - 1] + hDel[j]);
    std::vector<double> hiD(ny), hiA(ny), hmn(ny), hmh(ny);
    auto prof = [&](double y) {
        const double dd = 0.5 * cfg.Ly - std::fabs(y), w = cfg.sponge_y;
        if (w <= 0 || dd >= w) return 1.0;
        const double r = 1 - dd / w;
        return 1 - cfg.sponge_rate * r * r;
    };
    for (int j = 0; j < ny; ++j) {
        hiD[j] = 1 / hDel[j]; hiA[j] = 1 / hA[j];
        hmn[j] = prof(Y[j]); hmh[j] = prof(0.5 * (Y[j] + Y[j + 1]));
    }
    yn.upload(Y); iD.upload(hiD); iA.upload(hiA); masky_n.upload(hmn); masky_h.upload(hmh);
    yax.X = yn.data(); yax.n = ny + 1;
    bool uni = true;
    for (int j = 0; j < ny; ++j) uni = uni && hDel[j] == hDel[0];
    yax.uniform = uni; yax.x0 = Y[0]; yax.ih = 1 / hDel[0];

    // LU factors of (k_x^2 - Lap_y) on the nodes 1..ny-1 (Dirichlet at the walls), per k_x: the bound field
    const std::vector<double> kx = kxd.host();
    std::vector<double> hcp(size_t(nk), 0.0), hid(size_t(nk), 0.0);
    for (int i = 0; i < nx; ++i) {
        double cprev = 0;
        for (int j = 1; j < ny; ++j) {
            const double lo = j > 1 ? -hiA[j] * hiD[j - 1] : 0, up = j < ny - 1 ? -hiA[j] * hiD[j] : 0;
            const double den = kx[i] * kx[i] + hiA[j] * (hiD[j - 1] + hiD[j]) - lo * cprev;
            hid[size_t(j) * nx + i] = 1 / den;
            hcp[size_t(j) * nx + i] = cprev = up / den;
        }
    }
    pcp.upload(hcp); pid.upload(hid);

    // eigenvectors of -Lap_y on the nodes: T v = mu A v with T symmetric; S = A^-1/2 T A^-1/2 is a symmetric
    // tridiagonal matrix with eigenvectors u, v = A^-1/2 u. Slot s = m + 1 holds the eigenvalue m (slot 0 unused).
    if (cfg.self_fields) {
        const int M = ny - 1;
        std::vector<double> d(M), e(M, 0.0), z;
        for (int j = 1; j < ny; ++j) d[j - 1] = hiA[j] * (hiD[j - 1] + hiD[j]);
        for (int j = 1; j < ny - 1; ++j) e[j - 1] = -hiD[j] / std::sqrt(hA[j] * hA[j + 1]);
        if (!tridiagonal_eigen(d, e, z)) throw std::runtime_error("eigenvectors of the y Laplacian: no convergence");
        std::vector<double> hf(size_t(ny) * ny, 0.0), hb(size_t(ny) * ny, 0.0), hl(ny, 0.0);
        for (int m = 0; m < M; ++m) {
            hl[m + 1] = d[m];
            for (int j = 1; j < ny; ++j) {
                const double u = z[size_t(j - 1) * M + m], sq = std::sqrt(hA[j]);
                hf[size_t(m + 1) * ny + j] = u * sq;          // coefficient = v^T A rho
                hb[size_t(j) * ny + m + 1] = u / sq;          // value = sum v coefficient
            }
        }
        evf.upload(hf); evb.upload(hb); evl.upload(hl);
    }
}

// zeta: slices zeta_n, half-cell m between slices m and m+1 with length h_m. hcg[m+2] = h_m for
// m = -2..N, continued with the end cells. The current of slice n is the linear interpolation of the two
// adjacent half-cells (as in GEM-PIC-RZ): (h_n j_{n-1/2} + h_{n-1} j_{n+1/2}) / (h_{n-1} + h_n).
void Simulation::setup_zeta() {
    const std::vector<double>& Z = cfg.zeta;
    std::vector<double> hh(N - 1);
    for (int m = 0; m < N - 1; ++m) hh[m] = Z[m + 1] - Z[m];
    bool uni = true;
    for (int m = 0; m < N - 1; ++m) uni = uni && hh[m] == hh[0];
    if (uni) for (double& v : hh) v = cfg.Lwin / (N - 1);   // exactly the uniform step
    std::vector<double> g(N + 3), lo(N), hi(N), ws(N);
    for (int m = -2; m <= N; ++m) g[m + 2] = hh[std::min(std::max(m, 0), N - 2)];
    for (int n = 0; n < N; ++n) {
        const double hm = g[n - 1 + 2], hp = g[n + 2];
        lo[n] = hp / (hm + hp); hi[n] = hm / (hm + hp);
        ws[n] = 0.5 * ((n > 0 ? hh[n - 1] : 0) + (n < N - 1 ? hh[n] : 0));
    }
    zs.upload(Z); hcg.upload(g); wlo.upload(lo); whi.upload(hi); wsl.upload(ws);
    zax.X = zs.data(); zax.n = N; zax.uniform = uni; zax.x0 = 0; zax.ih = 1 / hh[0];
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
    const double ylo = yf ? -1e300 : cfg.ynode[1], yhi = yf ? 1e300 : cfg.ynode[cfg.ny - 1];
    const double yoff = yf ? 0.5 * cfg.Ly : 0;        // Fourier y: box coordinate, physical y = p.y + Y0
    Axis zh = zax; zh.X = cfg.zeta.data();             // host copy of the axis
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
            p.x = x - vx * zz / vz + 0.5 * cfg.Lx; p.y = y - vy * zz / vz + yoff;
            p.zeta = cfg.zeta_center + zz / vz;
            p.px = px; p.py = py; p.pz = pz; p.w = W; p.alive = 1;
            int hc; double f;
            zh.locate(p.zeta, hc, f);
            if (p.zeta < 0 || hc > N - 2 || p.y < ylo || p.y > yhi) { if (comm.rank == 0) ++lost_total; continue; }
            if (hc >= hlo && hc <= hhi) mine.push_back(p);
        }
    }
    set_particles(mine);
    X0 = -0.5 * cfg.Lx; Y0 = -yoff;
}

// ---------------------------------------------------------------- particles (device code)
namespace {
struct PushCtx {
    const float* F; int a, ns, N, nx, ny, nk, self, yf; double dx, dy, Lx, Ly;
    Axis zax, yax;
    Laser laser; double X0, zc, dz_push, push_phase, zhand, zmax, ylo, yhi, s1;
};
struct State { double x, y, zeta, px, py, pz; };

// Quadratic (TSC) weights around the nearest point of the index coordinate xi (points at integers).
HD inline void tsc_index(double xi, int idx[3], double w[3]) {
    const int i0 = int(::floor(xi + 0.5));
    const double f = xi - i0;
    w[0] = 0.5 * (0.5 - f) * (0.5 - f); w[1] = 0.75 - f * f; w[2] = 0.5 * (0.5 + f) * (0.5 + f);
    for (int k = 0; k < 3; ++k) idx[k] = i0 - 1 + k;
}
// TSC on the periodic uniform x grid with nodes at i*d.
HD inline void tsc(double x, double d, int n, int idx[3], double w[3]) {
    tsc_index(x / d, idx, w);
    for (int k = 0; k < 3; ++k) idx[k] = (idx[k] % n + n) % n;
}
// TSC on the free-hand y grid, in the index coordinate xi(y) (linear within a cell): nodes at the integers,
// half-nodes at the half-integers (stored at index j for j + 1/2). Points outside 0..ny-1 are walls or beyond.
HD inline void tsc_y(const Axis& ax, double y, int jn[3], double wn[3], int jh[3], double wh[3]) {
    int i; double f;
    ax.locate(y, i, f);
    const double xi = i + f;
    tsc_index(xi, jn, wn);
    tsc_index(xi - 0.5, jh, wh);
}
HD inline bool is_half(int comp) { return comp == 1 || comp == 3 || comp == 5; }   // E_y, B_x, B_z

// Grid fields of level s0 at (x, y, zeta): TSC transverse, linear in zeta between the slices of this rank
// (own slices and the two ghost slices of the rank above).
HD inline void gather(const PushCtx& c, double x, double y, double zeta, double E[3], double B[3]) {
    E[0] = E[1] = E[2] = B[0] = B[1] = B[2] = 0;
    if (!c.self) return;
    int n; double fz;
    c.zax.locate(zeta, n, fz);
    const int fl = n - c.a;
    if (fl < 0 || fl > c.ns) return;
    int ix[3], jn[3], jh[3]; double wx[3], wn[3], wh[3];
    tsc(x, c.dx, c.nx, ix, wx);
    if (c.yf) tsc(y, c.dy, c.ny, jn, wn);           // Fourier y: all components on the nodes, periodic
    else tsc_y(c.yax, y, jn, wn, jh, wh);
    for (int comp = 0; comp < 6; ++comp) {
        const float* p0 = c.F + (size_t(comp) * (c.ns + 2) + fl) * c.nk;
        const float* p1 = p0 + c.nk;
        const bool hf = !c.yf && is_half(comp);
        const int* jj = hf ? jh : jn;
        const double* wy = hf ? wh : wn;
        double v = 0;
        for (int j = 0; j < 3; ++j) {
            if (jj[j] < 0 || jj[j] >= c.ny) continue;
            for (int i = 0; i < 3; ++i) {
                const int k = jj[j] * c.nx + ix[i];
                v += wx[i] * wy[j] * ((1 - fz) * p0[k] + fz * p1[k]);
            }
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
    gather(c, ::fmod(::fmod(Y.x, c.Lx) + c.Lx, c.Lx), c.yf ? ::fmod(::fmod(Y.y, c.Ly) + c.Ly, c.Ly) : Y.y, Y.zeta, E, B);
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
    // lost: out of the window in zeta, or within one cell of a wall in y
    p.alive = (p.zeta < 0 || p.zeta >= c.zmax || p.y < c.ylo || p.y > c.yhi) ? 0 : 1;
    // a particle that ends the step in the half-cells of the rank below belongs to that rank
    if (p.alive == 1 && p.zeta < c.zhand) { p.alive = 2; p.rem = 0; }
}
}  // namespace

// Push the particles [i0, i1) to the plane s1: own particles from s0, adopted ones (alive = 3) for their rest.
void Simulation::push(double s0, double s1, long i0, long i1) {
    PushCtx c;
    c.F = F.data(); c.a = a; c.ns = ns; c.N = N; c.nx = cfg.nx; c.ny = cfg.ny; c.nk = nk; c.self = cfg.self_fields;
    c.yf = yf; c.dx = dx; c.dy = dy; c.Lx = cfg.Lx; c.Ly = cfg.Ly; c.zax = zax; c.yax = yax; c.laser = laser; c.X0 = X0;
    c.zc = cfg.zeta_center; c.dz_push = cfg.dz_push; c.push_phase = cfg.push_phase;
    c.zhand = comm.rank == 0 ? -1e300 : cfg.zeta[a + 1];
    c.zmax = cfg.zeta[N - 1]; c.s1 = s1;
    c.ylo = yf ? -1e300 : cfg.ynode[1]; c.yhi = yf ? 1e300 : cfg.ynode[cfg.ny - 1];
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
// half-cell in zeta, TSC transverse; j_x and j_z on the y-nodes, j_y on the half-nodes), exchange of the
// half-cells at the block edges, (1,2,1)/4 filter along zeta on the half-cell charges, spectra in x, the
// transverse filters (cos^2 in k_x, (1,2,1)/4 on the cell charges along y), and no TEM current between the
// walls (the y-mean of j_y at k_x = 0).
void Simulation::currents() {
    jw.zero();
    const int nh = ns + 3, a_ = a, nx = cfg.nx, ny = cfg.ny, Ng = N;
    const long nkk = nk;
    const double dx_ = dx, dy_ = dy;
    const bool yfo = yf;
    const Axis za = zax, ya = yax;
    const double* hc = hcg.data(); const double* ia = iA.data(); const double* id = iD.data();
    double* J = jw.data();
    const Particle* P = parts.data();
    par_for(np, [=] HD(long ip) {
        const Particle& p = P[ip];
        if (p.alive != 1) return;
        int m; double fz;
        za.locate(p.zeta, m, fz);
        const int hw = m - (a_ - 2);
        if (hw < 0 || hw >= nh || m < 0 || m > Ng - 2) return;
        const double g = ::sqrt(1 + p.px * p.px + p.py * p.py + p.pz * p.pz);
        const double vx = p.px / g, vy = p.py / g, vz = p.pz / g;
        const double f = q * p.w / (hc[m + 2] * dx_);
        int ix[3], jn[3], jh[3]; double wx[3], wn[3], wh[3];
        tsc(p.x, dx_, nx, ix, wx);
        if (yfo) {                                   // Fourier y: all components on the periodic nodes
            tsc(p.y, dy_, ny, jn, wn);
            for (int v = 0; v < 3; ++v)
                for (int u = 0; u < 3; ++u) {
                    const long k = long(jn[v]) * nx + ix[u];
                    const double w = f * wx[u] * wn[v] * ia[0];
                    atomic_add(&J[(0L * nh + hw) * nkk + k], w * vx / vz);
                    atomic_add(&J[(1L * nh + hw) * nkk + k], w * vy / vz);
                    atomic_add(&J[(2L * nh + hw) * nkk + k], w);
                }
            return;
        }
        tsc_y(ya, p.y, jn, wn, jh, wh);
        for (int v = 0; v < 3; ++v) {
            const bool nod = jn[v] >= 1 && jn[v] <= ny - 1, hal = jh[v] >= 0 && jh[v] <= ny - 1;
            for (int u = 0; u < 3; ++u) {
                if (nod) {
                    const long k = long(jn[v]) * nx + ix[u];
                    const double w = f * wx[u] * wn[v] * ia[jn[v]];
                    atomic_add(&J[(0L * nh + hw) * nkk + k], w * vx / vz);
                    atomic_add(&J[(2L * nh + hw) * nkk + k], w);
                }
                if (hal) {
                    const long k = long(jh[v]) * nx + ix[u];
                    atomic_add(&J[(1L * nh + hw) * nkk + k], f * wx[u] * wh[v] * id[jh[v]] * vy / vz);
                }
            }
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
    // filter along zeta (on the charges h_m j_m) into the half-cells a-1 .. b-1; jx + i jy packed into S1, jz into jkz
    const bool zf = cfg.zeta_filter, zu = zax.uniform;
    cpx* s1 = S1.data(); cpx* jz = jkz.data();
    par_for(long(ns + 1) * nk, [=] HD(long idx) {
        const int hk = int(idx / nkk);
        const long i = idx % nkk;
        const int hc_ = a_ - 1 + hk, hw = hk + 1;
        double v[3] = {0, 0, 0};
        if (hc_ >= 0 && hc_ <= Ng - 2)
            for (int comp = 0; comp < 3; ++comp) {
                const double* Jc = J + size_t(comp) * nh * nkk;
                if (zf && zu) v[comp] = 0.25 * Jc[(hw - 1) * nkk + i] + 0.5 * Jc[hw * nkk + i] + 0.25 * Jc[(hw + 1) * nkk + i];
                else v[comp] = zf ? (0.25 * hc[hc_ + 1] * Jc[(hw - 1) * nkk + i] + 0.5 * hc[hc_ + 2] * Jc[hw * nkk + i] +
                                0.25 * hc[hc_ + 3] * Jc[(hw + 1) * nkk + i]) / hc[hc_ + 2]
                             : Jc[hw * nkk + i];
            }
        s1[idx] = C(v[0], v[1]);
        jz[idx] = C(v[2], 0);
    });
    fft_->run(S1.data(), ns + 1, false);
    fft_->run(jkz.data(), ns + 1, false);
    cpx* jx = jkx.data(); cpx* jy = jky.data();
    const double* fl = filt.data();
    if (yfo) {
        // Fourier y: unpack the pair, apply the transverse filter (zero at k = 0 and at the Nyquist modes)
        const double* fly = filty.data();
        par_for(long(ns + 1) * nk, [=] HD(long idx) {
            const long i = idx % nkk, base = idx - i;
            const int ixx = int(i % nx), j = int(i / nx);
            const long m = base + long((ny - j) % ny) * nx + (nx - ixx) % nx;
            const cpx z = s1[idx], zm = conj(s1[m]);
            const double f = (ixx == 0 && j == 0) ? 0.0 : fl[ixx] * fly[j];
            jx[idx] = f * (0.5 * (z + zm)); jy[idx] = f * (0.5 * C((z - zm).im, -(z - zm).re)); jz[idx] = f * jz[idx];
        });
        device_sync();
        exchange_jz_ghosts();
        return;
    }
    // free-hand y: unpack the pair
    par_for(long(ns + 1) * nk, [=] HD(long idx) {
        const long i = idx % nkk;
        const int ixx = int(i % nx);
        const long m = idx - ixx + (nx - ixx) % nx;
        const cpx z = s1[idx], zm = conj(s1[m]);
        jx[idx] = 0.5 * (z + zm); jy[idx] = 0.5 * C((z - zm).im, -(z - zm).re);
    });
    // transverse filters per y-line
    const bool kf = cfg.k_filter;
    const double Ly = cfg.Ly;
    par_for(long(ns + 1) * nx, [=] HD(long line) {
        const int ixx = int(line % nx);
        const long o = (line / nx) * nkk + ixx;          // element j at o + j*nx
        const double fx = fl[ixx];
        if (kf) {
            cpx* const nodal[2] = {jx, jz};
            for (int t = 0; t < 2; ++t) {                // nodes: charges Q_j = A_j J_j, Q = 0 at the walls
                cpx* A = nodal[t];
                cpx qm = C(0), q0 = A[o + nx] / ia[1];
                for (int j = 1; j < ny; ++j) {
                    const cpx qp = j + 1 < ny ? A[o + long(j + 1) * nx] / ia[j + 1] : C(0);
                    A[o + long(j) * nx] = (0.25 * qm + 0.5 * q0 + 0.25 * qp) * ia[j];
                    qm = q0; q0 = qp;
                }
            }
            cpx qm = jy[o] / id[0], q0 = qm;             // half-nodes: Q_j = Delta_j J_j, mirrored at the walls
            for (int j = 0; j < ny; ++j) {
                const cpx qp = j + 1 < ny ? jy[o + long(j + 1) * nx] / id[j + 1] : q0;
                jy[o + long(j) * nx] = (0.25 * qm + 0.5 * q0 + 0.25 * qp) * id[j];
                qm = q0; q0 = qp;
            }
        }
        cpx mean = C(0);
        if (ixx == 0) {
            for (int j = 0; j < ny; ++j) mean = mean + jy[o + long(j) * nx] / id[j];
            mean = mean / Ly;
        }
        jx[o] = C(0); jz[o] = C(0);
        for (int j = 0; j < ny; ++j) {
            const long k = o + long(j) * nx;
            jx[k] = fx * jx[k]; jz[k] = fx * jz[k]; jy[k] = fx * (jy[k] - mean);
        }
    });
    device_sync();
    exchange_jz_ghosts();
}

// filtered j_z of the half-cells a-2 and b from the neighbours (bound field of the sponge, d_zeta^2 rho)
void Simulation::exchange_jz_ghosts() {
    const size_t pl = size_t(nk);
    std::vector<cpx> lo(pl, C(0)), hi(pl, C(0));
    if (comm.rank < comm.size - 1) {
        std::vector<cpx> buf(pl);
        jkz.to_host(buf.data(), size_t(ns - 1) * pl, pl);
        comm.isend(comm.rank + 1, TAG_JZU, buf.data(), pl);
    }
    if (comm.rank > 0) {
        std::vector<cpx> buf(pl);
        jkz.to_host(buf.data(), pl, pl);
        comm.isend(comm.rank - 1, TAG_JZD, buf.data(), pl);
        comm.recv(comm.rank - 1, TAG_JZU, lo.data(), pl);
    }
    if (comm.rank < comm.size - 1) comm.recv(comm.rank + 1, TAG_JZD, hi.data(), pl);
    jzg.from_host(lo.data(), 0, pl); jzg.from_host(hi.data(), pl, pl);
}

// ---------------------------------------------------------------- field march (device code)
// Per k_x the y-line of slice n is coupled through the finite differences of y:
//   b+ f_j = (f_{j+1} - f_j)/Delta_j  (nodes -> half-node j+1/2, f = 0 at the walls j = 0, ny)
//   b- g_j = (g_j - g_{j-1})/A_j      (half-nodes -> node j; set to 0 on the wall nodes)
// a = i k_x is a number and commutes with b+-, so the cross terms cancel exactly as in the Fourier form, and
//   [1 + (h/4)(D + h/4)(k_x^2 - b- b+)] E_z = E_z(n+1) - (h/2) Q(n+1) - (h/4) S_E - h j_z   (nodes, Dirichlet)
//   [1 + (h/4)(D + h/4)(k_x^2 - b+ b-)] B_z = B_z(n+1) - (h/2) C(n+1) - (h/4) S_B           (half-nodes, Neumann)
// are two tridiagonal systems along y (Thomas). R and L follow by back-substitution.
namespace {
struct MarchCtx {
    cpx *Rx, *Ry, *Ez, *Bz, *Lx, *Ly;
    const cpx *bnd, *j0x, *j0y, *jkx, *jky, *jkz;
    cpx *sRx, *sRy, *sLx, *sLy, *srhs; double* scp;
    const double *kx, *ky, *iD, *iA, *hcg, *wlo, *whi;
    double D; int a, b, N, nx, ny; long nk;
};

// Fourier y. Slice n of level s1 for the mode i: R by the trapezoidal rule in s, L, E_z, B_z by the
// trapezoidal rule over the half-cell [n, n+1] of length h_n; B_z and E_z decouple and follow in closed
// form, then R and L. The values of slice n+1 (level s1) come from the arrays or, at the top of the block,
// from `bnd`.
HD inline void march_point(const MarchCtx& c, long i, int n) {
    const int l = n - c.a;
    const long o = long(l) * c.nk + i;
    const int ixx = int(i % c.nx), j = int(i / c.nx);
    const cpx a = ix(c.kx[ixx]), b = ix(c.ky[j]);
    const double k2 = c.kx[ixx] * c.kx[ixx] + c.ky[j] * c.ky[j], D = c.D, wl = c.wlo[n], wu = c.whi[n];
    const long od = o, ou = o + c.nk;                // half-cells n-1 and n
    const cpx Jx = wl * (c.j0x[od] + c.jkx[od]) + wu * (c.j0x[ou] + c.jkx[ou]);
    const cpx Jy = wl * (c.j0y[od] + c.jky[od]) + wu * (c.j0y[ou] + c.jky[ou]);
    const cpx Rsx = c.Rx[o] + D * (a * c.Ez[o] + b * c.Bz[o]) - D * Jx;
    const cpx Rsy = c.Ry[o] + D * (b * c.Ez[o] - a * c.Bz[o]) - D * Jy;
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
    const double h = c.hcg[n + 2];
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

// Free-hand y: the y-line of slice n for k_x index ik.
HD inline void march_line(const MarchCtx& c, int ik, int n) {
    const int nx = c.nx, ny = c.ny;
    const long P0 = long(n - c.a) * c.nk + ik;      // slice n of the block = half-cell n-1 (current planes)
    const long P1 = P0 + c.nk;                      // slice n+1 = half-cell n
    const cpx a = ix(c.kx[ik]);
    const double kx2 = c.kx[ik] * c.kx[ik], D = c.D, wl = c.wlo[n], wu = c.whi[n];
    const double* iD = c.iD; const double* iA = c.iA;
    auto S = [&](cpx* A, int j) -> cpx& { return A[long(j) * nx + ik]; };   // scratch line
    // known part of R (level s0 and the currents of both levels)
    for (int j = 0; j < ny; ++j) {
        const long o = P0 + long(j) * nx;
        if (j == 0) S(c.sRx, 0) = C(0);
        else {
            const cpx Jx = wl * (c.j0x[o] + c.jkx[o]) + wu * (c.j0x[o + c.nk] + c.jkx[o + c.nk]);
            const cpx bm = (c.Bz[o] - c.Bz[o - nx]) * iA[j];
            S(c.sRx, j) = c.Rx[o] + D * (a * c.Ez[o] + bm) - D * Jx;
        }
        const cpx Jy = wl * (c.j0y[o] + c.jky[o]) + wu * (c.j0y[o + c.nk] + c.jky[o + c.nk]);
        const cpx ezu = j + 1 < ny ? c.Ez[o + nx] : C(0);
        S(c.sRy, j) = c.Ry[o] + D * ((ezu - c.Ez[o]) * iD[j] - a * c.Bz[o]) - D * Jy;
    }
    if (n == c.N - 1) {             // front of the window: L = E_z = B_z = 0
        for (int j = 0; j < ny; ++j) {
            const long o = P0 + long(j) * nx;
            c.Rx[o] = S(c.sRx, j); c.Ry[o] = S(c.sRy, j); c.Ez[o] = c.Bz[o] = c.Lx[o] = c.Ly[o] = C(0);
        }
        return;
    }
    const bool top = n == c.b - 1;  // slice n+1 is the boundary slice from the rank above
    auto nxt = [&](int comp, const cpx* A, int j) -> cpx {
        if (j < 0 || j >= ny) return C(0);
        return top ? c.bnd[comp * c.nk + long(j) * nx + ik] : A[P1 + long(j) * nx];
    };
    auto pRx = [&](int j) { return nxt(0, c.Rx, j); }; auto pRy = [&](int j) { return nxt(1, c.Ry, j); };
    auto pEz = [&](int j) { return j == 0 ? C(0) : nxt(2, c.Ez, j); }; auto pBz = [&](int j) { return nxt(3, c.Bz, j); };
    auto pLx = [&](int j) { return nxt(4, c.Lx, j); }; auto pLy = [&](int j) { return nxt(5, c.Ly, j); };
    const double h = c.hcg[n + 2], cc = 0.25 * h * (D + 0.25 * h);
    // known part of L
    for (int j = 0; j < ny; ++j) {
        const long u = P1 + long(j) * nx;           // half-cell n, level s1
        if (j == 0) S(c.sLx, 0) = C(0);
        else {
            const cpx bm = (pBz(j) - pBz(j - 1)) * iA[j];
            S(c.sLx, j) = pLx(j) - 0.25 * h * (a * pEz(j) - bm) - 0.5 * h * c.jkx[u];
        }
        const cpx bp = (pEz(j + 1) - pEz(j)) * iD[j];
        S(c.sLy, j) = pLy(j) - 0.25 * h * (a * pBz(j) + bp) - 0.5 * h * c.jky[u];
    }
    // E_z on the nodes 1..ny-1
    double cp = 0;
    cpx dp = C(0);
    for (int j = 1; j < ny; ++j) {
        const cpx Qp = 0.5 * (-(a * (pRx(j) - pLx(j))) + ((pLy(j) - pRy(j)) - (pLy(j - 1) - pRy(j - 1))) * iA[j]);
        const cpx SE = -(a * (S(c.sRx, j) - S(c.sLx, j))) +
                       ((S(c.sLy, j) - S(c.sRy, j)) - (S(c.sLy, j - 1) - S(c.sRy, j - 1))) * iA[j];
        const cpx r = pEz(j) - 0.5 * h * Qp - 0.25 * h * SE - h * c.jkz[P1 + long(j) * nx];
        const double lo = j > 1 ? -cc * iA[j] * iD[j - 1] : 0, up = j < ny - 1 ? -cc * iA[j] * iD[j] : 0;
        const double den = 1 + cc * kx2 + cc * iA[j] * (iD[j - 1] + iD[j]) - lo * cp;
        dp = (r - lo * dp) / den;
        cp = up / den;
        S(c.srhs, j) = dp; c.scp[long(j) * nx + ik] = cp;
    }
    {
        cpx next = C(0);
        for (int j = ny - 1; j >= 1; --j) { next = S(c.srhs, j) - c.scp[long(j) * nx + ik] * next; c.Ez[P0 + long(j) * nx] = next; }
        c.Ez[P0] = C(0);
    }
    // B_z on the half-nodes 0..ny-1
    cp = 0; dp = C(0);
    for (int j = 0; j < ny; ++j) {
        const cpx fu = j + 1 < ny ? pRx(j + 1) + pLx(j + 1) : C(0), f0 = pRx(j) + pLx(j);
        const cpx Cp = 0.5 * (a * (pRy(j) + pLy(j)) - (fu - f0) * iD[j]);
        const cpx su = j + 1 < ny ? S(c.sRx, j + 1) + S(c.sLx, j + 1) : C(0), s0 = S(c.sRx, j) + S(c.sLx, j);
        const cpx SB = a * (S(c.sRy, j) + S(c.sLy, j)) - (su - s0) * iD[j];
        const cpx r = pBz(j) - 0.5 * h * Cp - 0.25 * h * SB;
        const double al = j > 0 ? iA[j] : 0, au = j < ny - 1 ? iA[j + 1] : 0;
        const double lo = -cc * iD[j] * al, up = -cc * iD[j] * au;
        const double den = 1 + cc * kx2 + cc * iD[j] * (al + au) - lo * cp;
        dp = (r - lo * dp) / den;
        cp = up / den;
        S(c.srhs, j) = dp; c.scp[long(j) * nx + ik] = cp;
    }
    {
        cpx next = C(0);
        for (int j = ny - 1; j >= 0; --j) { next = S(c.srhs, j) - c.scp[long(j) * nx + ik] * next; c.Bz[P0 + long(j) * nx] = next; }
    }
    // R and L
    for (int j = 0; j < ny; ++j) {
        const long o = P0 + long(j) * nx;
        if (j == 0) { c.Rx[o] = C(0); c.Lx[o] = C(0); }
        else {
            const cpx bm = (c.Bz[o] - c.Bz[o - nx]) * iA[j];
            c.Rx[o] = S(c.sRx, j) + D * (a * c.Ez[o] + bm);
            c.Lx[o] = S(c.sLx, j) - 0.25 * h * (a * c.Ez[o] - bm);
        }
        const cpx ezu = j + 1 < ny ? c.Ez[o + nx] : C(0);
        const cpx bp = (ezu - c.Ez[o]) * iD[j];
        c.Ry[o] = S(c.sRy, j) + D * (bp - a * c.Bz[o]);
        c.Ly[o] = S(c.sLy, j) - 0.25 * h * (a * c.Bz[o] + bp);
    }
}
}  // namespace

// The march of this block: every mode (Fourier y) or k_x line (free-hand y) is an independent recurrence over
// the slices, from b-1 down to a. GPU: one thread per mode or line; CPU: slice by slice, OpenMP over them.
void Simulation::march(double D) {
    const size_t pl = size_t(nk);
    if (comm.rank < comm.size - 1) {
        std::vector<cpx> buf(6 * pl);
        comm.recv(comm.rank + 1, TAG_BND, buf.data(), buf.size());
        bnd.from_host(buf.data(), 0, buf.size());
    }
    MarchCtx c{Rx.data(), Ry.data(), Ez.data(), Bz.data(), Lx.data(), Ly.data(), bnd.data(), j0x.data(), j0y.data(),
               jkx.data(), jky.data(), jkz.data(), sRx.data(), sRy.data(), sLx.data(), sLy.data(), srhs.data(), scp.data(),
               kxd.data(), yf ? kyd.data() : nullptr, iD.data(), iA.data(), hcg.data(), wlo.data(), whi.data(), D, a, b, N,
               cfg.nx, cfg.ny, long(nk)};
    const int lo = a, hi = b;
    if (yf) {
#if GEMPIC_GPU
        par_for(nk, [=] HD(long i) { for (int n = hi - 1; n >= lo; --n) march_point(c, i, n); });
#else
        for (int n = hi - 1; n >= lo; --n) par_for(nk, [=](long i) { march_point(c, i, n); });
#endif
    } else {
#if GEMPIC_GPU
        par_for(cfg.nx, [=] HD(long i) { for (int n = hi - 1; n >= lo; --n) march_line(c, int(i), n); });
#else
        for (int n = hi - 1; n >= lo; --n) par_for(cfg.nx, [=](long i) { march_line(c, int(i), n); });
#endif
    }
    device_sync();
    if (comm.rank > 0) {           // slice a of level s1 is the boundary of the block below
        std::vector<cpx> buf(6 * pl);
        int k = 0;
        for (Buf<cpx>* p : {&Rx, &Ry, &Ez, &Bz, &Lx, &Ly}) p->to_host(&buf[pl * k++], 0, pl);
        comm.isend(comm.rank - 1, TAG_BND, buf.data(), buf.size());
    }
}

// Co-moving field of the bunch: -(lap_perp + gamma^-2 d_zeta^2) psi = rho, R = (1 + beta) E_perp,
// E_perp = -grad psi. In the eigenvectors of -Lap_y (eigenvalue mu) each (k_x, mu) is one tridiagonal (Thomas)
// solve along the free-hand zeta grid, pipelined over the ranks.
void Simulation::comoving_field() {
    double sums[2] = {0, 0};
    for (auto& p : host_particles())
        if (p.alive == 1) { sums[0] += p.w * p.pz / std::sqrt(1 + p.px * p.px + p.py * p.py + p.pz * p.pz); sums[1] += p.w; }
    comm.allreduce_sum(sums, 2);
    const double beta = sums[1] > 0 ? sums[0] / sums[1] : 1;
    const double gm2 = 1 - beta * beta;
    beta_b = beta; gm2_b = gm2;
    const size_t pl = size_t(nk);
    const long nkk = nk;
    const int nx = cfg.nx, ny = cfg.ny;
    // free-hand y: half-cell charges to the eigen-coefficients (slot s = eigenvalue s - 1; slot 0 stays 0).
    // Fourier y: the spectra themselves (slot = iy, eigenvalue k_y^2).
    const bool yfo = yf;
    Buf<cpx> E1(yf ? 0 : (ns + 1) * pl);
    if (!yf) {
        cpx* e1 = E1.data(); const cpx* jz = jkz.data(); const double* ef = evf.data();
        par_for(long(ns + 1) * nk, [=] HD(long idx) {
            const long i = idx % nkk, base = idx - i;
            const int s = int(i / nx), ixx = int(i % nx);
            cpx v = C(0);
            if (s > 0) for (int j = 1; j < ny; ++j) v = v + ef[long(s) * ny + j] * jz[base + long(j) * nx + ixx];
            e1[idx] = v;
        });
        device_sync();
    }
    // forward elimination: cp in S1, dp in S2; the last row of the block below arrives in bnd[0], bnd[1]
    if (comm.rank > 0) {
        std::vector<cpx> buf(2 * pl);
        comm.recv(comm.rank - 1, TAG_TRIF, buf.data(), buf.size());
        bnd.from_host(buf.data(), 0, buf.size());
    }
    cpx* cp = S1.data(); cpx* dp = S2.data();
    const cpx* e1 = yf ? jkz.data() : E1.data(); const cpx* bd = bnd.data();
    const double* kx = kxd.data(); const double* ev = evl.data();
    const double* hc = hcg.data(); const double* wl = wlo.data(); const double* wu = whi.data();
    const int a_ = a, ns_ = ns, first = comm.rank == 0;
    par_for(nk, [=] HD(long i) {
        const int s = int(i / nx), ixx = int(i % nx);
        cpx cprev = first ? C(0) : bd[i], dprev = first ? C(0) : bd[nkk + i];
        for (int l = 0; l < ns_; ++l) {
            const long o = l * nkk + i;
            const int n = a_ + l;
            const double hm = hc[n + 1], hp = hc[n + 2];
            const double lo = -gm2 * 2 / ((hm + hp) * hm), up = -gm2 * 2 / ((hm + hp) * hp);
            const double dg = kx[ixx] * kx[ixx] + ev[s] + gm2 * 2 / (hm * hp);
            if ((!yfo && s == 0) || dg == 0) { cp[o] = C(0); dp[o] = C(0); continue; }
            const cpx rhs = (1 / beta) * (wl[n] * e1[o] + wu[n] * e1[o + nkk]);
            if (n == 0) { cprev = C(0); dprev = C(0); }
            const cpx den = C(dg) - lo * cprev;
            cp[o] = C(up) / den; dp[o] = (rhs - lo * dprev) / den;
            cprev = cp[o]; dprev = dp[o];
        }
    });
    device_sync();
    if (comm.rank < comm.size - 1) {
        std::vector<cpx> buf(2 * pl);
        S1.to_host(buf.data(), (ns - 1) * pl, pl); S2.to_host(&buf[pl], (ns - 1) * pl, pl);
        comm.isend(comm.rank + 1, TAG_TRIF, buf.data(), buf.size());
    }
    // back substitution, psi overwrites dp; psi of slice b arrives from the block above
    std::vector<cpx> topv(pl, C(0));
    if (comm.rank < comm.size - 1) comm.recv(comm.rank + 1, TAG_TRIB, topv.data(), pl);
    bnd.from_host(topv.data(), 0, pl);
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
    // back to the nodes (psi in S1), then R
    if (yf) {
        cpx* t = cp;
        par_for(long(ns) * nk, [=] HD(long idx) { t[idx] = dp[idx]; });
        device_sync();
    } else {
        const double* eb = evb.data();
        par_for(long(ns) * nk, [=] HD(long idx) {
            const long i = idx % nkk, base = idx - i;
            const int j = int(i / nx), ixx = int(i % nx);
            cpx v = C(0);
            if (j > 0) for (int s = 1; s < ny; ++s) v = v + eb[long(j) * ny + s] * dp[base + long(s) * nx + ixx];
            cp[idx] = v;
        });
        device_sync();
    }
    cpx* rx = Rx.data(); cpx* ry = Ry.data(); cpx* ez = Ez.data(); cpx* bz = Bz.data();
    const double* id = iD.data(); const double* ky = yf ? kyd.data() : nullptr;
    par_for(long(ns) * nk, [=] HD(long idx) {
        const long i = idx % nkk;
        const int j = int(i / nx), ixx = int(i % nx);
        rx[idx] = -(1 + beta) * (ix(kx[ixx]) * cp[idx]);
        if (yfo) ry[idx] = -(1 + beta) * (ix(ky[j]) * cp[idx]);
        else {
            const cpx up = j + 1 < ny ? cp[idx + nx] : C(0);
            ry[idx] = -(1 + beta) * ((up - cp[idx]) * id[j]);
        }
        ez[idx] = C(0); bz[idx] = C(0);
    });
    device_sync();
}

// Sponge on the radiation part R - R_b of the forward field (R_b = (1 + beta) E_perp of the charge of the
// slice, T psi = rho with T = k_x^2 + k_y^2 (Fourier y) or k_x^2 - Lap_y (free-hand y), plus the first
// 1/gamma^2 correction), then E and B of the own slices in real space, then the ghost slices.
void Simulation::sponge_and_real_fields() {
    const long nkk = nk;
    const int nx = cfg.nx, ny = cfg.ny;
    cpx* rx = Rx.data(); cpx* ry = Ry.data();
    cpx* s1 = S1.data(); cpx* s2 = S2.data();
    const cpx* jz = jkz.data();
    const double* kx = kxd.data(); const double* id = iD.data(); const double* ia = iA.data();
    if (cfg.sponge > 0 || cfg.sponge_y > 0) {
        const double* pc = pcp.data(); const double* pd = pid.data();
        const double* wl = wlo.data(); const double* wu = whi.data();
        const int a_ = a;
        // psi of the slice charge per y-line (LU factors of T = k_x^2 - Lap_y precomputed), into S2; R - R_b
        // packed into S1. The bound field includes the first correction in 1/gamma^2 of the co-moving field,
        // psi = T^-1 (rho + gamma^-2 T^-1 d_zeta^2 rho), and R_b = (1 + beta) E_perp.
        const double* hc = hcg.data();
        const double g2 = gm2_b, opb = 1 + beta_b;
        const cpx* jg = jzg.data();
        const int ns_ = ns, Ng = N;
        const bool yfo = yf;
        const double* ky = yf ? kyd.data() : nullptr;
        if (yfo) par_for(long(ns) * nk, [=] HD(long idx) {   // Fourier y: T = k^2 is a number
            const long i = idx % nkk;
            const int ixx = int(i % nx), j = int(i / nx), l = int(idx / nkk), n = a_ + l;
            auto half = [&](int hk) -> cpx {
                if (hk < 0) return jg[i];
                if (hk > ns_) return jg[nkk + i];
                return jz[long(hk) * nkk + i];
            };
            auto rho_at = [&](int dl) -> cpx {
                const int nn = n + dl;
                if (nn < 0 || nn > Ng - 1) return C(0);
                return wl[nn] * half(l + dl) + wu[nn] * half(l + dl + 1);
            };
            const double hm = hc[n + 1], hp = hc[n + 2];
            const double cm = 2 / ((hm + hp) * hm), cp_ = 2 / ((hm + hp) * hp);
            const double k2 = kx[ixx] * kx[ixx] + ky[j] * ky[j];
            cpx psi = C(0);
            if (k2 > 0) {
                const cpx d2 = g2 * (cp_ * rho_at(1) - (cm + cp_) * rho_at(0) + cm * rho_at(-1));
                psi = (rho_at(0) + d2 / k2) / k2;
            }
            s2[idx] = psi;
            const cpx bx = -opb * (ix(kx[ixx]) * psi), by = -opb * (ix(ky[j]) * psi);
            const cpx u = rx[idx] - bx, v = ry[idx] - by;
            s1[idx] = C(u.re - v.im, u.im + v.re);
        });
        else par_for(long(ns) * nx, [=] HD(long line) {
            const int ixx = int(line % nx), l = int(line / nx), n = a_ + l;
            const long o = long(l) * nkk + ixx;
            auto half = [&](int hk, long jo) -> cpx {        // half-cell plane hk = -1 .. ns+1 (ghosts a-2 and b)
                if (hk < 0) return jg[ixx + jo];
                if (hk > ns_) return jg[nkk + ixx + jo];
                return jz[long(hk) * nkk + ixx + jo];
            };
            auto rho_at = [&](int dl, long jo) -> cpx {     // charge of slice n + dl at y-index offset jo
                const int nn = n + dl;
                if (nn < 0 || nn > Ng - 1) return C(0);
                return wl[nn] * half(l + dl, jo) + wu[nn] * half(l + dl + 1, jo);
            };
            const double hm = hc[n + 1], hp = hc[n + 2];
            const double cm = 2 / ((hm + hp) * hm), cp_ = 2 / ((hm + hp) * hp);
            auto solve = [&](bool second) {               // T^-1 of (rho or its d_zeta^2 / gamma^2 + ...) into s2
                cpx dpv = C(0);
                for (int j = 1; j < ny; ++j) {
                    const long jo = long(j) * nx, k = o + jo;
                    cpx r;
                    if (!second) r = g2 * (cp_ * rho_at(1, jo) - (cm + cp_) * rho_at(0, jo) + cm * rho_at(-1, jo));
                    else r = rho_at(0, jo) + s2[k];
                    const double lo = j > 1 ? -ia[j] * id[j - 1] : 0;
                    dpv = (r - lo * dpv) * pd[long(j) * nx + ixx];
                    s2[k] = dpv;
                }
                cpx next = C(0);
                for (int j = ny - 1; j >= 1; --j) { next = s2[o + long(j) * nx] - pc[long(j) * nx + ixx] * next; s2[o + long(j) * nx] = next; }
            };
            solve(false);
            solve(true);
            s2[o] = C(0);
            const cpx a = ix(kx[ixx]);
            for (int j = 0; j < ny; ++j) {
                const long k = o + long(j) * nx;
                const cpx pu = j + 1 < ny ? s2[k + nx] : C(0);
                const cpx bx = -opb * (a * s2[k]), by = -opb * ((pu - s2[k]) * id[j]);
                const cpx u = rx[k] - bx, v = ry[k] - by;
                s1[k] = C(u.re - v.im, u.im + v.re);           // u + i v
            }
        });
        fft_->run(s1, ns, true);
        const double* mx = maskx.data(); const double* mn = masky_n.data(); const double* mh = masky_h.data();
        par_for(long(ns) * nk, [=] HD(long idx) {
            const long i = idx % nkk;
            const int j = int(i / nx), ixx = int(i % nx);
            s1[idx] = C(mx[ixx] * mn[j] * s1[idx].re, mx[ixx] * mh[j] * s1[idx].im);
        });
        fft_->run(s1, ns, false);
        if (yfo) par_for(long(ns) * nk, [=] HD(long idx) {
            const long i = idx % nkk, base = idx - i;
            const int j = int(i / nx), ixx = int(i % nx);
            const long m = base + long((ny - j) % ny) * nx + (nx - ixx) % nx;
            const cpx z = s1[idx], zm = conj(s1[m]);
            rx[idx] = 0.5 * (z + zm) - opb * (ix(kx[ixx]) * s2[idx]);
            ry[idx] = 0.5 * C((z - zm).im, -(z - zm).re) - opb * (ix(ky[j]) * s2[idx]);
        });
        else par_for(long(ns) * nk, [=] HD(long idx) {
            const long i = idx % nkk;
            const int j = int(i / nx), ixx = int(i % nx);
            const long m = idx - ixx + (nx - ixx) % nx;
            const cpx z = s1[idx], zm = conj(s1[m]);
            const cpx pu = j + 1 < ny ? s2[idx + nx] : C(0);
            const cpx bx = -opb * (ix(kx[ixx]) * s2[idx]), by = -opb * ((pu - s2[idx]) * id[j]);
            rx[idx] = j == 0 ? C(0) : 0.5 * (z + zm) + bx;
            ry[idx] = 0.5 * C((z - zm).im, -(z - zm).re) + by;
        });
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
        Fp[0 * fpl + idx] = float(0.5 * (r1 + l1));      // E_x  (nodes)
        Fp[1 * fpl + idx] = float(0.5 * (r2 + l2));      // E_y  (half-nodes of the free-hand y)
        Fp[3 * fpl + idx] = float(0.5 * (l2 - r2));      // B_x  (half-nodes of the free-hand y)
        Fp[4 * fpl + idx] = float(0.5 * (r1 - l1));      // B_y  (nodes)
    });
    const cpx* ez = Ez.data(); const cpx* bz = Bz.data();
    par_for(long(ns) * nk, [=] HD(long idx) { s1[idx] = C(ez[idx].re - bz[idx].im, ez[idx].im + bz[idx].re); });
    fft_->run(s1, ns, true);
    par_for(long(ns) * nk, [=] HD(long idx) {
        Fp[2 * fpl + idx] = float(s1[idx].re);           // E_z  (nodes)
        Fp[5 * fpl + idx] = float(s1[idx].im);           // B_z  (half-nodes of the free-hand y)
    });
    device_sync();
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

// The transverse window follows the bunch centroid in whole cells (in x; in y only with Fourier y). The
// centroid of step k is reduced over the ranks without blocking and used at step k + window_lag, so the ranks
// need not wait for each other.
void Simulation::window_shift_post() {
    if (!cfg.move_x && !(yf && cfg.move_y)) return;
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
    if (!cfg.move_x && !(yf && cfg.move_y)) return;
    if (long(cent_.size()) < cfg.window_lag) return;       // centroid of step istep + 1 - lag not yet due
    comm.wait(cent_.front().first);
    const std::vector<double> c = cent_.front().second;
    cent_.pop_front();
    if (c[2] <= 0) return;
    const int nxs = cfg.move_x ? int(std::lround((c[0] / c[2] - (X0 + 0.5 * cfg.Lx)) / dx)) : 0;
    const int nys = yf && cfg.move_y ? int(std::lround((c[1] / c[2] - (Y0 + 0.5 * cfg.Ly)) / dy)) : 0;
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
        const int nx = cfg.nx, ny = cfg.ny;
        const double* kx = kxs.data(); const double* ky = yf ? kys.data() : nullptr;
        auto rot = [&](Buf<cpx>& A) {
            cpx* p = A.data();
            par_for(long(A.size()), [=] HD(long idx) {
                const long i = idx % nkk;
                const double ph = kx[i % nx] * sx + (nys ? ky[i / nx] * sy : 0.0);
                p[idx] = p[idx] * C(::cos(ph), ::sin(ph));
            });
        };
        rot(Rx); rot(Ry); rot(Ez); rot(Bz); rot(j0x); rot(j0y);
        Buf<float> tmp(F.size());
        float* t = tmp.data(); const float* f = F.data();
        par_for(long(F.size()), [=] HD(long idx) {
            const long i = idx % nkk, base = idx - i;
            const int ixx = int(i % nx), j = int(i / nx);
            const int sxx = ((ixx + nxs) % nx + nx) % nx, syy = ((j + nys) % ny + ny) % ny;
            t[idx] = f[base + long(syy) * nx + sxx];
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

// field energy of the own slices in code units (dual-cell weights along zeta and y)
double Simulation::local_field_energy() {
    Buf<double> acc(1);
    double* A = acc.data();
    const float* Fp = F.data();
    const long nkk = nk, fpl = long(ns + 2) * nk;
    const int a_ = a, ns_ = ns, nx = cfg.nx;
    const double* ws = wsl.data(); const double* ia = iA.data(); const double* id = iD.data();
    par_for(nk, [=] HD(long i) {
        const int j = int(i / nx);
        const double wn = 1 / ia[j], wh = 1 / id[j];
        double e = 0;
        for (int l = 0; l < ns_; ++l) {
            const double w = ws[a_ + l];
            for (int comp = 0; comp < 6; ++comp) {
                const double v = Fp[comp * fpl + l * nkk + i];
                e += 0.5 * w * (is_half(comp) ? wh : wn) * v * v;
            }
        }
        atomic_add(A, e);
    });
    device_sync();
    return acc.host()[0] * dx;
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
