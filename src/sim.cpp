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

double Config::plasma_density(double x, double y, double z) const {
    if (!plasma_on) return 0;
    const double r2 = x * x + y * y;
    if (r2 >= r_max * r_max) return 0;
    double f = 1;
    if (!pz_points.empty()) {
        if (z <= pz_points.front()) f = pz_values.front();
        else if (z >= pz_points.back()) f = pz_values.back();
        else {
            size_t k = 1;
            while (pz_points[k] < z) ++k;
            const double t = (z - pz_points[k - 1]) / (pz_points[k] - pz_points[k - 1]);
            f = (1 - t) * pz_values[k - 1] + t * pz_values[k];
        }
    }
    const double g = parabolic ? 1 + channel_depth * r2 / (r_channel * r_channel) : 1.0;
    return f * g > 0 ? n0 * f * g : 0.0;
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

    // laser blocks: laser, laser2, ... (type plane_wave: the analytic external wave, at most one; gaussian: on the grid)
    for (int k = 1;; ++k) {
        const std::string b = k == 1 ? "laser" : "laser" + std::to_string(k);
        if (!d.has_block(b)) { if (k == 1) continue; break; }
        const std::string type = d.str(b, "type", "plane_wave");
        if (type == "plane_wave") {
            if (c.laser_on || c.a0 != 0) throw std::runtime_error(b + ": only one laser of type plane_wave");
            c.laser_on = d.flag(b, "enabled", true);
            c.a0 = d.num(b, "a0", Kind::None);
            c.kL = c.lambda0 / d.num(b, "wavelength", Kind::Length, c.lambda0);
            c.theta = d.num(b, "angle", Kind::Angle, 0.0);
            c.tau = d.num(b, "tau_fwhm", Kind::Time) / c.Tu;
            c.cep = d.num(b, "cep", Kind::Angle, 0.0);
            c.pol = d.str(b, "polarization", "s");
            if (c.pol != "s" && c.pol != "p" && c.pol != "circular")
                throw std::runtime_error(b + ".polarization must be s, p or circular");
            c.start_fwhm = d.num(b, "start_fwhm", Kind::None, 3.5);
            c.stop_fwhm = d.num(b, "stop_fwhm", Kind::None, c.start_fwhm);   // 0: run to z_end
        } else if (type == "gaussian") {
            if (!d.flag(b, "enabled", true)) continue;
            GridLaser g;
            g.a0 = d.num(b, "a0", Kind::None);
            g.kL = c.lambda0 / d.num(b, "wavelength", Kind::Length, c.lambda0);
            g.waist = L(b, "waist");
            g.tau = d.num(b, "tau_fwhm", Kind::Time) / c.Tu;
            g.zeta_c = L(b, "zeta_center");
            g.z_focus = Ld(b, "z_focus", 0.0);
            g.cep = d.num(b, "cep", Kind::Angle, 0.0);
            const std::string pol = d.str(b, "polarization", "linear");
            if (pol == "circular") g.circular = true;
            else if (pol != "linear") throw std::runtime_error(b + ".polarization must be linear or circular");
            g.pol_angle = d.num(b, "pol_angle", Kind::Angle, 0.0);
            g.helicity = int(d.integer(b, "helicity", 1));
            if (g.helicity != 1 && g.helicity != -1) throw std::runtime_error(b + ".helicity must be 1 or -1");
            c.glasers.push_back(g);
        } else throw std::runtime_error(b + ".type must be plane_wave or gaussian");
    }
    if (d.has_block("plasma")) {
        c.plasma_on = true;
        const std::string dn = d.str("plasma", "density");
        if (dn.size() > 3 && dn.compare(dn.size() - 3, 3, "n_c") == 0)
            c.n0 = parse_with_unit(dn.substr(0, dn.size() - 3), Kind::None, "plasma.density");
        else c.n0 = d.num("plasma", "density", Kind::Density) / c.nc;
        if (!(c.n0 > 0)) throw std::runtime_error("plasma.density must be positive");
        if (d.has("plasma", "z_points") || d.has("plasma", "z_values")) {
            c.pz_points = d.list("plasma", "z_points", Kind::Length);
            for (double& v : c.pz_points) v /= c.Lu;
            c.pz_values = d.list("plasma", "z_values", Kind::None);
            if (c.pz_points.size() != c.pz_values.size() || c.pz_points.empty())
                throw std::runtime_error("plasma: z_points and z_values need the same number of entries");
            for (size_t k = 1; k < c.pz_points.size(); ++k)
                if (!(c.pz_points[k] > c.pz_points[k - 1])) throw std::runtime_error("plasma.z_points must increase");
        }
        const std::string prof = d.str("plasma", "profile", "uniform");
        if (prof == "parabolic") {
            c.parabolic = true;
            c.channel_depth = d.num("plasma", "channel_depth", Kind::None);
            c.r_channel = L("plasma", "r_channel");
        } else if (prof != "uniform") throw std::runtime_error("plasma.profile must be uniform or parabolic");
        c.r_max = Ld("plasma", "r_max", 1e300);
        c.ppc_x = int(d.integer("plasma", "ppc_x", 2)); c.ppc_y = int(d.integer("plasma", "ppc_y", c.ppc_x));
        if (c.ppc_x < 1 || c.ppc_y < 1) throw std::runtime_error("plasma: ppc_x, ppc_y must be >= 1");
        c.smooth_length = Ld("plasma", "smooth_length", 0.0);
        if (c.smooth_length < 0) throw std::runtime_error("plasma.smooth_length must be >= 0");
    }
    c.iterations = int(d.integer("numerics", "iterations", 2));
    c.vz_min = d.num("numerics", "vz_min", Kind::None, 0.8);
    c.vz_beam = d.num("numerics", "vz_beam", Kind::None, 0.9);
    c.trap_advance = d.num("numerics", "trap_advance", Kind::None, 1.0);
    {
        const std::string pr = d.str("numerics", "predictor", "extrapolate");
        if (pr == "previous") c.predictor_extrapolate = false;
        else if (pr != "extrapolate") throw std::runtime_error("numerics.predictor must be extrapolate or previous");
    }
    if (c.iterations < 1) throw std::runtime_error("numerics.iterations must be >= 1");
    if (!(c.vz_min > 0 && c.vz_min < c.vz_beam && c.vz_beam < 1)) throw std::runtime_error("numerics: need 0 < vz_min < vz_beam < 1");
    if (!(c.trap_advance >= 1)) throw std::runtime_error("numerics.trap_advance must be >= 1");
    if ((c.plasma_on || !c.glasers.empty()) && !c.self_fields)
        throw std::runtime_error("numerics.self_fields = false is for test particles in the analytic laser only (no plasma, no grid lasers)");
    if (d.has_block("bunch")) {
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
    } else {
        c.nmacro = 0; c.charge = 0; c.gamma = 1; c.zeta_center = 0.75 * c.Lwin;
    }
    c.dir = d.str("output", "dir", "out");
    c.every = int(d.integer("output", "every", 10));
    c.dump_particles = d.flag("output", "particles", true);
    c.frames = d.flag("output", "frames", false);
    c.axis = d.flag("output", "axis", false);
    c.freeze_bunch = d.flag("numerics", "freeze_bunch", false);
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
    for (size_t k = 0; k < glasers.size(); ++k)
        std::fprintf(f, "  grid laser %zu: a0 = %g, lambda = %.4g um, waist = %.4g um, tau_FWHM = %.4g fs, centre at zeta = %.4g um, focus at z = %.4g um, %s\n",
                     k + 1, glasers[k].a0, lambda0 / glasers[k].kL * 1e6, glasers[k].waist * um, glasers[k].tau * Tu * 1e15,
                     glasers[k].zeta_c * um, glasers[k].z_focus * um, glasers[k].circular ? "circular" : "linear");
    if (plasma_on)
        std::fprintf(f, "  plasma: n0 = %.4g cm^-3 (%.4g n_c), %s, r_max = %.4g um, %d x %d per cell, smooth_length %.4g um; %d passes per slice, predictor %s, vz_min %.3g, vz_beam %.3g\n",
                     n0 * nc * 1e-6, n0, parabolic ? "parabolic channel" : "uniform", r_max < 1e299 ? r_max * um : -1.0, ppc_x, ppc_y,
                     smooth_length * um, iterations, predictor_extrapolate ? "extrapolate" : "previous", vz_min, vz_beam);
    std::fprintf(f, "  output: dir '%s', every %d steps (%ld steps to z_end)%s%s\n", dir.c_str(), every,
                 long(std::ceil(z_end / dz - 1e-9)), frames ? ", frames" : "", axis ? ", axis lineouts" : "");
    if (freeze_bunch) std::fprintf(f, "  bunch frozen (numerics.freeze_bunch): not pushed, a rigid source\n");
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
    setup_plasma();
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
    zs.upload(Z); hcg.upload(g); wlo.upload(lo); whi.upload(hi); wsl.upload(ws); cfg_wlo_ = lo;
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
            p.q = q; p.qm = q; p.sp = SP_BUNCH; p.z = 0;
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
    Laser laser; double X0, zc, dz_push, push_phase, zhand, zmax, ylo, yhi, s1, vz_min; int freeze;
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
HD inline bool rhs(const PushCtx& c, const State& Y, double z, double qm, State& d) {
    const double g = ::sqrt(1 + Y.px * Y.px + Y.py * Y.py + Y.pz * Y.pz);
    if (Y.pz < 0.3 * g) return false;
    const double gm = (1 + Y.px * Y.px + Y.py * Y.py) / (g + Y.pz);
    double E[3], B[3], Ee[3], Be[3];
    gather(c, ::fmod(::fmod(Y.x, c.Lx) + c.Lx, c.Lx), c.yf ? ::fmod(::fmod(Y.y, c.Ly) + c.Ly, c.Ly) : Y.y, Y.zeta, E, B);
    c.laser.field(Y.x + c.X0, z, z - (Y.zeta - c.zc), Ee, Be);
    for (int k = 0; k < 3; ++k) { E[k] += Ee[k]; B[k] += Be[k]; }
    const double ip = 1 / Y.pz;
    d.x = Y.px * ip; d.y = Y.py * ip; d.zeta = -gm * ip;
    d.px = qm * (g * E[0] + Y.py * B[2] - Y.pz * B[1]) * ip;
    d.py = qm * (g * E[1] + Y.pz * B[0] - Y.px * B[2]) * ip;
    d.pz = qm * (g * E[2] + Y.px * B[1] - Y.py * B[0]) * ip;
    return true;
}
HD inline State axpy(const State& a, double h, const State& b) {
    return State{a.x + h * b.x, a.y + h * b.y, a.zeta + h * b.zeta, a.px + h * b.px, a.py + h * b.py, a.pz + h * b.pz};
}
// RK4 from z0 to the plane s1. A sub-step is at most dz_push and at most push_phase of the laser phase.
// A particle that enters the half-cells of the rank below (zeta < zhand) at the start of a sub-step is
// handed over with the rest of the step.
HD inline void push_one(const PushCtx& c, Particle& p, double z0) {
    if (c.freeze && p.sp == SP_BUNCH) { p.z = c.s1; return; }   // numerics.freeze_bunch: rigid, (x, y, zeta, p) fixed
    State Y{p.x, p.y, p.zeta, p.px, p.py, p.pz};
    double z = z0;
    while (z < c.s1 - 1e-12 * ::fabs(c.s1)) {
        if (Y.zeta < c.zhand) {
            p.x = Y.x; p.y = Y.y; p.zeta = Y.zeta; p.px = Y.px; p.py = Y.py; p.pz = Y.pz;
            p.alive = 2; p.rem = c.s1 - z;
            return;
        }
        if (c.vz_min > 0 && Y.pz < c.vz_min * ::sqrt(1 + Y.px * Y.px + Y.py * Y.py + Y.pz * Y.pz)) {
            // too slow for the beam list: streams in the current march from (z, zeta)
            p.x = Y.x; p.y = Y.y; p.zeta = Y.zeta; p.px = Y.px; p.py = Y.py; p.pz = Y.pz; p.z = z;
            p.alive = 4;
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
        const double qm = p.qm;
        if (!(rhs(c, Y, z, qm, k1) && rhs(c, axpy(Y, 0.5 * ds, k1), z + 0.5 * ds, qm, k2) &&
              rhs(c, axpy(Y, 0.5 * ds, k2), z + 0.5 * ds, qm, k3) && rhs(c, axpy(Y, ds, k3), z + ds, qm, k4))) {
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
    p.x = Y.x; p.y = Y.y; p.zeta = Y.zeta; p.px = Y.px; p.py = Y.py; p.pz = Y.pz; p.z = c.s1;   // continuous coordinates
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
    c.zmax = cfg.zeta[N - 1]; c.s1 = s1; c.vz_min = cfg.plasma_on ? cfg.vz_min : 0; c.freeze = cfg.freeze_bunch;
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
        const double f = p.q * p.w / (hc[m + 2] * dx_);
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
    const bool keep0 = cfg.plasma_on;     // with plasma the k = 0 current is a 1D plasma wave, not a net current
    if (yfo) {
        // Fourier y: unpack the pair, apply the transverse filter (zero at the Nyquist modes, and at k = 0 in vacuum)
        const double* fly = filty.data();
        par_for(long(ns + 1) * nk, [=] HD(long idx) {
            const long i = idx % nkk, base = idx - i;
            const int ixx = int(i % nx), j = int(i / nx);
            const long m = base + long((ny - j) % ny) * nx + (nx - ixx) % nx;
            const cpx z = s1[idx], zm = conj(s1[m]);
            const double f = (ixx == 0 && j == 0 && !keep0) ? 0.0 : fl[ixx] * fly[j];
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
        if (ixx == 0 && !keep0) {
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
        add_grid_lasers();
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
    if (cfg.plasma_on) collect_restreaming();
    s += cfg.dz; ++istep;
    if (cfg.self_fields && cfg.plasma_on) {
        currents();
        march_plasma(s - cfg.dz, s);
        sponge_and_real_fields();
        const long n = long(ns + 1) * nk;
        cpx* a0 = j0x.data(); cpx* b0 = j0y.data(); const cpx* a1 = jkx.data(); const cpx* b1 = jky.data();
        par_for(n, [=] HD(long i) { a0[i] = a1[i]; b0[i] = b1[i]; });
        device_sync();
    } else if (cfg.self_fields) solve_fields(0.5 * cfg.dz);
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
        if (P[i].alive != 1 || P[i].sp != SP_BUNCH) return;
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
    for (auto& p : joining_) { p.x -= sx; p.y -= sy; }
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
        if (cfg.plasma_on) { rot(js0x); rot(js0y); rot(js0z); }
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
                        "<p.khat>  rms_pz  eta_rel[FWHM] a_env  W_field[J] W_kin[J]  N_trapped Q_trapped[pC]\n");
    double S[19] = {0};
    for (auto& p : host_particles()) {
        if (p.alive != 1) continue;
        if (p.sp != SP_BUNCH) { S[17] += 1; S[18] += p.q * p.w; continue; }   // trapped plasma electrons
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
    comm.allreduce_sum(S, 19);
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
    const double Qtr = S[18] * cfg.nc * std::pow(cfg.Lu, 3) * e_SI * 1e12;
    char line[600];
    std::snprintf(line, sizeof line, "%6ld %10.5f %8ld %10.4f %10.4f %10.4f %9.4f %9.4f %9.4f %12.6e %12.6e %14.8e %14.8e %14.8e %11.4e %8.4f %8.4f %11.4e %11.4e %8ld %11.4e\n",
                 istep, s * cfg.Lu * 1e3, long(S[14]), m(1) * um, m(2) * um, m(3) * um, rms(1, 4) * um, rms(2, 5) * um,
                 rms(3, 6) * um, m(7), m(8), m(9), m(10), m(11), rms(9, 12), eta_rel, aenv, Wf, Wk, long(S[17]), Qtr);
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

// ================================================================ plasma (streaming particles, as in GEM-PIC-RZ)
// Per step s0 -> s1 with plasma (single rank):
//  1. push the beam list (bunch + trapped electrons) to s1 in the fields of level s0; particles slower than
//     vz_min leave it and stream in this march from their (z, zeta)
//  2. deposit the beam current of level s1 (all half-cells), spectra
//  3. load fresh plasma at the front of the window; march from the front to the rear. Per half-cell n, in
//     `iterations` passes: push the streaming particles from zeta_{n+1} to zeta_n (Boris, dt = h/(1 - v_z)) in
//     the fields of slice n+1 (final, level s1) and slice n (level s0 in the first pass, the last solution
//     after that), deposit their current W = q w dt/Delta, smooth it (plasma.smooth_length), solve slice n.
//     The current of half-cell n-1 needed by R at slice n is predicted as beam(s1) + streaming(s0); once
//     half-cell n is final, R at slice n+1 is corrected for the difference.
//  4. streaming particles that reach z = s1 + (trap_advance - 1) dz are trapped: they join the beam list if
//     v_z >= vz_beam, otherwise they stream again in the next step from their zeta
//  5. sponge, real fields
namespace {
HD inline void boris(double& ux, double& uy, double& uz, const double E[3], const double B[3], double qm, double dt) {
    const double qh = 0.5 * qm * dt;
    const double umx = ux + qh * E[0], umy = uy + qh * E[1], umz = uz + qh * E[2];
    const double gm = ::sqrt(1.0 + umx * umx + umy * umy + umz * umz);
    const double tx = qh * B[0] / gm, ty = qh * B[1] / gm, tz = qh * B[2] / gm;
    const double t2 = tx * tx + ty * ty + tz * tz;
    const double upx = umx + (umy * tz - umz * ty), upy = umy + (umz * tx - umx * tz), upz = umz + (umx * ty - umy * tx);
    const double sf = 2.0 / (1.0 + t2);
    const double vx = umx + sf * (upy * tz - upz * ty), vy = umy + sf * (upz * tx - upx * tz), vz = umz + sf * (upx * ty - upy * tx);
    ux = vx + qh * E[0]; uy = vy + qh * E[1]; uz = vz + qh * E[2];
}

struct StreamCtx {
    PushCtx pc;                 // fields (gather), laser, grids
    const double* zs;           // slice positions
    double* J;                  // real current plane of the half-cell: jx, jy, jz (3 nk)
    const double *ia, *id;      // 1/A_j, 1/Delta_j (Fourier y: 1/dy)
    double Delta, s1, sTrap, vz_beam;
    int n;
};

// TSC deposit of q w (dt / Delta) v / (h dx dy) at (x, y) into the half-cell plane (box coordinates)
HD inline void stream_deposit(const StreamCtx& c, double x, double y, double f, double vx, double vy, double vz) {
    const PushCtx& P = c.pc;
    int ix_[3], jn[3], jh[3]; double wx[3], wn[3], wh[3];
    tsc(x, P.dx, P.nx, ix_, wx);
    const long nk = P.nk;
    if (P.yf) {
        tsc(y, P.dy, P.ny, jn, wn);
        for (int v = 0; v < 3; ++v)
            for (int u = 0; u < 3; ++u) {
                const long k = long(jn[v]) * P.nx + ix_[u];
                const double w = f * wx[u] * wn[v] * c.ia[0];
                atomic_add(&c.J[k], w * vx); atomic_add(&c.J[nk + k], w * vy); atomic_add(&c.J[2 * nk + k], w * vz);
                atomic_add(&c.J[3 * nk + k], w);
            }
        return;
    }
    tsc_y(P.yax, y, jn, wn, jh, wh);
    for (int v = 0; v < 3; ++v) {
        const bool nod = jn[v] >= 1 && jn[v] <= P.ny - 1, hal = jh[v] >= 0 && jh[v] <= P.ny - 1;
        for (int u = 0; u < 3; ++u) {
            if (nod) {
                const long k = long(jn[v]) * P.nx + ix_[u];
                const double w = f * wx[u] * wn[v] * c.ia[jn[v]];
                atomic_add(&c.J[k], w * vx); atomic_add(&c.J[2 * nk + k], w * vz); atomic_add(&c.J[3 * nk + k], w);
            }
            if (hal) {
                const long k = long(jh[v]) * P.nx + ix_[u];
                atomic_add(&c.J[nk + k], f * wx[u] * wh[v] * c.id[jh[v]] * vy);
            }
        }
    }
}

// one streaming particle through half-cell n (from its zeta, normally zeta_{n+1}, to zeta_n)
HD inline void stream_one(const StreamCtx& c, Particle& p) {
    const int n = c.n;
    const double zn = c.zs[n], zn1 = c.zs[n + 1], h = zn1 - zn;
    if (p.alive == 6) {             // waiting: joins the march inside the half-cell that holds its zeta
        if (p.zeta > zn) p.alive = 1;
        else return;
    }
    if (p.alive != 1) return;
    const double hl = p.zeta >= zn1 - 1e-9 * h ? h : (p.zeta - zn > 0 ? p.zeta - zn : 0.0);
    const PushCtx& P = c.pc;
    const double g0 = ::sqrt(1 + p.px * p.px + p.py * p.py + p.pz * p.pz);
    const double omv0 = (1 + p.px * p.px + p.py * p.py) / (g0 * (g0 + p.pz));    // 1 - v_z, no cancellation
    const double dt0 = hl / omv0;
    const double xm = p.x + 0.5 * dt0 * p.px / g0, ym = p.y + 0.5 * dt0 * p.py / g0;
    const double zm = p.z + 0.5 * dt0 * p.pz / g0, zetam = p.zeta - 0.5 * hl;
    double E[3], B[3], Ee[3], Be[3];
    gather(P, ::fmod(::fmod(xm, P.Lx) + P.Lx, P.Lx), P.yf ? ::fmod(::fmod(ym, P.Ly) + P.Ly, P.Ly) : ym, zetam, E, B);
    P.laser.field(xm + P.X0, zm, zm - (zetam - P.zc), Ee, Be);
    for (int k = 0; k < 3; ++k) { E[k] += Ee[k]; B[k] += Be[k]; }
    double ux = p.px, uy = p.py, uz = p.pz;
    boris(ux, uy, uz, E, B, p.qm, dt0);
    double g1 = ::sqrt(1 + ux * ux + uy * uy + uz * uz);
    double dt = 2 * hl / (omv0 + (1 + ux * ux + uy * uy) / (g1 * (g1 + uz)));    // time for |dzeta| = hl
    ux = p.px; uy = p.py; uz = p.pz;
    boris(ux, uy, uz, E, B, p.qm, dt);
    g1 = ::sqrt(1 + ux * ux + uy * uy + uz * uz);
    double vx = 0.5 * (p.px / g0 + ux / g1), vy = 0.5 * (p.py / g0 + uy / g1), vz = 0.5 * (p.pz / g0 + uz / g1);
    double zeta_end = zn;
    bool trapped = false;
    if (p.z + vz * dt >= c.sTrap) {    // advanced trap_advance steps inside this half-cell: trapped
        const double dtp = (c.sTrap - p.z) / (vz > 1e-3 ? vz : 1e-3) > 0 ? (c.sTrap - p.z) / (vz > 1e-3 ? vz : 1e-3) : 0.0;
        ux = p.px; uy = p.py; uz = p.pz;
        boris(ux, uy, uz, E, B, p.qm, dtp);
        g1 = ::sqrt(1 + ux * ux + uy * uy + uz * uz);
        vx = 0.5 * (p.px / g0 + ux / g1); vy = 0.5 * (p.py / g0 + uy / g1); vz = 0.5 * (p.pz / g0 + uz / g1);
        dt = dtp;
        zeta_end = p.zeta - (1 - vz) * dtp;
        trapped = true;
    }
    const double xo = p.x, yo = p.y;
    p.x += vx * dt; p.y += vy * dt;
    p.z = trapped ? c.s1 : p.z + vz * dt;
    p.px = ux; p.py = uy; p.pz = uz;
    p.zeta = zeta_end;
    // streaming current: q w dt / Delta per unit time, into half-cell n
    const double f = p.q * p.w * dt / (c.Delta * h * P.dx);
    stream_deposit(c, 0.5 * (xo + p.x), 0.5 * (yo + p.y), f, vx, vy, vz);
    if (!P.yf && (p.y < P.ylo || p.y > P.yhi)) { p.alive = 0; return; }
    if (trapped) {
        p.alive = 5;
        const double vze = uz / g1;
        if (vze >= c.vz_beam)        // joins the beam list: its beam current of level s1 from now on
            stream_deposit(c, p.x, p.y, p.q * p.w / (h * P.dx * vze), ux / g1, uy / g1, vze);
    }
}
}  // namespace

void Simulation::setup_plasma() {
    if (!cfg.plasma_on) return;
    if (comm.size > 1) throw std::runtime_error("plasma: MPI along zeta is not available with plasma yet (run on one rank)");
    const size_t pl = size_t(nk);
    for (Buf<cpx>* p : {&js0x, &js0y, &js0z}) p->resize((ns + 1) * pl);
    jsr.resize(4 * pl);
    rho_lines.resize(size_t(N) * (cfg.nx + cfg.ny));
    jsn.resize(3 * pl); jsin.resize(3 * pl); bk.resize(4 * pl); sp3.resize(3 * pl);
    jpred.resize(3 * pl); jtmp.resize(3 * pl); jsi0.resize(3 * pl); dlt.resize(3 * pl);
    if (!yf && scp.size() == 0) {
        for (Buf<cpx>* p : {&sRx, &sRy, &sLx, &sLy, &srhs}) p->resize(pl);
        scp.resize(pl);
    }
}

// fresh plasma at the front of the window (a regular lattice of ppc_x x ppc_y particles per cell), and the
// particles that join the march at their zeta
void Simulation::load_plasma(double s0, double s1, const std::vector<Particle>& joining) {
    std::vector<Particle> v;
    const double zmid = 0.5 * (s0 + s1), Delta = s1 - s0;
    const int nx = cfg.nx, ny = cfg.ny, px = cfg.ppc_x, py = cfg.ppc_y;
    const double zf = cfg.zeta[N - 1];
    for (int j = 0; j < ny; ++j) {
        double y0, hy;
        if (yf) { y0 = j * dy; hy = dy; }
        else {
            if (j == 0 || j == ny - 1) continue;          // no plasma in the cells at the walls
            y0 = cfg.ynode[j]; hy = cfg.ynode[j + 1] - cfg.ynode[j];
        }
        for (int i = 0; i < nx; ++i)
            for (int b2 = 0; b2 < py; ++b2)
                for (int a2 = 0; a2 < px; ++a2) {
                    Particle p{};
                    p.x = (i + (a2 + 0.5) / px) * dx;
                    p.y = y0 + (b2 + 0.5) / py * hy;
                    const double n = cfg.plasma_density(p.x + X0, p.y + Y0, zmid);
                    if (n <= 0) continue;
                    p.w = n * dx * hy * Delta / (px * py);
                    p.q = -1; p.qm = -1; p.sp = SP_PLASMA; p.alive = 1;
                    p.zeta = zf; p.z = s0;
                    v.push_back(p);
                }
    }
    fresh_last = long(v.size());
    for (Particle p : joining) { p.alive = 6; v.push_back(p); }
    nstrm = long(v.size());
    if (strm.size() < v.size()) { strm.resize(v.size() + v.size() / 8 + 64); strm_bk.resize(strm.size()); }
    strm.from_host(v.data(), 0, v.size());
}

// spectra of the streaming current of one half-cell (jsr) into jsn: transverse filters as for the beam, the
// k = 0 part kept, and the smoothing (1 - a^2 lap_perp)^-1 of plasma.smooth_length
void Simulation::stream_spectra() {
    const long nkk = nk;
    const int nx = cfg.nx, ny = cfg.ny;
    const double* J = jsr.data();
    cpx* pk = sp3.data();                       // plane 0: jx + i jy, plane 1: jz
    par_for(nk, [=] HD(long i) { pk[i] = C(J[i], J[nkk + i]); pk[nkk + i] = C(J[2 * nkk + i], 0); });
    fft_->run(pk, 2, false);
    cpx* ox = jsn.data(); cpx* oy = ox + nkk; cpx* oz = ox + 2 * nkk;
    const double* fl = filt.data();
    const double* kxs_ = kxs.data();
    const double a2 = cfg.smooth_length * cfg.smooth_length;
    if (yf) {
        const double* fly = filty.data(); const double* kys_ = kys.data();
        par_for(nk, [=] HD(long i) {
            const int ixx = int(i % nx), j = int(i / nx);
            const long m = long((ny - j) % ny) * nx + (nx - ixx) % nx;
            const cpx z = pk[i], zm = conj(pk[m]);
            const double k2 = kxs_[ixx] * kxs_[ixx] + kys_[j] * kys_[j];
            const double f = fl[ixx] * fly[j] / (1 + a2 * k2);
            ox[i] = f * (0.5 * (z + zm)); oy[i] = f * (0.5 * C((z - zm).im, -(z - zm).re)); oz[i] = f * pk[nkk + i];
        });
        device_sync();
        return;
    }
    par_for(nk, [=] HD(long i) {
        const int ixx = int(i % nx);
        const long m = i - ixx + (nx - ixx) % nx;
        const cpx z = pk[i], zm = conj(pk[m]);
        ox[i] = 0.5 * (z + zm); oy[i] = 0.5 * C((z - zm).im, -(z - zm).re); oz[i] = pk[nkk + i];
    });
    const double* ia = iA.data(); const double* id = iD.data();
    const bool kf = cfg.k_filter;
    cpx* rhs = srhs.data(); double* cpv = scp.data();
    par_for(nx, [=] HD(long line) {
        const int ixx = int(line);
        const long o = ixx;
        const double fx = fl[ixx];
        if (kf) {                                         // (1,2,1)/4 on the cell charges, as for the beam
            cpx* const nodal[2] = {ox, oz};
            for (int t = 0; t < 2; ++t) {
                cpx* A = nodal[t];
                cpx qm = C(0), q0 = A[o + nx] / ia[1];
                for (int j = 1; j < ny; ++j) {
                    const cpx qp = j + 1 < ny ? A[o + long(j + 1) * nx] / ia[j + 1] : C(0);
                    A[o + long(j) * nx] = (0.25 * qm + 0.5 * q0 + 0.25 * qp) * ia[j];
                    qm = q0; q0 = qp;
                }
            }
            cpx qm = oy[o] / id[0], q0 = qm;
            for (int j = 0; j < ny; ++j) {
                const cpx qp = j + 1 < ny ? oy[o + long(j + 1) * nx] / id[j + 1] : q0;
                oy[o + long(j) * nx] = (0.25 * qm + 0.5 * q0 + 0.25 * qp) * id[j];
                qm = q0; q0 = qp;
            }
        }
        ox[o] = C(0); oz[o] = C(0);
        for (int j = 0; j < ny; ++j) {
            const long k = o + long(j) * nx;
            ox[k] = fx * ox[k]; oy[k] = fx * oy[k]; oz[k] = fx * oz[k];
        }
        if (a2 <= 0) return;
        const double kx2 = kxs_[ixx] * kxs_[ixx];
        // nodes (j_x, j_z): (1 + a^2 (k_x^2 - b- b+)) f = g, f = 0 at the walls (as E_z in the march)
        cpx* const nodal[2] = {ox, oz};
        for (int t = 0; t < 2; ++t) {
            cpx* A = nodal[t];
            double cp = 0; cpx dp = C(0);
            for (int j = 1; j < ny; ++j) {
                const double lo = j > 1 ? -a2 * ia[j] * id[j - 1] : 0, up = j < ny - 1 ? -a2 * ia[j] * id[j] : 0;
                const double den = 1 + a2 * kx2 + a2 * ia[j] * (id[j - 1] + id[j]) - lo * cp;
                dp = (A[o + long(j) * nx] - lo * dp) / den;
                cp = up / den;
                rhs[o + long(j) * nx] = dp; cpv[o + long(j) * nx] = cp;
            }
            cpx next = C(0);
            for (int j = ny - 1; j >= 1; --j) { next = rhs[o + long(j) * nx] - cpv[o + long(j) * nx] * next; A[o + long(j) * nx] = next; }
        }
        // half-nodes (j_y): (1 + a^2 (k_x^2 - b+ b-)) f = g (as B_z in the march)
        double cp = 0; cpx dp = C(0);
        for (int j = 0; j < ny; ++j) {
            const double al = j > 0 ? ia[j] : 0, au = j < ny - 1 ? ia[j + 1] : 0;
            const double lo = -a2 * id[j] * al, up = -a2 * id[j] * au;
            const double den = 1 + a2 * kx2 + a2 * id[j] * (al + au) - lo * cp;
            dp = (oy[o + long(j) * nx] - lo * dp) / den;
            cp = up / den;
            rhs[o + long(j) * nx] = dp; cpv[o + long(j) * nx] = cp;
        }
        cpx next = C(0);
        for (int j = ny - 1; j >= 0; --j) { next = rhs[o + long(j) * nx] - cpv[o + long(j) * nx] * next; oy[o + long(j) * nx] = next; }
    });
    device_sync();
}

// the march of one slice n (all modes or k_x lines)
void Simulation::march_slice(int n, double D) {
    MarchCtx c{Rx.data(), Ry.data(), Ez.data(), Bz.data(), Lx.data(), Ly.data(), bnd.data(), j0x.data(), j0y.data(),
               jkx.data(), jky.data(), jkz.data(), sRx.data(), sRy.data(), sLx.data(), sLy.data(), srhs.data(), scp.data(),
               kxd.data(), yf ? kyd.data() : nullptr, iD.data(), iA.data(), hcg.data(), wlo.data(), whi.data(), D, a, b, N,
               cfg.nx, cfg.ny, long(nk)};
    if (yf) par_for(nk, [=] HD(long i) { march_point(c, i, n); });
    else par_for(cfg.nx, [=] HD(long i) { march_line(c, int(i), n); });
    device_sync();
}

// E and B of slice n in real space (F plane fl = n - a), from its spectra
void Simulation::real_fields_slice(int n) {
    const long nkk = nk, l = n - a;
    const cpx *rx = Rx.data() + l * nkk, *ry = Ry.data() + l * nkk, *lx = Lx.data() + l * nkk, *ly = Ly.data() + l * nkk;
    const cpx *ez = Ez.data() + l * nkk, *bz = Bz.data() + l * nkk;
    cpx* t = sp3.data();
    par_for(nk, [=] HD(long i) {
        t[i] = C(rx[i].re - ry[i].im, rx[i].im + ry[i].re);
        t[nkk + i] = C(lx[i].re - ly[i].im, lx[i].im + ly[i].re);
        t[2 * nkk + i] = C(ez[i].re - bz[i].im, ez[i].im + bz[i].re);
    });
    fft_->run(t, 3, true);
    float* Fp = F.data() + l * nkk;
    const long fpl = long(ns + 2) * nkk;
    par_for(nk, [=] HD(long i) {
        const double r1 = t[i].re, r2 = t[i].im, l1 = t[nkk + i].re, l2 = t[nkk + i].im;
        Fp[0 * fpl + i] = float(0.5 * (r1 + l1)); Fp[1 * fpl + i] = float(0.5 * (r2 + l2));
        Fp[3 * fpl + i] = float(0.5 * (l2 - r2)); Fp[4 * fpl + i] = float(0.5 * (r1 - l1));
        Fp[2 * fpl + i] = float(t[2 * nkk + i].re); Fp[5 * fpl + i] = float(t[2 * nkk + i].im);
    });
    device_sync();
}

// the march of a step with streaming plasma (single rank: a = 0, b = N; half-cell m at plane m + 1).
// Prediction of the streaming current of half-cell n-1 (needed by R at slice n before the particles get there):
// its value of level s0 plus the change from s0 to s1, extrapolated linearly from the half-cells n and n+1
// (numerics.predictor = extrapolate, default) or without the change (previous, as in GEM-PIC-RZ). The change of
// a laser-driven current is the rotation of the carrier phase in s; without it the error of R at slice n is
// about (omega_p^2 dz / 2 omega)^2 of the laser field.
void Simulation::march_plasma(double s0, double s1) {
    const double Delta = s1 - s0, D = 0.5 * Delta;
    const long nkk = nk;
    load_plasma(s0, s1, joining_);
    joining_.clear();
    const bool extrap = cfg.predictor_extrapolate;
    std::vector<double> cen(N - 1);
    for (int m = 0; m < N - 1; ++m) cen[m] = 0.5 * (cfg.zeta[m] + cfg.zeta[m + 1]);
    auto copy3 = [&](cpx* dst, const cpx* x, const cpx* y, const cpx* z) {
        par_for(nk, [=] HD(long i) { dst[i] = x[i]; dst[nkk + i] = y[i]; dst[2 * nkk + i] = z[i]; });
    };
    // set the streaming part of jk(m) from `old` (the part it holds now) to `nw`; old <- nw
    auto replace = [&](int m, cpx* old, const cpx* nw) {
        cpx* jx = jkx.data() + (m + 1) * nkk; cpx* jy = jky.data() + (m + 1) * nkk; cpx* jz = jkz.data() + (m + 1) * nkk;
        par_for(nk, [=] HD(long i) {
            jx[i] = jx[i] + nw[i] - old[i]; jy[i] = jy[i] + nw[nkk + i] - old[nkk + i]; jz[i] = jz[i] + nw[2 * nkk + i] - old[2 * nkk + i];
            old[i] = nw[i]; old[nkk + i] = nw[nkk + i]; old[2 * nkk + i] = nw[2 * nkk + i];
        });
    };
    cpx* pin = jpred.data();         // streaming part now inside jk(n-1)
    cpx* tmp = jtmp.data();
    cpx* dl = dlt.data();            // change s0 -> s1 of half-cell n+1 (extrapolation)
    // front: half-cell N-2 predicted with its value of level s0
    jpred.zero();
    copy3(tmp, js0x.data() + (N - 1) * nkk, js0y.data() + (N - 1) * nkk, js0z.data() + (N - 1) * nkk);
    replace(N - 2, pin, tmp);
    march_slice(N - 1, D);
    real_fields_slice(N - 1);

    StreamCtx sc;
    {
        PushCtx& c = sc.pc;
        c.F = F.data(); c.a = a; c.ns = ns; c.N = N; c.nx = cfg.nx; c.ny = cfg.ny; c.nk = nk; c.self = 1;
        c.yf = yf; c.dx = dx; c.dy = dy; c.Lx = cfg.Lx; c.Ly = cfg.Ly; c.zax = zax; c.yax = yax; c.laser = laser; c.X0 = X0;
        c.zc = cfg.zeta_center; c.dz_push = cfg.dz_push; c.push_phase = cfg.push_phase; c.zhand = -1e300;
        c.zmax = cfg.zeta[N - 1]; c.s1 = s1; c.vz_min = 0; c.freeze = 0;
        c.ylo = yf ? -1e300 : cfg.ynode[1]; c.yhi = yf ? 1e300 : cfg.ynode[cfg.ny - 1];
    }
    sc.zs = zs.data(); sc.J = jsr.data(); sc.ia = iA.data(); sc.id = iD.data();
    sc.Delta = Delta; sc.s1 = s1; sc.sTrap = s1 + (cfg.trap_advance - 1) * Delta; sc.vz_beam = cfg.vz_beam;
    const int passes = cfg.iterations;
    bool have_dl = false;
    for (int n = N - 2; n >= 0; --n) {
        const long hp = (n + 1) * nkk;           // plane of half-cell n
        cpx* sin_ = jsin.data();                  // streaming part inside jk(n): the prediction so far
        par_for(3 * nkk, [=] HD(long i) { sin_[i] = pin[i]; });
        cpx* si0 = jsi0.data();                   // ... as used by the solve of slice n+1 (for the correction)
        par_for(3 * nkk, [=] HD(long i) { si0[i] = pin[i]; });
        if (n >= 1) {                             // start of the prediction of half-cell n-1: its level s0
            jpred.zero();
            copy3(tmp, js0x.data() + n * nkk, js0y.data() + n * nkk, js0z.data() + n * nkk);
            replace(n - 1, pin, tmp);
        }
        if (passes > 1) {
            int k = 0;
            for (Buf<cpx>* p : {&Rx, &Ry, &Ez, &Bz}) {
                cpx* dst = bk.data() + (k++) * nkk; const cpx* src = p->data() + n * nkk;
                par_for(nk, [=] HD(long i) { dst[i] = src[i]; });
            }
            Particle* d = strm_bk.data(); const Particle* sp_ = strm.data();
            par_for(nstrm, [=] HD(long i) { d[i] = sp_[i]; });
        }
        // weights of the linear extrapolation of the change to half-cell n-1
        const double wex = (n >= 1 && n + 1 <= N - 2 && have_dl) ? (cen[n] - cen[n - 1]) / (cen[n + 1] - cen[n]) : 0.0;
        for (int pass = 0; pass < passes; ++pass) {
            if (pass > 0) {
                int k = 0;
                for (Buf<cpx>* p : {&Rx, &Ry, &Ez, &Bz}) {
                    cpx* dst = p->data() + n * nkk; const cpx* src = bk.data() + (k++) * nkk;
                    par_for(nk, [=] HD(long i) { dst[i] = src[i]; });
                }
                Particle* d = strm.data(); const Particle* sp_ = strm_bk.data();
                par_for(nstrm, [=] HD(long i) { d[i] = sp_[i]; });
            }
            jsr.zero();
            sc.n = n;
            Particle* P = strm.data();
            const StreamCtx scc = sc;
            par_for(nstrm, [=] HD(long i) { stream_one(scc, P[i]); });
            device_sync();
            stream_spectra();
            replace(n, sin_, jsn.data());
            if (n >= 1 && extrap) {               // prediction of half-cell n-1 = s0 + change, extrapolated
                const cpx *u = jsn.data(), *sx = js0x.data() + hp, *sy = js0y.data() + hp, *sz = js0z.data() + hp;
                const cpx *px_ = js0x.data() + n * nkk, *py_ = js0y.data() + n * nkk, *pz_ = js0z.data() + n * nkk;
                const double w = wex;
                par_for(nk, [=] HD(long i) {
                    const cpx d0 = u[i] - sx[i], d1 = u[nkk + i] - sy[i], d2 = u[2 * nkk + i] - sz[i];
                    tmp[i] = px_[i] + d0 + w * (d0 - dl[i]);
                    tmp[nkk + i] = py_[i] + d1 + w * (d1 - dl[nkk + i]);
                    tmp[2 * nkk + i] = pz_[i] + d2 + w * (d2 - dl[2 * nkk + i]);
                });
                replace(n - 1, pin, tmp);
            }
            march_slice(n, D);
            real_fields_slice(n);
        }
        {   // electron density of half-cell n on the lines y = 0 and x = 0 (frames)
            const int nx = cfg.nx, ny = cfg.ny;
            const int i0 = ((int(std::lround(-X0 / dx)) % nx) + nx) % nx;
            const int j0 = yf ? ((int(std::lround(-Y0 / dy)) % ny) + ny) % ny : ny / 2;
            const double* J = jsr.data() + 3 * nkk; double* rl = rho_lines.data() + long(n) * (nx + ny);
            par_for(nx + ny, [=] HD(long k) { rl[k] = k < nx ? J[long(j0) * nx + k] : J[(k - nx) * nx + i0]; });
        }
        {   // half-cell n is final: correct R at slice n+1 for the prediction it used, keep the change for the
            // extrapolation and the current as the prediction of the next step
            const double q = D * cfg_wlo_[n + 1];
            cpx* rx = Rx.data() + (n + 1) * nkk; cpx* ry = Ry.data() + (n + 1) * nkk;
            cpx *sx = js0x.data() + hp, *sy = js0y.data() + hp, *sz = js0z.data() + hp;
            const cpx* u = jsn.data();
            par_for(nk, [=] HD(long i) {
                rx[i] = rx[i] - q * (u[i] - si0[i]); ry[i] = ry[i] - q * (u[nkk + i] - si0[nkk + i]);
                dl[i] = u[i] - sx[i]; dl[nkk + i] = u[nkk + i] - sy[i]; dl[2 * nkk + i] = u[2 * nkk + i] - sz[i];
                sx[i] = u[i]; sy[i] = u[nkk + i]; sz[i] = u[2 * nkk + i];
            });
            device_sync();
            have_dl = true;
        }
    }
    // trapped particles: fast ones join the beam list, slow ones stream again in the next step
    std::vector<Particle> sv(static_cast<size_t>(nstrm));
    strm.to_host(sv.data(), 0, sv.size());
    std::vector<Particle> add;
    for (auto& p : sv) {
        if (p.alive != 5) continue;
        const double g = std::sqrt(1 + p.px * p.px + p.py * p.py + p.pz * p.pz);
        p.z = s1;
        if (p.pz / g >= cfg.vz_beam) { p.alive = 1; p.sp = SP_TRAPPED; add.push_back(p); ++trapped_total; }
        else { p.alive = 6; joining_.push_back(p); }
    }
    if (!add.empty()) {
        std::vector<Particle> all = host_particles(), keep;
        keep.reserve(all.size() + add.size());
        for (auto& p : all) if (p.alive == 1) keep.push_back(p);
        keep.insert(keep.end(), add.begin(), add.end());
        set_particles(keep);
    }
    // the sponge damps R - R_b with the bound field of the beam list only (the plasma is not a v = c source)
    cpx* jz = jkz.data(); const cpx* sz = js0z.data();
    par_for(long(ns + 1) * nk, [=] HD(long i) { jz[i] = jz[i] - sz[i]; });
    device_sync();
}

// the beam particles that became slower than vz_min stream in the current march
void Simulation::collect_restreaming() {
    std::vector<Particle> all = host_particles(), keep;
    bool any = false;
    for (auto& p : all) if (p.alive == 4) { any = true; break; }
    if (!any) return;
    for (auto& p : all) {
        if (p.alive == 4) { Particle q = p; q.alive = 6; joining_.push_back(q); ++restreamed_total; }
        else if (p.alive == 1) keep.push_back(p);
    }
    set_particles(keep);
}

// a Gaussian laser pulse on the grid: R = 2 E_perp of a paraxial beam (complex beam parameter), focus at z_focus
void Simulation::add_grid_lasers() {
    if (cfg.glasers.empty()) return;
    const int nx = cfg.nx, ny = cfg.ny;
    const size_t pl = size_t(nk);
    std::vector<cpx> pk(pl);
    for (const auto& Lz : cfg.glasers) {
        const double amp = Lz.a0 * Lz.kL;                       // field in m c omega0 / e
        const double zR = 0.5 * Lz.kL * Lz.waist * Lz.waist;
        const std::complex<double> qb(1.0, (0.0 - Lz.z_focus) / zR);
        const std::complex<double> rot(std::cos(Lz.pol_angle), std::sin(Lz.pol_angle));
        for (int n = a; n < b; ++n) {
            const double d = cfg.zeta[n] - Lz.zeta_c;
            const double env = std::exp(-2 * std::log(2.0) * d * d / (Lz.tau * Lz.tau));
            if (env < 1e-12) continue;
            const std::complex<double> car = std::exp(std::complex<double>(0, Lz.kL * d + Lz.cep));
            auto field = [&](double xp, double yp, double& ex, double& ey) {
                const std::complex<double> E = amp * env * std::exp(-(xp * xp + yp * yp) / (Lz.waist * Lz.waist) / qb) / qb * car;
                if (Lz.circular) { ex = E.real(); ey = Lz.helicity * E.imag(); }
                else { ex = E.real() * rot.real(); ey = E.real() * rot.imag(); }
            };
            for (int j = 0; j < ny; ++j)
                for (int i = 0; i < nx; ++i) {
                    const double xp = i * dx + X0;
                    double ex, ey, ex2, ey2;
                    if (yf) {
                        field(xp, j * dy + Y0, ex, ey);
                        pk[size_t(j) * nx + i] = C(2 * ex, 2 * ey);                         // R_x + i R_y
                    } else {                                                           // R_x on nodes, R_y on half-nodes
                        field(xp, cfg.ynode[j], ex, ey);
                        field(xp, 0.5 * (cfg.ynode[j] + cfg.ynode[j + 1]), ex2, ey2);
                        pk[size_t(j) * nx + i] = C(j == 0 ? 0.0 : 2 * ex, 2 * ey2);
                    }
                }
            sp3.from_host(pk.data(), 0, pl);
            cpx* t = sp3.data();
            fft_->run(t, 1, false);
            const long l = n - a, nkk = nk;
            cpx* rx = Rx.data() + l * nkk; cpx* ry = Ry.data() + l * nkk;
            const bool yfo = yf;
            par_for(nk, [=] HD(long idx) {
                const int ixx = int(idx % nx), j = int(idx / nx);
                const long m = yfo ? long((ny - j) % ny) * nx + (nx - ixx) % nx : idx - ixx + (nx - ixx) % nx;
                const cpx z = t[idx], zm = conj(t[m]);
                rx[idx] = rx[idx] + 0.5 * (z + zm); ry[idx] = ry[idx] + 0.5 * C((z - zm).im, -(z - zm).re);
            });
            device_sync();
        }
    }
}

// Frame for the GUI: E and B (and with plasma n_e/n0 of the streaming particles) in the planes y = 0 and x = 0
// through the axis. One header line (JSON), then float32 arrays: per component the plane y = 0 [nzeta][nx],
// then the plane x = 0 [nzeta][ny]. Free-hand y: components on half-nodes are averaged to y = 0 in the plane
// y = 0 and given at the half-nodes in the plane x = 0 (y_half in the header).
void Simulation::write_frame(const std::string& file) {
    const int nx = cfg.nx, ny = cfg.ny, ncomp = 6;
    const int i0 = ((int(std::lround(-X0 / dx)) % nx) + nx) % nx;
    const int j0 = yf ? ((int(std::lround(-Y0 / dy)) % ny) + ny) % ny : ny / 2;
    const std::vector<float> Fh = cfg.self_fields ? F.host() : std::vector<float>(F.size(), 0.0f);
    std::vector<float> mine(size_t(ns) * ncomp * (nx + ny));
    const long fpl = long(ns + 2) * nk;
    for (int l = 0; l < ns; ++l)
        for (int c = 0; c < ncomp; ++c) {
            const float* P = Fh.data() + c * fpl + long(l) * nk;
            float* o = &mine[(size_t(l) * ncomp + c) * (nx + ny)];
            const bool half = !yf && is_half(c);
            for (int i = 0; i < nx; ++i) o[i] = half ? 0.5f * (P[long(j0 - 1) * nx + i] + P[long(j0) * nx + i]) : P[long(j0) * nx + i];
            for (int j = 0; j < ny; ++j) o[nx + j] = P[long(j) * nx + i0];
        }
    std::vector<char> all = comm.gather_bytes(mine.data(), mine.size() * sizeof(float));
    if (comm.rank != 0) return;
    const float* A = reinterpret_cast<const float*>(all.data());
    const double um = cfg.Lu * 1e6;
    std::string h = "{\"step\": " + std::to_string(istep) + ", \"z_mm\": " + std::to_string(s * cfg.Lu * 1e3) +
                    ", \"nzeta\": " + std::to_string(N) + ", \"nx\": " + std::to_string(nx) + ", \"ny\": " + std::to_string(ny);
    auto arr = [&](const std::vector<double>& v) {
        std::string t = "[";
        char b[32];
        for (size_t k = 0; k < v.size(); ++k) { std::snprintf(b, sizeof b, "%s%.6g", k ? ", " : "", v[k]); t += b; }
        return t + "]";
    };
    std::vector<double> zeta(N), xs(nx), yn(ny), yh(ny);
    for (int n = 0; n < N; ++n) zeta[n] = cfg.zeta[n] * um;
    for (int i = 0; i < nx; ++i) xs[i] = (((i - i0) % nx + nx + nx / 2) % nx - nx / 2) * dx * um;   // relative to the axis
    for (int j = 0; j < ny; ++j) {
        yn[j] = yf ? (((j - j0) % ny + ny + ny / 2) % ny - ny / 2) * dy * um : cfg.ynode[j] * um;
        yh[j] = yf ? yn[j] : 0.5 * (cfg.ynode[j] + cfg.ynode[j + 1]) * um;
    }
    const bool dens = cfg.plasma_on;
    h += ", \"zeta_um\": " + arr(zeta) + ", \"x_um\": " + arr(xs) + ", \"y_um\": " + arr(yn) + ", \"y_half_um\": " + arr(yh) +
         ", \"comps\": [\"ex\", \"ey\", \"ez\", \"bx\", \"by\", \"bz\"" + (dens ? ", \"ne\"" : "") + "]" +
         ", \"half\": [false, " + (yf ? "false" : "true") + ", false, " + (yf ? "false" : "true") + ", false, " + (yf ? "false" : "true") +
         (dens ? ", false" : "") + "], \"n0\": " + std::to_string(cfg.n0) + ", \"lambda0_um\": " + std::to_string(cfg.lambda0 * 1e6) + "}\n";
    const std::string tmp = file + ".tmp";
    FILE* f = std::fopen(tmp.c_str(), "wb");
    if (!f) throw std::runtime_error("cannot write " + tmp);
    std::fwrite(h.data(), 1, h.size(), f);
    std::vector<float> buf(size_t(N) * std::max(nx, ny));
    for (int c = 0; c < ncomp; ++c) {
        for (int n = 0; n < N; ++n) for (int i = 0; i < nx; ++i) buf[size_t(n) * nx + i] = A[(size_t(n) * ncomp + c) * (nx + ny) + i];
        std::fwrite(buf.data(), sizeof(float), size_t(N) * nx, f);
        for (int n = 0; n < N; ++n) for (int j = 0; j < ny; ++j) buf[size_t(n) * ny + j] = A[(size_t(n) * ncomp + c) * (nx + ny) + nx + j];
        std::fwrite(buf.data(), sizeof(float), size_t(N) * ny, f);
    }
    if (dens) {                                          // n_e / n0 at the slices: mean of the adjacent half-cells
        const std::vector<double> rl = rho_lines.host();
        const int L = nx + ny;
        auto at = [&](int n, int k) {
            const double lo = n > 0 ? rl[size_t(n - 1) * L + k] : 0, hi = n < N - 1 ? rl[size_t(n) * L + k] : 0;
            return float(-0.5 * (lo + hi) / cfg.n0);
        };
        for (int n = 0; n < N; ++n) for (int i = 0; i < nx; ++i) buf[size_t(n) * nx + i] = at(n, i);
        std::fwrite(buf.data(), sizeof(float), size_t(N) * nx, f);
        for (int n = 0; n < N; ++n) for (int j = 0; j < ny; ++j) buf[size_t(n) * ny + j] = at(n, nx + j);
        std::fwrite(buf.data(), sizeof(float), size_t(N) * ny, f);
    }
    std::fclose(f);
    std::rename(tmp.c_str(), file.c_str());
}

// axis.txt (output.axis): one line per output step: z [mm], then E_z and E_x - B_y (the transverse force on an
// electron moving at c along z, per -e) on the axis x = 0, y = 0 at every slice, in code units m c omega0 / e.
// The first line (#) holds zeta [um] of the slices.
void Simulation::write_axis(const std::string& file, bool first) {
    const int nx = cfg.nx, ny = cfg.ny;
    const int i0 = ((int(std::lround(-X0 / dx)) % nx) + nx) % nx;
    const int j0 = yf ? ((int(std::lround(-Y0 / dy)) % ny) + ny) % ny : ny / 2;
    const std::vector<float> Fh = cfg.self_fields ? F.host() : std::vector<float>(F.size(), 0.0f);
    const long fpl = long(ns + 2) * nk;
    std::vector<double> mine(size_t(ns) * 2);
    for (int l = 0; l < ns; ++l) {
        const long k = long(l) * nk + long(j0) * nx + i0;
        // free-hand y: B_y (comp 4) is on the nodes like E_z and E_x
        mine[2 * l] = Fh[2 * fpl + k];
        mine[2 * l + 1] = Fh[0 * fpl + k] - Fh[4 * fpl + k];
    }
    std::vector<char> all = comm.gather_bytes(mine.data(), mine.size() * sizeof(double));
    if (comm.rank != 0) return;
    const double* A = reinterpret_cast<const double*>(all.data());
    FILE* f = std::fopen(file.c_str(), first ? "w" : "a");
    if (!f) throw std::runtime_error("cannot write " + file);
    if (first) {
        std::fprintf(f, "# GEM-PIC-3D axis lineouts: z_mm, then Ez(zeta_n) for n = 0..%d, then (Ex - By)(zeta_n); code units\n# zeta_um", N - 1);
        for (int n = 0; n < N; ++n) std::fprintf(f, " %.6g", cfg.zeta[n] * cfg.Lu * 1e6);
        std::fprintf(f, "\n");
    }
    std::fprintf(f, "%.9g", s * cfg.Lu * 1e3);
    for (int n = 0; n < N; ++n) std::fprintf(f, " %.7e", A[2 * n]);
    for (int n = 0; n < N; ++n) std::fprintf(f, " %.7e", A[2 * n + 1]);
    std::fprintf(f, "\n");
    std::fclose(f);
}
