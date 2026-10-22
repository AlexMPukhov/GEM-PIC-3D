// Configuration: units, grids, the deck reader Config::from_deck and the run summary.
#include "sim_internal.hpp"

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
    // output groups (as in QUARZ): the main output (block output) and the diagnostics diag.<name>
    auto words = [](std::string t) {
        for (char& ch : t) if (ch == ',') ch = ' ';
        std::vector<std::string> w;
        std::istringstream is(t);
        for (std::string x; is >> x;) w.push_back(x);
        return w;
    };
    auto group = [&](const std::string& b, const std::string& name, const OutGroup* main) {
        OutGroup g;
        g.name = name; g.dir = name.empty() ? c.dir : c.dir + "/" + name;
        g.every = int(d.integer(b, "every", main ? 0 : 10));
        g.beam_every = int(d.integer(b, "beam_every", main ? g.every : 0));
        if (g.every < 0 || g.beam_every < 0) throw std::runtime_error(b + ": every and beam_every must be >= 0");
        static const char* fn[7] = {"ex", "ey", "ez", "bx", "by", "bz", "ne"};
        const std::string fs = d.str(b, "fields", "all");
        if (fs == "all") g.fields = 127;
        else if (fs == "none") g.fields = 0;
        else {
            g.fields = 0;
            for (const auto& w : words(fs)) {
                int k = 0;
                while (k < 7 && w != fn[k]) ++k;
                if (k == 7) throw std::runtime_error(b + ".fields: unknown component '" + w + "' (ex ey ez bx by bz ne, all, none)");
                g.fields |= 1 << k;
            }
        }
        const std::string bs = d.str(b, "beams", "all");
        if (bs == "all") g.beams = 3;
        else if (bs == "none") g.beams = 0;
        else {
            g.beams = 0;
            for (const auto& w : words(bs)) {
                if (w == "bunch") g.beams |= 1;
                else if (w == "trapped") g.beams |= 2;
                else throw std::runtime_error(b + ".beams: unknown beam '" + w + "' (bunch trapped, all, none)");
            }
        }
        g.axis = d.flag(b, "axis", false);
        const std::string fmt = d.str(b, "format", main ? (main->openpmd ? (main->native ? "both" : "openpmd") : "native") : "native");
        if (fmt == "native") { g.native = true; g.openpmd = false; }
        else if (fmt == "openpmd") { g.native = false; g.openpmd = true; }
        else if (fmt == "both") { g.native = g.openpmd = true; }
        else throw std::runtime_error(b + ".format must be native, openpmd or both");
        g.xi_stride = int(d.integer(b, "xi_stride", 1));
        g.particle_stride = int(d.integer(b, "particle_stride", 1));
        if (g.xi_stride < 1 || g.particle_stride < 1) throw std::runtime_error(b + ": xi_stride and particle_stride must be >= 1");
        g.openpmd_file = d.str(b, "openpmd_file", main ? main->openpmd_file : "openpmd/data_%06T");
        {
            const size_t pc = g.openpmd_file.find('%'), te = pc == std::string::npos ? pc : g.openpmd_file.find('T', pc);
            bool ok = te != std::string::npos;
            for (size_t k = pc + 1; ok && k < te; ++k) ok = std::isdigit((unsigned char)g.openpmd_file[k]);
            if (!ok) throw std::runtime_error(b + ".openpmd_file needs %T or %0<n>T for the step");
        }
        const std::string grid = d.str(b, "openpmd_grid", main ? (main->openpmd_uniform ? (main->openpmd_native ? "both" : "uniform") : "native") : "uniform");
        if (grid == "uniform") { g.openpmd_uniform = true; g.openpmd_native = false; }
        else if (grid == "native") { g.openpmd_uniform = false; g.openpmd_native = true; }
        else if (grid == "both") { g.openpmd_uniform = g.openpmd_native = true; }
        else throw std::runtime_error(b + ".openpmd_grid must be uniform, native or both");
        g.openpmd_dy = Ld(b, "openpmd_dy", main ? main->openpmd_dy : 0);
        g.openpmd_dzeta = Ld(b, "openpmd_dzeta", main ? main->openpmd_dzeta : 0);
        g.hdf5_single = d.flag(b, "hdf5_single", main ? main->hdf5_single : true);
        g.hdf5_compression = int(d.integer(b, "hdf5_compression", main ? main->hdf5_compression : 0));
        if (g.hdf5_compression < 0 || g.hdf5_compression > 9) throw std::runtime_error(b + ".hdf5_compression must be 0..9");
        if (g.openpmd) c.openpmd = true;
        return g;
    };
    c.groups.push_back(group("output", "", nullptr));
    if (c.groups[0].every != c.every) c.groups[0].every = c.every;   // output.every: diag.txt and the main group
    for (const auto& n : words(d.str("diag", "names", ""))) {
        const std::string b = "diag." + n;
        if (!d.has_block(b)) throw std::runtime_error("diag.names: no block '" + b + "'");
        for (const auto& g : c.groups) if (g.name == n) throw std::runtime_error("diag.names: '" + n + "' given twice");
        c.groups.push_back(group(b, n, &c.groups[0]));
    }
    c.freeze_bunch = d.flag("numerics", "freeze_bunch", false);
    c.deck_text = d.text();                            // the deck after overrides, stored in openPMD files
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
    std::fprintf(f, "  output: dir '%s', every %d steps (%ld steps to z_end)%s\n", dir.c_str(), every,   // read by the GUI
                 long(std::ceil(z_end / dz - 1e-9)), frames ? ", frames" : "");
    for (const auto& g : groups) {
        std::string fl;
        static const char* fn[7] = {"ex", "ey", "ez", "bx", "by", "bz", "ne"};
        for (int k = 0; k < 7; ++k) if (g.fields >> k & 1) fl += std::string(fl.empty() ? "" : " ") + fn[k];
        std::fprintf(f, "    %-10s fields every %d (%s), beams every %d (%s%s%s), %s%s, xi_stride %d, particle_stride %d -> %s\n",
                     g.name.empty() ? "output" : ("diag." + g.name).c_str(), g.every, fl.empty() ? "none" : fl.c_str(), g.beam_every,
                     g.beams & 1 ? "bunch" : "", g.beams == 3 ? " " : "", g.beams & 2 ? "trapped" : (g.beams ? "" : "none"),
                     g.native && g.openpmd ? "native + openPMD" : g.openpmd ? "openPMD" : "native", g.axis ? ", axis" : "",
                     g.xi_stride, g.particle_stride, g.dir.c_str());
    }
    if (freeze_bunch) std::fprintf(f, "  bunch frozen (numerics.freeze_bunch): not pushed, a rigid source\n");
    std::fprintf(f, "  self fields %s, zeta filter %s, transverse filter %s\n", self_fields ? "on" : "off", zeta_filter ? "on" : "off",
                 k_filter ? "on" : "off");
}
