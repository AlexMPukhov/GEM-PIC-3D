// Bunches from files (bunch.file, as in GEM-PIC-RZ): the particles of an openPMD file (this code, GEM-PIC-RZ,
// FBPIC, HiPACE++, WarpX, ...) with a selection, placed at zeta_center, optionally rescaled and resampled.
#include "sim_internal.hpp"
#include "beam_import.hpp"

void Config::parse_bunch_file(const Deck& d, const std::string& b, Bunch& u) {
    auto Ld = [&](const std::string& k, double def) { return d.has(b, k) ? d.num(b, k, Kind::Length) / Lu : def; };
    for (const char* k : {"energy", "gamma", "sigma_x", "sigma_y", "sigma_z", "emittance_x", "emittance_y", "energy_spread",
                          "distribution_z", "distribution_r", "species"})
        if (d.has(b, k)) throw std::runtime_error(b + "." + k + ": not used with a bunch from a file (everything comes from the file)");
    std::string path = d.str(b, "file");
    if (!std::filesystem::exists(path)) {              // relative to the directory of the deck
        const std::filesystem::path alt = std::filesystem::path(d.file).parent_path() / path;
        if (std::filesystem::exists(alt)) path = alt.string();
    }
    gem::ImportedBeam imp;
    try {
        imp = gem::importBeam(path, d.str(b, "file_species", ""), d.integer(b, "file_iteration", -1));
    } catch (const std::exception& e) { throw std::runtime_error(b + ".file: " + e.what()); }
    // the longitudinal coordinate relative to the charge centroid of the species (the time of the file does not matter)
    double sw = 0, sz = 0;
    for (const auto& p : imp.p) { sw += p.w; sz += p.w * p.z; }
    const double zc = sz / sw;
    Expr sel;
    if (d.has(b, "select")) sel = d.expr(b, "select", {"x", "y", "z", "ux", "uy", "uz", "gamma"});
    std::vector<Bunch::FileParticle> v;
    v.reserve(imp.p.size());
    for (const auto& p : imp.p) {
        const Bunch::FileParticle f{p.x / Lu, p.y / Lu, (p.z - zc) / Lu, p.ux, p.uy, p.uz, p.w};
        if (!sel.empty() && sel({f.x, f.y, f.zeta, f.ux, f.uy, f.uz, std::sqrt(1 + f.ux * f.ux + f.uy * f.uy + f.uz * f.uz)}) == 0.0) continue;
        v.push_back(f);
    }
    if (v.empty()) throw std::runtime_error(b + ".select: no particle of the file passes the selection");
    u.x0 = Ld("x0", 0); u.y0 = Ld("y0", 0);
    const double zcen = d.num(b, "zeta_center", Kind::Length) / Lu;
    double W = 0, mz = 0;
    for (const auto& f : v) { W += f.w; mz += f.w * f.zeta; }
    mz /= W;
    double scale = 1;
    if (d.has(b, "charge")) {
        const double q = std::fabs(d.num(b, "charge", Kind::Charge));
        if (!(q > 0)) throw std::runtime_error(b + ".charge must not be zero");
        scale = q / (W * e_SI);
    }
    for (auto& f : v) { f.zeta += zcen - mz; f.x += u.x0; f.y += u.y0; f.w *= scale; }
    W *= scale;
    // moments: the summary, and gamma for the co-moving start field (energy spread accumulated centred)
    double g1 = 0, x1 = 0, x2 = 0, y1 = 0, y2 = 0, z2 = 0, xu = 0, ux1 = 0, ux2 = 0, yu = 0, uy1 = 0, uy2 = 0;
    for (const auto& f : v) {
        const double gm = std::sqrt(1 + f.ux * f.ux + f.uy * f.uy + f.uz * f.uz), w = f.w / W;
        g1 += w * gm; x1 += w * f.x; x2 += w * f.x * f.x; y1 += w * f.y; y2 += w * f.y * f.y;
        z2 += w * (f.zeta - zcen) * (f.zeta - zcen); ux1 += w * f.ux; ux2 += w * f.ux * f.ux; xu += w * f.x * f.ux;
        uy1 += w * f.uy; uy2 += w * f.uy * f.uy; yu += w * f.y * f.uy;
    }
    double g2 = 0;
    for (const auto& f : v) {
        const double gm = std::sqrt(1 + f.ux * f.ux + f.uy * f.uy + f.uz * f.uz);
        g2 += f.w / W * (gm - g1) * (gm - g1);
    }
    if (!(g1 > 1)) throw std::runtime_error(b + ".file: the mean gamma of the beam is not > 1");
    u.from_file = true;
    u.q = imp.charge_sign;
    u.charge = W * e_SI;
    u.gamma = g1;
    u.file_espread = std::sqrt(g2) / g1;
    u.sigx = std::sqrt(std::max(0.0, x2 - x1 * x1)); u.sigy = std::sqrt(std::max(0.0, y2 - y1 * y1)); u.sigz = std::sqrt(z2);
    const double cx = x2 - x1 * x1, cu = ux2 - ux1 * ux1, cxu = xu - x1 * ux1;
    const double cy = y2 - y1 * y1, cv = uy2 - uy1 * uy1, cyv = yu - y1 * uy1;
    u.file_emitx = std::sqrt(std::max(0.0, cx * cu - cxu * cxu));
    u.file_emity = std::sqrt(std::max(0.0, cy * cv - cyv * cyv));
    u.zeta_center = zcen;
    u.nmacro = d.integer(b, "macroparticles", long(v.size()));
    if (u.nmacro < 1) throw std::runtime_error(b + ".macroparticles must be >= 1");
    u.resample_smoothing = d.num(b, "resample_smoothing", Kind::None, 0.0);
    if (u.resample_smoothing < 0) throw std::runtime_error(b + ".resample_smoothing must be >= 0");
    std::string info = path + ", " + imp.format + " species " + imp.species;
    if (imp.iteration >= 0) info += " iteration " + std::to_string(imp.iteration);
    info += ": " + std::to_string(v.size()) + " of " + std::to_string(imp.p.size()) + " particles";
    if (!sel.empty()) info += " (select: " + sel.text() + ")";
    if (u.nmacro != long(v.size())) info += ", resampled to " + std::to_string(u.nmacro) + " macroparticles";
    u.file_info = info;
    u.file_particles.swap(v);
}

// The particles of the file are on the plane z = 0 with zeta as placed (z of the file relative to the centroid is
// taken as zeta, as in GEM-PIC-RZ). All ranks hold the list; each keeps the particles of its half-cells.
void Simulation::load_bunch_file(const Config::Bunch& B, int kb, std::vector<Particle>& mine, int hlo, int hhi, double ylo,
                                 double yhi, double yoff) {
    Axis zh = zax; zh.X = cfg.zeta.data();
    const double cw = 1 / (cfg.nc * std::pow(cfg.Lu, 3));           // real particles -> code weight
    auto add = [&](const Config::Bunch::FileParticle& f, double w) {
        Particle p{};
        p.x = f.x + 0.5 * cfg.Lx; p.y = f.y + yoff; p.zeta = f.zeta;
        p.px = f.ux; p.py = f.uy; p.pz = f.uz; p.w = w * cw; p.alive = 1;
        p.q = B.q; p.qm = B.q; p.sp = -kb; p.z = 0;
        int hc; double fr;
        zh.locate(p.zeta, hc, fr);
        if (p.zeta < 0 || hc > N - 2 || p.y < ylo || p.y > yhi) { if (comm.rank == 0) ++lost_total; return; }
        if (hc >= hlo && hc <= hhi) mine.push_back(p);
    };
    const auto& F = B.file_particles;
    if (B.nmacro == long(F.size())) {                                // the particles of the file as they are
        for (const auto& f : F) add(f, f.w);
        return;
    }
    // systematic resampling with probabilities ~ charge, equal weights; optional Gaussian jitter (as GEM-PIC-RZ)
    std::mt19937_64 rng(B.seed);
    std::normal_distribution<double> G(0, 1);
    std::uniform_real_distribution<double> U(0, 1);
    double W = 0;
    for (const auto& f : F) W += f.w;
    double sd[6] = {0, 0, 0, 0, 0, 0};
    if (B.resample_smoothing > 0) {
        double m1[6] = {0, 0, 0, 0, 0, 0}, m2[6] = {0, 0, 0, 0, 0, 0};
        for (const auto& f : F) {
            const double v[6] = {f.x, f.y, f.zeta, f.ux, f.uy, f.uz};
            for (int k = 0; k < 6; ++k) { m1[k] += f.w / W * v[k]; m2[k] += f.w / W * v[k] * v[k]; }
        }
        for (int k = 0; k < 6; ++k) sd[k] = B.resample_smoothing * std::sqrt(std::max(0.0, m2[k] - m1[k] * m1[k]));
    }
    const double u0 = U(rng);
    double cum = 0;
    size_t j = 0;
    for (long k = 0; k < B.nmacro; ++k) {
        const double target = (k + u0) / B.nmacro * W;
        while (j + 1 < F.size() && cum + F[j].w < target) cum += F[j++].w;
        Config::Bunch::FileParticle f = F[j];
        if (B.resample_smoothing > 0) {
            f.x += sd[0] * G(rng); f.y += sd[1] * G(rng); f.zeta += sd[2] * G(rng);
            f.ux += sd[3] * G(rng); f.uy += sd[4] * G(rng); f.uz += sd[5] * G(rng);
        }
        add(f, W / B.nmacro);
    }
}
