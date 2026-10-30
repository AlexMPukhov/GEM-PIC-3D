// Bunches: the bunch blocks of the deck (bunch, bunch2, ...: electrons or positrons, Gaussian or flat-top) and
// their loading at the start.
#include "sim_internal.hpp"

// bunch blocks (keys as in GEM-PIC-RZ where they exist); the first one also sets the single-bunch fields
void Config::parse_bunches(const Deck& d) {
    auto L = [&](const std::string& b, const std::string& k) { return d.num(b, k, Kind::Length) / Lu; };
    auto Ld = [&](const std::string& b, const std::string& k, double def) { return d.has(b, k) ? L(b, k) : def; };
    bunches.clear();
    for (int k = 1; d.has_block(k == 1 ? "bunch" : "bunch" + std::to_string(k)); ++k) {
        const std::string b = k == 1 ? "bunch" : "bunch" + std::to_string(k);
        Bunch u;
        u.name = d.str(b, "name", b);
        const std::string sp = d.str(b, "species", "electron");
        if (sp == "positron") u.q = 1;
        else if (sp != "electron") throw std::runtime_error(b + ".species must be electron or positron");
        u.charge = std::fabs(d.num(b, "charge", Kind::Charge));
        const int ng = d.has(b, "energy") + d.has(b, "gamma") + d.has(b, "gamma_function");
        if (ng != 1) throw std::runtime_error(b + ": give one of energy, gamma, gamma_function");
        u.gamma = d.has(b, "gamma") ? d.num(b, "gamma", Kind::None) : d.has(b, "energy") ? 1 + d.num(b, "energy", Kind::Energy) / 0.51099895e6 : 2;
        if (!(u.gamma > 1)) throw std::runtime_error(b + ": gamma must be > 1");
        if (d.str(b, "distribution_r", "gaussian") != "function") { u.sigx = L(b, "sigma_x"); u.sigy = Ld(b, "sigma_y", u.sigx); }
        const std::string dz = d.str(b, "distribution_z", "gaussian");
        if (dz == "gaussian") u.sigz = L(b, "sigma_z");
        else if (dz == "flattop") { u.length = L(b, "length"); if (!(u.length > 0)) throw std::runtime_error(b + ".length must be > 0"); }
        else if (dz == "function") {
            u.profile_z = d.expr(b, "profile_z", {"zeta"});
            u.zeta_min = L(b, "zeta_min"); u.zeta_max = L(b, "zeta_max");
            if (!(u.zeta_max > u.zeta_min)) throw std::runtime_error(b + ": zeta_max must be > zeta_min");
        } else throw std::runtime_error(b + ".distribution_z must be gaussian, flattop or function");
        const std::string dr = d.str(b, "distribution_r", "gaussian");
        if (dr == "function") {
            u.profile_r = d.expr(b, "profile_r", {"x", "y", "r", "zeta"});
            u.r_extent = L(b, "r_extent");
            if (!(u.r_extent > 0)) throw std::runtime_error(b + ".r_extent must be > 0");
        } else if (dr != "gaussian") throw std::runtime_error(b + ".distribution_r must be gaussian or function");
        if (d.has(b, "gamma_function")) u.gamma_fn = d.expr(b, "gamma_function", {"zeta"});
        if (d.has(b, "ux_function")) u.ux_fn = d.expr(b, "ux_function", {"x", "y", "r", "zeta"});
        if (d.has(b, "uy_function")) u.uy_fn = d.expr(b, "uy_function", {"x", "y", "r", "zeta"});
        u.emitx = Ld(b, "emittance_x", 0); u.emity = Ld(b, "emittance_y", u.emitx);
        u.espread = d.num(b, "energy_spread", Kind::None, 0.0);
        u.zeta_center = Ld(b, "zeta_center", 0.75 * Lwin);
        u.x0 = Ld(b, "x0", 0); u.y0 = Ld(b, "y0", 0);
        u.nmacro = d.integer(b, "macroparticles", 100000);
        u.noise = d.str(b, "shot_noise", "random");
        if (u.noise != "random" && u.noise != "fawley") throw std::runtime_error(b + ".shot_noise must be random or fawley");
        u.noise_lambda = Ld(b, "noise_wavelength", 0);
        u.beamlet = int(d.integer(b, "beamlet", 4));
        u.seed = (unsigned long)d.integer(b, "seed", k);
        if (u.noise == "fawley" && (u.noise_lambda <= 0 || u.beamlet < 3 || u.nmacro % u.beamlet))
            throw std::runtime_error(b + ": fawley noise needs noise_wavelength > 0, beamlet >= 3, macroparticles a multiple of beamlet");
        if (!u.profile_z.empty()) u.zeta_center = 0.5 * (u.zeta_min + u.zeta_max);
        if (!u.gamma_fn.empty()) {                     // the bunch's gamma (summary, messages): mean over the profile
            const double z0 = u.profile_z.empty() ? u.zeta_center - 3 * std::max(u.sigz, u.length / 6) : u.zeta_min;
            const double z1 = u.profile_z.empty() ? u.zeta_center + 3 * std::max(u.sigz, u.length / 6) : u.zeta_max;
            double s0 = 0, s1 = 0;
            for (int k = 0; k <= 400; ++k) {
                const double z = z0 + (z1 - z0) * k / 400;
                const double w = u.profile_z.empty() ? (u.length > 0 ? 1.0 : std::exp(-0.5 * std::pow((z - u.zeta_center) / u.sigz, 2))) : std::max(0.0, u.profile_z({z}));
                s0 += w; s1 += w * u.gamma_fn({z});
            }
            u.gamma = s0 > 0 ? s1 / s0 : 2;
            if (!(u.gamma > 1)) throw std::runtime_error(b + ".gamma_function: the mean gamma must be > 1");
        }
        for (const auto& o : bunches) if (o.name == u.name) throw std::runtime_error(b + ".name: '" + u.name + "' given twice");
        bunches.push_back(u);
    }
    if (bunches.empty()) { nmacro = 0; charge = 0; gamma = 1; zeta_center = 0.75 * Lwin; return; }
    const Bunch& f = bunches[0];
    charge = f.charge; gamma = f.gamma; sigx = f.sigx; sigy = f.sigy; sigz = f.sigz; emitx = f.emitx; emity = f.emity;
    espread = f.espread; zeta_center = f.zeta_center; nmacro = f.nmacro; noise = f.noise; noise_lambda = f.noise_lambda;
    beamlet = f.beamlet; seed = f.seed;
}

std::vector<Config::Bunch> Config::bunch_list() const {
    if (!bunches.empty()) return bunches;
    Bunch u;
    u.charge = charge; u.gamma = gamma; u.sigx = sigx; u.sigy = sigy; u.sigz = sigz; u.emitx = emitx; u.emity = emity;
    u.espread = espread; u.zeta_center = zeta_center; u.nmacro = nmacro; u.noise = noise; u.noise_lambda = noise_lambda;
    u.beamlet = beamlet; u.seed = seed;
    return {u};
}

// All ranks draw the same bunches (each from its own seed); each rank keeps the particles of its half-cells.
void Simulation::load_bunch() {
    using cplx = std::complex<double>;
    const cplx I(0, 1);
    const int hlo = comm.rank == 0 ? 0 : a + 1, hhi = comm.rank == comm.size - 1 ? N - 2 : b;
    const double ylo = yf ? -1e300 : cfg.ynode[1], yhi = yf ? 1e300 : cfg.ynode[cfg.ny - 1];
    const double yoff = yf ? 0.5 * cfg.Ly : 0;        // Fourier y: box coordinate, physical y = p.y + Y0
    Axis zh = zax; zh.X = cfg.zeta.data();             // host copy of the axis
    std::vector<Particle> mine;
    const std::vector<Config::Bunch> list = cfg.bunch_list();
    for (size_t kb = 0; kb < list.size(); ++kb) {
    const Config::Bunch& B = list[kb];
    std::mt19937_64 rng(B.seed);
    std::normal_distribution<double> G(0, 1);
    std::uniform_real_distribution<double> U(-0.5, 0.5);
    const double Ne = B.charge / e_SI;
    const double W = (B.nmacro > 0 ? Ne / B.nmacro : 0) / (cfg.nc * std::pow(cfg.Lu, 3));   // code weight
    const double p0 = std::sqrt(B.gamma * B.gamma - 1);
    const int M = B.noise == "fawley" ? B.beamlet : 1;
    const long nb = B.nmacro / M;
    // profile_z: the inverse of its cumulative distribution (4096 intervals); profile_r: rejection, with the
    // largest value of the profile from a scan
    std::vector<double> cz;
    if (!B.profile_z.empty()) {
        const int K = 4096;
        cz.assign(K + 1, 0.0);
        double prev = std::max(0.0, B.profile_z({B.zeta_min}));
        for (int k = 1; k <= K; ++k) {
            const double f = std::max(0.0, B.profile_z({B.zeta_min + (B.zeta_max - B.zeta_min) * k / K}));
            cz[k] = cz[k - 1] + 0.5 * (prev + f); prev = f;
        }
        if (!(cz[K] > 0)) throw std::runtime_error("bunch " + B.name + ": profile_z is zero on [zeta_min, zeta_max]");
    }
    auto sample_zeta = [&](double u01) {
        const int K = int(cz.size()) - 1;
        const double t = u01 * cz[K];
        const int k = int(std::upper_bound(cz.begin(), cz.end(), t) - cz.begin()) - 1;
        const int kk = std::min(std::max(k, 0), K - 1);
        const double f = cz[kk + 1] > cz[kk] ? (t - cz[kk]) / (cz[kk + 1] - cz[kk]) : 0.5;
        return B.zeta_min + (B.zeta_max - B.zeta_min) * (kk + f) / K;
    };
    double fr = 0, srx = B.sigx, sry = B.sigy;        // srx, sry: rms sizes (of profile_r: from the scan)
    if (!B.profile_r.empty()) {
        double w = 0, wx = 0, wy = 0;
        const double z0 = cz.empty() ? B.zeta_center - 3 * std::max(B.sigz, B.length / 6) : B.zeta_min;
        const double z1 = cz.empty() ? B.zeta_center + 3 * std::max(B.sigz, B.length / 6) : B.zeta_max;
        for (int kz = 0; kz <= 16; ++kz)
            for (int i = -48; i <= 48; ++i)
                for (int j = -48; j <= 48; ++j) {
                    const double xr = B.r_extent * i / 48, yr = B.r_extent * j / 48, r = std::hypot(xr, yr);
                    if (r >= B.r_extent) continue;
                    const double f = std::max(0.0, B.profile_r({xr, yr, r, z0 + (z1 - z0) * kz / 16}));
                    fr = std::max(fr, f); w += f; wx += f * xr * xr; wy += f * yr * yr;
                }
        if (!(fr > 0)) throw std::runtime_error("bunch " + B.name + ": profile_r is zero in r < r_extent");
        srx = std::sqrt(wx / w); sry = std::sqrt(wy / w);
        fr *= 1.2;
    }
    const bool fn = !cz.empty() || !B.profile_r.empty() || !B.gamma_fn.empty() || !B.ux_fn.empty() || !B.uy_fn.empty();
    for (long ib = 0; ib < nb; ++ib) {
        double x, y, z, px, py, pz, zf = 0;              // zf: window position of a function profile
        if (!fn) {                                       // (the random sequence of the built-in profiles)
            x = B.x0 + B.sigx * G(rng); y = B.y0 + B.sigy * G(rng);
            z = B.length > 0 ? B.length * U(rng) : B.sigz * G(rng);
            px = B.sigx > 0 ? B.emitx / B.sigx * G(rng) : 0;
            py = B.sigy > 0 ? B.emity / B.sigy * G(rng) : 0;
            pz = p0 * (1 + B.espread * G(rng));
        } else {
            zf = cz.empty() ? B.zeta_center + (B.length > 0 ? B.length * U(rng) : B.sigz * G(rng)) : sample_zeta(U(rng) + 0.5);
            double xr, yr;
            if (B.profile_r.empty()) { xr = B.sigx * G(rng); yr = B.sigy * G(rng); }
            else for (;;) {
                xr = B.r_extent * 2 * U(rng); yr = B.r_extent * 2 * U(rng);
                const double r = std::hypot(xr, yr);
                if (r >= B.r_extent) continue;
                const double f = B.profile_r({xr, yr, r, zf});
                if (f > fr) throw std::runtime_error("bunch " + B.name + ": profile_r exceeds the estimate of its maximum (narrow peak?)");
                if ((U(rng) + 0.5) * fr < f) break;
            }
            const double r = std::hypot(xr, yr);
            x = B.x0 + xr; y = B.y0 + yr;
            px = B.emitx > 0 && srx > 0 ? B.emitx / srx * G(rng) : 0;
            py = B.emity > 0 && sry > 0 ? B.emity / sry * G(rng) : 0;
            if (!B.ux_fn.empty()) px += B.ux_fn({xr, yr, r, zf});
            if (!B.uy_fn.empty()) py += B.uy_fn({xr, yr, r, zf});
            const double gl = B.gamma_fn.empty() ? B.gamma : B.gamma_fn({zf});
            const double ptot = std::sqrt(gl * gl - 1) * (1 + B.espread * G(rng));
            if (px * px + py * py >= ptot * ptot) throw std::runtime_error("bunch " + B.name + ": transverse momentum above the total");
            pz = std::sqrt(ptot * ptot - px * px - py * py);       // the energy is kept
            z = (zf - B.zeta_center) * pz / std::sqrt(1 + ptot * ptot);   // lab offset: p.zeta = zf
        }
        std::vector<double> dz(M, 0.0);
        if (M > 1) {
            // beamlet: M particles spread evenly over noise_wavelength, displaced so that the bunching at the
            // harmonics h < M/2 equals that of the real electrons (Fawley, PRST-AB 5, 070701 (2002))
            const double Neb = Ne / nb, lam = B.noise_lambda;
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
            p.zeta = B.zeta_center + zz / vz;
            p.px = px; p.py = py; p.pz = pz; p.w = W; p.alive = 1;
            p.q = B.q; p.qm = B.q; p.sp = -int(kb); p.z = 0;
            int hc; double f;
            zh.locate(p.zeta, hc, f);
            if (p.zeta < 0 || hc > N - 2 || p.y < ylo || p.y > yhi) { if (comm.rank == 0) ++lost_total; continue; }
            if (hc >= hlo && hc <= hhi) mine.push_back(p);
        }
    }
    }
    set_particles(mine);
    X0 = -0.5 * cfg.Lx; Y0 = -yoff;
}


// The co-moving field -(lap_perp + gamma^-2 d_zeta^2) psi = rho depends on the bunch's gamma: with several bunches
// it is solved for each bunch with its own beta and the forward fields R are summed (the equation is linear).
// The sponge's bound field keeps the mean beta of all bunches. Collective.
void Simulation::comoving_fields_per_bunch() {
    const std::vector<Particle> all = host_particles();
    const size_t n = size_t(ns) * nk;
    Buf<cpx> sx(n), sy(n);
    for (int k = 0; k < cfg.nbunch(); ++k) {
        std::vector<Particle> one;
        for (const auto& p : all) if (p.sp == -k) one.push_back(p);
        set_particles(one);
        currents();
        comoving_field();
        cpx* ax = sx.data(); cpx* ay = sy.data(); const cpx* rx = Rx.data(); const cpx* ry = Ry.data();
        par_for(long(n), [=] HD(long i) { ax[i] = ax[i] + rx[i]; ay[i] = ay[i] + ry[i]; });
        device_sync();
    }
    set_particles(all);
    currents();
    cpx* rx = Rx.data(); cpx* ry = Ry.data(); const cpx* ax = sx.data(); const cpx* ay = sy.data();
    par_for(long(n), [=] HD(long i) { rx[i] = ax[i]; ry[i] = ay[i]; });
    device_sync();
    double m[2] = {0, 0};                                   // mean beta of all bunch particles (sponge)
    for (const auto& p : all)
        if (p.alive == 1) { m[0] += p.w * p.pz / std::sqrt(1 + p.px * p.px + p.py * p.py + p.pz * p.pz); m[1] += p.w; }
    comm.allreduce_sum(m, 2);
    beta_b = m[1] > 0 ? m[0] / m[1] : 1;
    gm2_b = 1 - beta_b * beta_b;
}
