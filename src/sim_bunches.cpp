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
        if (d.has(b, "energy") == d.has(b, "gamma")) throw std::runtime_error(b + ": give either energy or gamma");
        u.gamma = d.has(b, "gamma") ? d.num(b, "gamma", Kind::None) : 1 + d.num(b, "energy", Kind::Energy) / 0.51099895e6;
        if (!(u.gamma > 1)) throw std::runtime_error(b + ": gamma must be > 1");
        u.sigx = L(b, "sigma_x"); u.sigy = Ld(b, "sigma_y", u.sigx);
        const std::string dz = d.str(b, "distribution_z", "gaussian");
        if (dz == "gaussian") u.sigz = L(b, "sigma_z");
        else if (dz == "flattop") { u.length = L(b, "length"); if (!(u.length > 0)) throw std::runtime_error(b + ".length must be > 0"); }
        else throw std::runtime_error(b + ".distribution_z must be gaussian or flattop");
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
    for (long ib = 0; ib < nb; ++ib) {
        const double x = B.x0 + B.sigx * G(rng), y = B.y0 + B.sigy * G(rng);
        const double z = B.length > 0 ? B.length * U(rng) : B.sigz * G(rng);
        const double px = B.sigx > 0 ? B.emitx / B.sigx * G(rng) : 0;
        const double py = B.sigy > 0 ? B.emity / B.sigy * G(rng) : 0;
        const double pz = p0 * (1 + B.espread * G(rng));
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
