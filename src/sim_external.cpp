// Undulators and transport elements of the deck (as GEM-PIC-RZ): blocks undulator, quadrupole, plasma_lens,
// solenoid (each may be repeated: undulator2, ...). The fields are in src/external.hpp.
#include "sim_internal.hpp"

void Config::parse_external(const Deck& d) {
    auto L = [&](const std::string& b, const std::string& k) { return d.num(b, k, Kind::Length) / Lu; };
    auto Ld = [&](const std::string& b, const std::string& k, double def) { return d.has(b, k) ? L(b, k) : def; };
    const double omega0 = c_SI / Lu, Bunit = me_SI * omega0 / e_SI;          // code unit of B [T]
    // reference momentum of the magnets' resolution: the lowest of the bunches (gamma beta)
    double pref = 1e300;
    for (const auto& b : bunch_list()) if (b.nmacro > 0) pref = std::min(pref, std::sqrt(b.gamma * b.gamma - 1));
    if (pref > 1e299) pref = 1;
    undulators.clear(); magnets.clear();
    for (int k = 1;; ++k) {
        const std::string b = k == 1 ? "undulator" : "undulator" + std::to_string(k);
        if (!d.has_block(b)) break;
        Undulator u{};
        const std::string t = d.str(b, "type", "planar");
        if (t == "helical") u.helical = 1;
        else if (t != "planar") throw std::runtime_error(b + ".type must be planar or helical");
        const double lu = L(b, "period");
        if (!(lu > 0)) throw std::runtime_error(b + ".period must be positive");
        u.ku = 2 * M_PI / lu;
        if (d.has(b, "K") == d.has(b, "K_function")) throw std::runtime_error(b + ": give either K or K_function");
        u.B0 = d.has(b, "K") ? d.num(b, "K", Kind::None) * u.ku : 0.0;
        u.z0 = Ld(b, "z_start", 0);
        if (d.has(b, "periods") == d.has(b, "length")) throw std::runtime_error(b + ": give either periods or length");
        const double len = d.has(b, "periods") ? d.integer(b, "periods", 0) * lu : L(b, "length");
        if (!(len > 0)) throw std::runtime_error(b + ": the length must be positive");
        u.z1 = u.z0 + len;
        const long rp = d.integer(b, "ramp_periods", 1);
        if (rp < 0 || (u.helical && rp < 1) || 2 * rp * lu > len + 1e-9)
            throw std::runtime_error(b + ".ramp_periods must be >= 0 (helical >= 1) and fit twice into the length");
        u.ramp = rp * lu;
        u.focusing = u.helical ? 0 : d.flag(b, "natural_focusing", true);
        if (d.has(b, "K_function")) {                 // taper K(z), z the lab position: 32 points per period
            const Expr Kf = d.expr(b, "K_function", {"z"});
            u.nk = int(std::ceil(len / lu * 32)) + 1; u.dzk = len / (u.nk - 1); u.koff = int(ktab.size());
            for (int j = 0; j < u.nk; ++j) ktab.push_back(Kf({u.z0 + j * u.dzk}) * u.ku);
            for (int j = 0; j < u.nk; ++j) u.B0 = std::max(u.B0, std::fabs(ktab[u.koff + j]));
        }
        undulators.push_back(u);
    }
    const char* kinds[3] = {"quadrupole", "plasma_lens", "solenoid"};
    for (int kind = 0; kind < 3; ++kind)
        for (int k = 1;; ++k) {
            const std::string b = k == 1 ? std::string(kinds[kind]) : kinds[kind] + std::to_string(k);
            if (!d.has_block(b)) break;
            Magnet m{};
            m.kind = kind;
            m.z0 = Ld(b, "z_start", 0);
            m.z1 = m.z0 + L(b, "length");
            if (!(m.z1 > m.z0)) throw std::runtime_error(b + ".length must be positive");
            m.R = 1e300; m.ramp = 0;
            double kb;                                         // betatron wavenumber of the reference particle
            if (kind < 2) {
                m.g = d.num(b, "gradient", Kind::Gradient) / Bunit * Lu;
                if (kind == 1) m.R = Ld(b, "radius", 1e300);
                kb = std::sqrt(std::fabs(m.g) / pref);
            } else {
                m.g = d.num(b, "field", Kind::Field) / Bunit;
                m.ramp = Ld(b, "ramp", 0);
                if (m.ramp < 0 || 2 * m.ramp > m.z1 - m.z0) throw std::runtime_error(b + ".ramp must be >= 0 and fit twice into the length");
                kb = std::fabs(m.g) / (2 * pref);
            }
            m.dsmax = kb > 0 ? 0.1 / kb : 1e300;
            if (kind == 2 && m.ramp > 0) m.dsmax = std::min(m.dsmax, 0.25 * m.ramp);
            magnets.push_back(m);
        }
}

// summary lines of the undulators and magnets
void Config::summary_external(FILE* f) const {
    const double um = Lu * 1e6, Bunit = me_SI * c_SI / Lu / e_SI;
    for (const auto& u : undulators) {
        const double lu = 2 * M_PI / u.ku, K = u.B0 / u.ku;
        std::fprintf(f, "  undulator: %s, period %.4g mm, K = %.4g%s (B0 = %.4g T), z = %.4g ... %.4g mm, ramps %.3g periods%s\n",
                     u.helical ? "helical" : "planar", lu * Lu * 1e3, K, u.nk > 0 ? " (largest of the taper K_function)" : "",
                     u.B0 * Bunit, u.z0 * Lu * 1e3, u.z1 * Lu * 1e3, u.ramp / lu, u.helical || u.focusing ? "" : ", no natural focusing");
        for (const auto& b : bunch_list())
            if (b.nmacro > 0)
                std::fprintf(f, "    bunch %s: resonant wavelength %.4g um\n", b.name.c_str(),
                             lu * (1 + (u.helical ? K * K : 0.5 * K * K)) / (2 * b.gamma * b.gamma) * um);
        if (dz > lu / 16 * (1 + 1e-9)) std::fprintf(f, "    WARNING: dz > lambda_u / 16 (the current oscillates with the period; the push itself\n"
                                                "    takes sub-steps of lambda_u / 32 inside the undulator)\n");
    }
    const char* nm[3] = {"quadrupole", "plasma lens", "solenoid"};
    for (const auto& m : magnets)
        std::fprintf(f, "  %s: z = %.4g ... %.4g mm, %s %.4g %s%s\n", nm[m.kind], m.z0 * Lu * 1e3, m.z1 * Lu * 1e3,
                     m.kind < 2 ? "gradient" : "field", m.kind < 2 ? m.g * Bunit / Lu : m.g * Bunit, m.kind < 2 ? "T/m" : "T",
                     m.kind == 1 && m.R < 1e299 ? (", radius " + std::to_string(m.R * um) + " um").c_str() : "");
}
