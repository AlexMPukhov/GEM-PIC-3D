// Ions: the ion blocks of the deck, the ADK tables of the field ionization and ionization.txt (as GEM-PIC-RZ).
#include "sim_internal.hpp"
#include "ionization.hpp"

// ion blocks: ion, ion2, ... (keys as in GEM-PIC-RZ), and plasma.free_electrons
void Config::parse_ions(const Deck& d) {
    free_electrons = d.flag("plasma", "free_electrons", true);
    double qsum = 0;
    for (int k = 1; d.has_block(k == 1 ? "ion" : "ion" + std::to_string(k)); ++k) {
        const std::string b = k == 1 ? "ion" : "ion" + std::to_string(k);
        if (!plasma_on) throw std::runtime_error(b + ": ion blocks need a plasma block");
        IonSpecies s;
        const std::string mode = d.str(b, "ionization", "none");
        if (mode == "adk") s.ionization = true;
        else if (mode != "none") throw std::runtime_error(b + ".ionization must be none or adk");
        const bool el = d.has(b, "element"), ms = d.has(b, "mass");
        if (el == ms) throw std::runtime_error(b + ": give either element or mass");
        int Znuc = 0;
        if (el) {
            const gem::Element* e = gem::findElement(d.str(b, "element"));
            if (!e) throw std::runtime_error(b + ".element: unknown element '" + d.str(b, "element") + "'");
            s.mass = e->mass_u * 1.66053906660e-27 / me_SI;
            s.name = e->symbol;
            Znuc = e->Znuc;
            s.ip_eV = e->ip_eV;
            if (!s.ionization && Znuc > 1 && !d.has(b, "charge_state"))
                throw std::runtime_error(b + ": charge_state is required for " + e->symbol);
        } else {
            s.mass = d.num(b, "mass", Kind::Mass) / me_SI;
            if (!(s.mass > 0)) throw std::runtime_error(b + ".mass must be positive");
        }
        s.Z0 = int(d.integer(b, "charge_state", s.ionization ? 0 : 1));
        if (d.has(b, "ionization_potentials")) {
            std::vector<double> ip = d.list(b, "ionization_potentials", Kind::Energy);
            if (el && int(ip.size()) != Znuc)
                throw std::runtime_error(b + ".ionization_potentials: give all " + std::to_string(Znuc) + " levels of " + s.name);
            s.ip_eV = ip;
            if (!el) Znuc = int(ip.size());
        }
        if (Znuc == 0) Znuc = std::max(s.Z0, 1);                 // mass without potentials: no levels
        if (s.Z0 < 0 || s.Z0 > Znuc) throw std::runtime_error(b + ".charge_state must be 0 ... " + std::to_string(Znuc));
        if (s.ionization) {
            if (s.ip_eV.empty()) throw std::runtime_error(b + ": ionization of an ion given by mass needs ionization_potentials");
            s.Zmax = int(d.integer(b, "max_charge_state", Znuc));
            if (s.Zmax <= s.Z0 || s.Zmax > Znuc)
                throw std::runtime_error(b + ".max_charge_state must be > charge_state and <= " + std::to_string(Znuc));
        } else {
            if (s.Z0 < 1) throw std::runtime_error(b + ".charge_state must be >= 1 without ionization");
            if (d.has(b, "max_charge_state")) throw std::runtime_error(b + ".max_charge_state needs ionization = adk");
            s.Zmax = s.Z0;
        }
        if (d.has(b, "fraction") && d.has(b, "density")) throw std::runtime_error(b + ": give either fraction or density");
        if (d.has(b, "density")) s.atoms = d.num(b, "density", Kind::Density) / nc / n0;
        else if (s.Z0 >= 1) s.atoms = d.num(b, "fraction", Kind::None, 1.0) / s.Z0;
        else throw std::runtime_error(b + ": neutral atoms (charge_state = 0) need density");
        if (!(s.atoms > 0)) throw std::runtime_error(b + ": fraction (density) must be positive");
        if (!free_electrons && s.Z0 != 0) throw std::runtime_error(b + ": with plasma.free_electrons = false all ions start neutral (charge_state = 0)");
        s.mobile = d.flag(b, "mobile", true);
        s.name = d.str(b, "name", el ? s.name + (s.Z0 > 1 ? std::to_string(s.Z0) + "+" : "") : "ion" + std::to_string(k));
        qsum += s.Z0 * s.atoms;
        ions.push_back(s);
    }
    if (qsum > 1 + 1e-12)
        throw std::runtime_error("ion blocks: the ion charge (sum of Z0 n_atom) exceeds the free-electron density");
}

// ADK tables of the ionizing species (device), the per-species constants, the ionization.txt header
void Simulation::setup_ions() {
    if (cfg.ions.empty()) return;
    std::vector<IonDev> iv;
    std::vector<AdkLevel> lev;
    const double omega0 = c_SI / cfg.Lu, E0 = me_SI * c_SI * omega0 / e_SI;
    for (const auto& s : cfg.ions) {
        IonDev d{};
        d.mass = s.mass; d.mobile = s.mobile; d.ionizing = s.ionization; d.Zmax = s.Zmax; d.off = int(lev.size()); d.Emin = 1e300;
        if (s.ionization) {
            const std::vector<AdkLevel> L = adk_levels(s.ip_eV, s.Z0, s.Zmax, E0, omega0, 2 * cfg.Lwin);
            for (int k = s.Z0; k < s.Zmax; ++k) d.Emin = std::min(d.Emin, L[k].Emin);
            lev.insert(lev.end(), L.begin(), L.end());
        }
        iv.push_back(d);
    }
    ion_dev.upload(iv);
    if (!lev.empty()) adk_dev.upload(lev);
    bcnt.resize(1);
    ion_stats.assign(cfg.ions.size(), std::vector<double>());
}

// ionization.txt (rank 0, one line per step and ionizing species): step, z [mm], name, mean charge state and the
// charge-state fractions of the ions that left the window in this step, charge freed in this step [pC]
void Simulation::write_ionization(FILE* f, bool header) {
    if (!cfg.ionizing()) return;
    if (!header && comm.size > 1)                       // collective: sums over the ranks
        for (size_t k = 0; k < cfg.ions.size(); ++k) {
            if (!cfg.ions[k].ionization) continue;
            std::vector<double>& h = ion_stats[k];
            if (h.empty()) h.assign(cfg.ions[k].Zmax + 2, 0.0);
            comm.allreduce_sum(h.data(), int(h.size()));
        }
    if (!f) return;
    if (header) {
        std::fprintf(f, "# GEM-PIC-3D ionization: ions leaving the window per step\n# step z_mm name <Z> freed_pC f(0) f(1) ... f(Zmax)\n");
        return;
    }
    const double pC = cfg.nc * std::pow(cfg.Lu, 3) * e_SI * 1e12;
    for (size_t k = 0; k < cfg.ions.size(); ++k) {
        const auto& is = cfg.ions[k];
        if (!is.ionization) continue;
        const std::vector<double>& h = ion_stats[k];      // [0..Zmax]: weight per charge state, [Zmax+1]: freed
        double tot = 0, zm = 0;
        for (int z = 0; z <= is.Zmax; ++z) { tot += h.empty() ? 0 : h[z]; zm += h.empty() ? 0 : z * h[z]; }
        std::fprintf(f, "%6ld %10.5f %s %.6f %.6e", istep, s * cfg.Lu * 1e3, is.name.c_str(), tot > 0 ? zm / tot : 0.0,
                     h.empty() ? 0.0 : h[is.Zmax + 1] * pC);
        for (int z = 0; z <= is.Zmax; ++z) std::fprintf(f, " %.6e", tot > 0 ? h[z] / tot : 0.0);
        std::fprintf(f, "\n");
    }
    std::fflush(f);
}
