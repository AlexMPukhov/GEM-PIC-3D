// Tests: field ionization (ADK rate, Monte Carlo ionization of H and N in a plane-wave pulse).
#include "test_util.hpp"
#include "ionization.hpp"
#include "atomic_data.hpp"

// 17. ADK rate: for hydrogen-like levels (U = Z^2 U_H, Z after the ionization) the formula is the Landau rate
//     4 omega_a Z^5 (E_a/E) exp(-2 Z^3 E_a / 3E); the code-unit tables equal the SI formula.
void test_adk_rate() {
    double dl = 0;
    for (int Z : {1, 2, 7})
        for (double e : {0.02, 0.1, 0.5}) {
            const double E = e * Z * Z * Z * E_ATOMIC_VM;
            const double L = 4 * OMEGA_ATOMIC * std::pow(Z, 5) * (E_ATOMIC_VM / E) * std::exp(-2.0 * Z * Z * Z * E_ATOMIC_VM / (3 * E));
            dl = std::max(dl, std::fabs(adk_rate_SI(Z * Z * U_H_EV, Z, E) / L - 1));
        }
    check(dl < 1e-12, "ionization: ADK rate of hydrogen-like levels = Landau rate", dl, 0, 1e-12);
    const gem::Element* N = gem::findElement("N");
    const double lam = 0.8e-6, omega0 = 2 * M_PI * 299792458.0 / lam, E0 = 9.1093837015e-31 * 299792458.0 * omega0 / 1.602176634e-19;
    const std::vector<AdkLevel> lev = adk_levels(N->ip_eV, 5, 7, E0, omega0, 1e3);
    double dt = 0;
    for (int k = 5; k < 7; ++k)
        for (double a : {0.5, 1.0, 2.0}) {
            const double E = a * (k == 5 ? 1.0 : 1.5);
            const double ref = adk_rate_SI(N->ip_eV[k], k + 1, E * E0) / omega0;
            if (E >= lev[k].Emin && E < lev[k].Epeak) dt = std::max(dt, std::fabs(adk_rate(lev[k], E) / ref - 1));
        }
    check(dt < 1e-12, "ionization: ADK table in code units = SI formula (N5+, N6+)", dt, 0, 1e-12);
}

// 18. Field ionization in a plane-wave pulse (the analytic laser: tau_fwhm = 30 k0^-1, lambda0 = 0.8 um, s-polarized)
//     passing atoms that are loaded at the front of the window (density 1e-6 n_c, so the plasma fields are
//     negligible). Reference: the rate equations in the field of the pulse, E = -dA/deta (A from its definition,
//     differentiated numerically), integrated with fine steps along the path of an atom at rest; the
//     Monte Carlo scatter is about 0.5 / sqrt(4096).
//     (a) neutral hydrogen, fixed atoms, a0 = 0.01: ionized fraction 1 - exp(-int w dt);
//     (b) N5+ -> N6+ -> N7+, mobile ions, a0 = 2: the fractions of the three charge states (exact two-level
//         solution per fine step).
std::vector<double> ionization_run(const char* el, int Z0, int Zmax, bool mobile, double a0, std::vector<double>& ref) {
    Config c = Config::base_units(0.8e-6);
    const double tau = 30, h = 2 * M_PI / 40;
    const int N = int(6 * tau / h) + 1;
    c.nx = 8; c.Lx = 16; c.y_uniform(8, 16); c.zeta_uniform(N, h * (N - 1));
    c.sponge = c.sponge_y = 0; c.move_x = c.move_y = false;
    c.plasma_on = true; c.n0 = 1e-6; c.ppc_x = 8; c.ppc_y = 8; c.free_electrons = Z0 > 0;
    c.nmacro = 0; c.dz = 10; c.dz_push = 10; c.z_end = 10;
    c.laser_on = true; c.a0 = a0; c.kL = 1; c.tau = tau; c.theta = 0; c.pol = "s"; c.cep = 0;
    c.zeta_center = 0.5 * c.Lwin; c.start_fwhm = 0; c.stop_fwhm = 0;   // the pulse peak at the centre of the window
    const gem::Element* E = gem::findElement(el);
    Config::IonSpecies s;
    s.name = el; s.mass = E->mass_u * 1822.888486; s.Z0 = Z0; s.Zmax = Zmax; s.ionization = true; s.mobile = mobile;
    s.ip_eV = E->ip_eV; s.atoms = Z0 > 0 ? 1.0 / Z0 : 1.0;
    c.ions.push_back(s);
    Simulation sim(c, comm);
    sim.init();
    sim.step();
    std::vector<double> h0 = sim.ion_stats[0], frac(Zmax - Z0 + 1);
    double tot = 0;
    for (int z = 0; z <= Zmax; ++z) tot += h0.empty() ? 0 : h0[z];
    for (int z = Z0; z <= Zmax; ++z) frac[z - Z0] = tot > 0 ? h0[z] / tot : 0;
    // reference: A(eta) = a0 g cos(eta - eta_c), g = exp(-2 ln2 (eta - eta_c)^2 / tau^2), E = -dA/deta; the atom
    // crosses the window, eta from -(Lwin - zc) to zc (eta_c = 0)
    const double omega0 = 299792458.0 / c.Lu, E0 = 9.1093837015e-31 * 299792458.0 * omega0 / 1.602176634e-19;
    auto A = [&](double et) { return a0 * std::exp(-2 * std::log(2.0) * et * et / (tau * tau)) * std::cos(et); };
    std::vector<double> n(Zmax - Z0 + 1, 0.0);
    n[0] = 1;
    const double dt = h / 50, eps = 1e-4;
    for (double et = -(c.Lwin - c.zeta_center); et < c.zeta_center; et += dt) {
        const double e = std::fabs(-(A(et + 0.5 * dt + eps) - A(et + 0.5 * dt - eps)) / (2 * eps)) * E0;
        std::vector<double> w(Zmax - Z0, 0.0);
        for (int k = Z0; k < Zmax; ++k) w[k - Z0] = adk_rate_SI(E->ip_eV[k], k + 1, e) * dt / omega0;
        if (Zmax - Z0 == 1) { const double d = n[0] * (1 - std::exp(-w[0])); n[0] -= d; n[1] += d; }
        else {                                   // two levels, exact for constant rates over the step
            const double a = w[0], b = w[1], ea = std::exp(-a), eb = std::exp(-b);
            const double n1 = n[1] * eb + (std::fabs(b - a) > 1e-12 ? n[0] * a * (ea - eb) / (b - a) : n[0] * a * ea);
            const double n0n = n[0] * ea;
            n[2] += (n[0] - n0n) + (n[1] - n1); n[0] = n0n; n[1] = n1;
        }
    }
    ref = n;
    return frac;
}
void test_field_ionization() {
    std::vector<double> ref;
    std::vector<double> f = ionization_run("H", 0, 1, false, 0.01, ref);
    close("ionization: H ionized fraction (fixed atoms, a0 = 0.01) = 1 - exp(-int w dt)", f[1], ref[1], 0.03);
    f = ionization_run("N", 5, 7, true, 2.0, ref);
    close("ionization: N5+ fraction after the pulse (a0 = 2) = rate equations", f[0], ref[0], 0.05);
    close("ionization: N6+ fraction after the pulse = rate equations", f[1], ref[1], 0.03);
    close("ionization: N7+ fraction after the pulse = rate equations", f[2], ref[2], 0.10);
}
