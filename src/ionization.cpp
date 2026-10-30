// ADK rates in code units (see ionization.hpp; ported from GEM-PIC-RZ src/ionization.cpp).
#include "ionization.hpp"
#include <algorithm>
#include <cmath>

namespace {
// ln of omega_a C^2 (U/2U_H) (2 (U/U_H)^{3/2} E_a)^{2n*-1} (SI, E in V/m), the power 2n*-1 and the exponent
// factor -(2/3)(U/U_H)^{3/2} E_a
void adk_coefficients(double U_eV, int Zafter, double& logpre, double& power, double& expo) {
    const double x = U_eV / U_H_EV;
    const double nstar = Zafter / std::sqrt(x);
    const double logC2 = 2.0 * nstar * std::log(2.0) - std::log(nstar) - std::lgamma(nstar + 1.0) - std::lgamma(nstar);
    power = 2.0 * nstar - 1.0;
    logpre = std::log(OMEGA_ATOMIC) + logC2 + std::log(0.5 * x) + power * std::log(2.0 * std::pow(x, 1.5) * E_ATOMIC_VM);
    expo = -(2.0 / 3.0) * std::pow(x, 1.5) * E_ATOMIC_VM;
}
double raw(const AdkLevel& L, double E) {
    E = std::min(E, L.Epeak);
    return E > 0 ? std::exp(L.logpre - L.power * std::log(E) + L.expo / E) : 0.0;
}
}  // namespace

double adk_rate_SI(double U_eV, int Zafter, double E_Vm) {
    if (!(E_Vm > 0)) return 0.0;
    double logpre, power, expo;
    adk_coefficients(U_eV, Zafter, logpre, power, expo);
    return std::exp(logpre - power * std::log(E_Vm) + expo / E_Vm);
}

std::vector<AdkLevel> adk_levels(const std::vector<double>& ip_eV, int Z0, int Zmax, double E0_SI, double omega0, double T) {
    std::vector<AdkLevel> lev(std::max(Zmax, 0), AdkLevel{0, 0, 0, 1e300, 1e300});
    for (int k = Z0; k < Zmax; ++k) {
        AdkLevel& L = lev[k];
        double logpre, power, expo;
        adk_coefficients(ip_eV[k], k + 1, logpre, power, expo);
        // w_code(E) = w_SI(E E0) / omega0
        L.power = power;
        L.logpre = logpre - std::log(omega0) - power * std::log(E0_SI);
        L.expo = expo / E0_SI;
        L.Epeak = power > 0 ? -L.expo / power : 1e300;
        // skip threshold: w(Emin) T = 1e-10 (w increases below Epeak)
        double lo = 1e-12, hi = std::min(L.Epeak, 1e6);
        if (raw(L, hi) * T < 1e-10) L.Emin = hi;
        else {
            for (int it = 0; it < 200; ++it) {
                const double mid = std::sqrt(lo * hi);
                (raw(L, mid) * T < 1e-10 ? lo : hi) = mid;
            }
            L.Emin = lo;
        }
    }
    return lev;
}
