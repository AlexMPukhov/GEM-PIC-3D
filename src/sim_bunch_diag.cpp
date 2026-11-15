// Bunch diagnostics as in GEM-PIC-RZ: bunches.txt (one line per output step and beam species: charge, energy,
// energy spread, sizes, normalized emittances, length, centroid) and slices_NNNNNN.txt (the same moments per
// slice of length output.slice_length along zeta: e.g. the centroid of the tail of a bunch).
#include "sim_internal.hpp"

namespace {
// moments of groups of beam particles; the second moments are accumulated centred (two passes), so the energy
// spread is exact also for dE/E << 1 at large gamma and does not depend on the number of MPI ranks
struct Moments {
    static constexpr int N1 = 8, N2 = 8;
    std::vector<double> s1, s2;        // per group: w, n, w x, w y, w zeta, w ux, w uy, w gamma; centred sums
    explicit Moments(size_t groups) : s1(groups * N1, 0.0), s2(groups * N2, 0.0) {}
    void add1(size_t g, double w, const double v[6]) {
        double* a = &s1[g * N1];
        a[0] += w; a[1] += 1;
        for (int k = 0; k < 6; ++k) a[2 + k] += w * v[k];
    }
    double mean(size_t g, int k) const { const double* a = &s1[g * N1]; return a[0] > 0 ? a[2 + k] / a[0] : 0; }
    void add2(size_t g, double w, const double v[6]) {
        double d[6];
        for (int k = 0; k < 6; ++k) d[k] = v[k] - mean(g, k);
        double* a = &s2[g * N2];
        a[0] += w * d[0] * d[0]; a[1] += w * d[1] * d[1]; a[2] += w * d[2] * d[2]; a[3] += w * d[3] * d[3];
        a[4] += w * d[4] * d[4]; a[5] += w * d[0] * d[3]; a[6] += w * d[1] * d[4]; a[7] += w * d[5] * d[5];
    }
    double var(size_t g, int k) const { const double w = s1[g * N1]; return w > 0 ? s2[g * N2 + k] / w : 0; }
};

// the line of one group: N, Q [pC], E [MeV], dE/E, sig_x, sig_y, eps_nx, eps_ny [um], <zeta>, sig_z [um], <x>, <y> [um]
void print_group(FILE* f, const Moments& M, size_t g, double pC, double um) {
    const double w = M.s1[g * Moments::N1], gm = M.mean(g, 5);
    const double ex = std::sqrt(std::max(0.0, M.var(g, 0) * M.var(g, 3) - M.var(g, 5) * M.var(g, 5)));
    const double ey = std::sqrt(std::max(0.0, M.var(g, 1) * M.var(g, 4) - M.var(g, 6) * M.var(g, 6)));
    std::fprintf(f, " %10ld %12.6f %13.6f %10.3e %10.5f %10.5f %10.5f %10.5f %11.4f %10.4f %11.5f %11.5f\n",
                 long(M.s1[g * Moments::N1 + 1]), w * pC, w > 0 ? (gm - 1) * 0.51099895 : 0.0, gm > 0 ? std::sqrt(M.var(g, 7)) / gm : 0.0,
                 std::sqrt(M.var(g, 0)) * um, std::sqrt(M.var(g, 1)) * um, ex * um, ey * um, M.mean(g, 2) * um,
                 std::sqrt(M.var(g, 2)) * um, M.mean(g, 0) * um, M.mean(g, 1) * um);
}
}  // namespace

// Collective. nsl slices of length len per beam species (nsl = 1: the whole species).
static Moments beam_moments(Simulation& S, int nsl, double len) {
    const std::vector<Particle> P = S.host_particles();
    const size_t ng = S.cfg.beam_names().size() * size_t(nsl);
    Moments M(ng);
    auto group = [&](const Particle& p, size_t& g, double v[6]) {
        if (p.alive != 1) return false;
        const int sl = nsl > 1 ? int(std::floor(p.zeta / len)) : 0;
        if (sl < 0 || sl >= nsl) return false;
        g = size_t(S.cfg.beam_index(p.sp)) * nsl + sl;
        v[0] = p.x + S.X0; v[1] = p.y + S.Y0; v[2] = p.zeta; v[3] = p.px; v[4] = p.py;
        v[5] = std::sqrt(1 + p.px * p.px + p.py * p.py + p.pz * p.pz);
        return true;
    };
    size_t g; double v[6];
    for (const auto& p : P) if (group(p, g, v)) M.add1(g, p.w, v);
    S.comm.allreduce_sum(M.s1.data(), int(M.s1.size()));
    for (const auto& p : P) if (group(p, g, v)) M.add2(g, p.w, v);
    S.comm.allreduce_sum(M.s2.data(), int(M.s2.size()));
    return M;
}

void Simulation::write_bunches(FILE* f, bool header) {
    const Moments M = beam_moments(*this, 1, 0);
    if (comm.rank != 0 || !f) return;
    if (header)
        std::fprintf(f, "# step  z[mm]  bunch  N  Q[pC]  E[MeV]  dE/E(rms)  sig_x[um]  sig_y[um]  eps_nx[um]  eps_ny[um]  <zeta>[um]  sig_z[um]  "
                        "<x>[um]  <y>[um]    (trapped = self-injected plasma electrons, inj_<ion> = from field ionization)\n");
    const double pC = cfg.nc * std::pow(cfg.Lu, 3) * e_SI * 1e12, um = cfg.Lu * 1e6;
    const std::vector<std::string> names = cfg.beam_names();
    for (size_t k = 0; k < names.size(); ++k) {
        std::fprintf(f, "%6ld %10.5f %-12s", istep, s * cfg.Lu * 1e3, names[k].c_str());
        print_group(f, M, k, pC, um);
    }
    std::fflush(f);
}

void Simulation::write_slices(const std::string& file) {
    const double len = cfg.slice_length;
    const int nsl = std::max(1, int(std::ceil(cfg.Lwin / len - 1e-9)));
    const Moments M = beam_moments(*this, nsl, len);
    if (comm.rank != 0) return;
    FILE* f = std::fopen(file.c_str(), "w");
    if (!f) throw std::runtime_error("cannot write " + file);
    std::fprintf(f, "# GEM-PIC-3D beam slices of %.9g um, step %ld, z = %.9g mm\n", len * cfg.Lu * 1e6, istep, s * cfg.Lu * 1e3);
    std::fprintf(f, "# bunch  zeta_centre[um]  N  Q[pC]  E[MeV]  dE/E(rms)  sig_x[um]  sig_y[um]  eps_nx[um]  eps_ny[um]  <zeta>[um]  sig_z[um]  <x>[um]  <y>[um]\n");
    const double pC = cfg.nc * std::pow(cfg.Lu, 3) * e_SI * 1e12, um = cfg.Lu * 1e6;
    const std::vector<std::string> names = cfg.beam_names();
    for (size_t k = 0; k < names.size(); ++k)
        for (int sl = 0; sl < nsl; ++sl) {
            const size_t g = k * nsl + sl;
            if (!(M.s1[g * Moments::N1] > 0)) continue;
            std::fprintf(f, "%-12s %11.4f", names[k].c_str(), (sl + 0.5) * len * um);
            print_group(f, M, g, pC, um);
        }
    std::fclose(f);
}
