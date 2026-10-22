// Output: diag.txt, field energy, particle dumps, GUI frames and axis lineouts.
#include "sim_internal.hpp"

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
