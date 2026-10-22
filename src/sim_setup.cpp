// Setup of a Simulation (grids, FFTs, buffers, bunch loading) and the time step with the window shift.
#include "sim_internal.hpp"

// ---------------------------------------------------------------- setup
Simulation::Simulation(const Config& c, Comm& cm) : cfg(c), comm(cm) {
    N = cfg.nzeta; nk = cfg.nx * cfg.ny;
    dx = cfg.Lx / cfg.nx;
    if (int(cfg.zeta.size()) != N || int(cfg.ynode.size()) != cfg.ny + 1) throw std::runtime_error("grids not set");
    // blocks of consecutive slices, rank P-1 at the front of the window
    const int P = comm.size, base = N / P, rem = N % P;
    a = 0;
    for (int r = 0; r < comm.rank; ++r) a += base + (r < rem);
    ns = base + (comm.rank < rem);
    b = a + ns;
    if (P > 1 && base < 4) throw std::runtime_error("MPI: at least 4 slices per rank are needed");

    // x: Fourier modes
    const int nx = cfg.nx;
    std::vector<double> hkx(nx), hsx(nx), hf(nx), hm(nx);
    auto prof = [&](double dd, double w) {          // dd: distance from the box edge
        if (w <= 0 || dd >= w) return 1.0;
        const double r = 1 - dd / w;
        return 1 - cfg.sponge_rate * r * r;
    };
    for (int ixx = 0; ixx < nx; ++ixx) {
        const int mx = ixx <= nx / 2 ? ixx : ixx - nx;
        hsx[ixx] = 2 * M_PI * mx / cfg.Lx;
        hkx[ixx] = (2 * ixx == nx) ? 0 : hsx[ixx];          // odd derivatives vanish at Nyquist
        hf[ixx] = cfg.k_filter ? std::pow(std::cos(0.5 * hsx[ixx] * dx), 2) : 1.0;
        if (2 * ixx == nx) hf[ixx] = 0;
        hm[ixx] = prof(std::min(ixx * dx, cfg.Lx - ixx * dx), cfg.sponge);
    }
    kxd.upload(hkx); kxs.upload(hsx); filt.upload(hf); maskx.upload(hm);
    yf = cfg.y_fourier;
    if (yf) setup_y_fourier(); else setup_y();
    setup_zeta();

    const size_t pl = size_t(nk);
    for (Buf<cpx>* p : {&Rx, &Ry, &Ez, &Bz, &Lx, &Ly}) p->resize(ns * pl);
    bnd.resize(6 * pl);
    jzg.resize(2 * pl);
    for (Buf<cpx>* p : {&j0x, &j0y, &jkx, &jky, &jkz, &S1, &S2}) p->resize((ns + 1) * pl);
    if (!yf) {
        for (Buf<cpx>* p : {&sRx, &sRy, &sLx, &sLy, &srhs}) p->resize(pl);
        scp.resize(pl);
    }
    jw.resize(3 * (ns + 3) * pl);
    F.resize(6 * (ns + 2) * pl);
    fft_ = std::make_unique<FftMany>(cfg.nx, cfg.ny, yf);

    laser.on = cfg.laser_on; laser.a0 = cfg.a0; laser.kL = cfg.kL; laser.tau = cfg.tau; laser.cep = cfg.cep;
    laser.kx = std::sin(cfg.theta); laser.kz = std::cos(cfg.theta);
    laser.pol = cfg.pol == "s" ? 0 : cfg.pol == "p" ? 1 : 2;
    laser.eta_c = cfg.start_fwhm * cfg.tau;   // the bunch centre starts at eta = 0
    setup_plasma();
}

// Fourier y: k_y per iy as for x, all components on the nodes j*dy of the box (physical y = j*dy + Y0).
void Simulation::setup_y_fourier() {
    const int ny = cfg.ny;
    dy = cfg.Ly / ny;
    std::vector<double> hky(ny), hsy(ny), hf(ny), hm(ny), hl(ny), hi(ny, 1 / dy);
    for (int j = 0; j < ny; ++j) {
        const int my = j <= ny / 2 ? j : j - ny;
        hsy[j] = 2 * M_PI * my / cfg.Ly;
        hky[j] = (2 * j == ny) ? 0 : hsy[j];
        hf[j] = cfg.k_filter ? std::pow(std::cos(0.5 * hsy[j] * dy), 2) : 1.0;
        if (2 * j == ny) hf[j] = 0;
        const double dd = std::min(j * dy, cfg.Ly - j * dy), w = cfg.sponge_y;
        double m = 1;
        if (w > 0 && dd < w) { const double r = 1 - dd / w; m = 1 - cfg.sponge_rate * r * r; }
        hm[j] = m; hl[j] = hky[j] * hky[j];
    }
    kyd.upload(hky); kys.upload(hsy); filty.upload(hf); masky_n.upload(hm); masky_h.upload(hm); evl.upload(hl);
    iD.upload(hi); iA.upload(hi);
    hDel.assign(ny, dy); hA.assign(ny, dy);
    yax.X = nullptr; yax.n = ny + 1; yax.uniform = 1; yax.x0 = 0; yax.ih = 1 / dy;
}

// Free-hand y: nodes j = 0..ny (0 and ny are the conducting walls), half-nodes j+1/2 = midpoints, j = 0..ny-1.
// E_x, B_y (R_x, L_x), E_z, j_x, j_z live on the nodes; E_y, B_x (R_y, L_y), B_z, j_y on the half-nodes.
void Simulation::setup_y() {
    const int ny = cfg.ny, nx = cfg.nx;
    const std::vector<double>& Y = cfg.ynode;
    hDel.assign(ny, 0); hA.assign(ny, 0);
    for (int j = 0; j < ny; ++j) hDel[j] = Y[j + 1] - Y[j];
    hA[0] = 0.5 * hDel[0];
    for (int j = 1; j < ny; ++j) hA[j] = 0.5 * (hDel[j - 1] + hDel[j]);
    std::vector<double> hiD(ny), hiA(ny), hmn(ny), hmh(ny);
    auto prof = [&](double y) {
        const double dd = 0.5 * cfg.Ly - std::fabs(y), w = cfg.sponge_y;
        if (w <= 0 || dd >= w) return 1.0;
        const double r = 1 - dd / w;
        return 1 - cfg.sponge_rate * r * r;
    };
    for (int j = 0; j < ny; ++j) {
        hiD[j] = 1 / hDel[j]; hiA[j] = 1 / hA[j];
        hmn[j] = prof(Y[j]); hmh[j] = prof(0.5 * (Y[j] + Y[j + 1]));
    }
    yn.upload(Y); iD.upload(hiD); iA.upload(hiA); masky_n.upload(hmn); masky_h.upload(hmh);
    yax.X = yn.data(); yax.n = ny + 1;
    bool uni = true;
    for (int j = 0; j < ny; ++j) uni = uni && hDel[j] == hDel[0];
    yax.uniform = uni; yax.x0 = Y[0]; yax.ih = 1 / hDel[0];

    // LU factors of (k_x^2 - Lap_y) on the nodes 1..ny-1 (Dirichlet at the walls), per k_x: the bound field
    const std::vector<double> kx = kxd.host();
    std::vector<double> hcp(size_t(nk), 0.0), hid(size_t(nk), 0.0);
    for (int i = 0; i < nx; ++i) {
        double cprev = 0;
        for (int j = 1; j < ny; ++j) {
            const double lo = j > 1 ? -hiA[j] * hiD[j - 1] : 0, up = j < ny - 1 ? -hiA[j] * hiD[j] : 0;
            const double den = kx[i] * kx[i] + hiA[j] * (hiD[j - 1] + hiD[j]) - lo * cprev;
            hid[size_t(j) * nx + i] = 1 / den;
            hcp[size_t(j) * nx + i] = cprev = up / den;
        }
    }
    pcp.upload(hcp); pid.upload(hid);

    // eigenvectors of -Lap_y on the nodes: T v = mu A v with T symmetric; S = A^-1/2 T A^-1/2 is a symmetric
    // tridiagonal matrix with eigenvectors u, v = A^-1/2 u. Slot s = m + 1 holds the eigenvalue m (slot 0 unused).
    if (cfg.self_fields) {
        const int M = ny - 1;
        std::vector<double> d(M), e(M, 0.0), z;
        for (int j = 1; j < ny; ++j) d[j - 1] = hiA[j] * (hiD[j - 1] + hiD[j]);
        for (int j = 1; j < ny - 1; ++j) e[j - 1] = -hiD[j] / std::sqrt(hA[j] * hA[j + 1]);
        if (!tridiagonal_eigen(d, e, z)) throw std::runtime_error("eigenvectors of the y Laplacian: no convergence");
        std::vector<double> hf(size_t(ny) * ny, 0.0), hb(size_t(ny) * ny, 0.0), hl(ny, 0.0);
        for (int m = 0; m < M; ++m) {
            hl[m + 1] = d[m];
            for (int j = 1; j < ny; ++j) {
                const double u = z[size_t(j - 1) * M + m], sq = std::sqrt(hA[j]);
                hf[size_t(m + 1) * ny + j] = u * sq;          // coefficient = v^T A rho
                hb[size_t(j) * ny + m + 1] = u / sq;          // value = sum v coefficient
            }
        }
        evf.upload(hf); evb.upload(hb); evl.upload(hl);
    }
}

// zeta: slices zeta_n, half-cell m between slices m and m+1 with length h_m. hcg[m+2] = h_m for
// m = -2..N, continued with the end cells. The current of slice n is the linear interpolation of the two
// adjacent half-cells (as in GEM-PIC-RZ): (h_n j_{n-1/2} + h_{n-1} j_{n+1/2}) / (h_{n-1} + h_n).
void Simulation::setup_zeta() {
    const std::vector<double>& Z = cfg.zeta;
    std::vector<double> hh(N - 1);
    for (int m = 0; m < N - 1; ++m) hh[m] = Z[m + 1] - Z[m];
    bool uni = true;
    for (int m = 0; m < N - 1; ++m) uni = uni && hh[m] == hh[0];
    if (uni) for (double& v : hh) v = cfg.Lwin / (N - 1);   // exactly the uniform step
    std::vector<double> g(N + 3), lo(N), hi(N), ws(N);
    for (int m = -2; m <= N; ++m) g[m + 2] = hh[std::min(std::max(m, 0), N - 2)];
    for (int n = 0; n < N; ++n) {
        const double hm = g[n - 1 + 2], hp = g[n + 2];
        lo[n] = hp / (hm + hp); hi[n] = hm / (hm + hp);
        ws[n] = 0.5 * ((n > 0 ? hh[n - 1] : 0) + (n < N - 1 ? hh[n] : 0));
    }
    zs.upload(Z); hcg.upload(g); wlo.upload(lo); whi.upload(hi); wsl.upload(ws); cfg_wlo_ = lo;
    zax.X = zs.data(); zax.n = N; zax.uniform = uni; zax.x0 = 0; zax.ih = 1 / hh[0];
}

std::vector<Particle> Simulation::host_particles() const {
    std::vector<Particle> v(static_cast<size_t>(np));
    parts.to_host(v.data(), 0, size_t(np));
    return v;
}
void Simulation::set_particles(const std::vector<Particle>& v) {
    np = long(v.size());
    if (parts.size() < v.size() || parts.size() > 2 * v.size() + 1024) parts.resize(v.size() + v.size() / 4 + 64);
    parts.from_host(v.data(), 0, v.size());
}

// All ranks draw the same bunch; each keeps the particles of its half-cells.
void Simulation::load_bunch() {
    using cplx = std::complex<double>;
    const cplx I(0, 1);
    std::mt19937_64 rng(cfg.seed);
    std::normal_distribution<double> G(0, 1);
    const double Ne = cfg.charge / e_SI;
    W_macro_ne = cfg.nmacro > 0 ? Ne / cfg.nmacro : 0;
    const double W = W_macro_ne / (cfg.nc * std::pow(cfg.Lu, 3));     // code weight
    const double p0 = std::sqrt(cfg.gamma * cfg.gamma - 1);
    const int M = cfg.noise == "fawley" ? cfg.beamlet : 1;
    const long nb = cfg.nmacro / M;
    const int hlo = comm.rank == 0 ? 0 : a + 1, hhi = comm.rank == comm.size - 1 ? N - 2 : b;
    const double ylo = yf ? -1e300 : cfg.ynode[1], yhi = yf ? 1e300 : cfg.ynode[cfg.ny - 1];
    const double yoff = yf ? 0.5 * cfg.Ly : 0;        // Fourier y: box coordinate, physical y = p.y + Y0
    Axis zh = zax; zh.X = cfg.zeta.data();             // host copy of the axis
    std::vector<Particle> mine;
    for (long ib = 0; ib < nb; ++ib) {
        const double x = cfg.sigx * G(rng), y = cfg.sigy * G(rng), z = cfg.sigz * G(rng);
        const double px = cfg.sigx > 0 ? cfg.emitx / cfg.sigx * G(rng) : 0;
        const double py = cfg.sigy > 0 ? cfg.emity / cfg.sigy * G(rng) : 0;
        const double pz = p0 * (1 + cfg.espread * G(rng));
        std::vector<double> dz(M, 0.0);
        if (M > 1) {
            // beamlet: M particles spread evenly over noise_wavelength, displaced so that the bunching at the
            // harmonics h < M/2 equals that of the real electrons (Fawley, PRST-AB 5, 070701 (2002))
            const double Neb = Ne / nb, lam = cfg.noise_lambda;
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
            p.zeta = cfg.zeta_center + zz / vz;
            p.px = px; p.py = py; p.pz = pz; p.w = W; p.alive = 1;
            p.q = q; p.qm = q; p.sp = SP_BUNCH; p.z = 0;
            int hc; double f;
            zh.locate(p.zeta, hc, f);
            if (p.zeta < 0 || hc > N - 2 || p.y < ylo || p.y > yhi) { if (comm.rank == 0) ++lost_total; continue; }
            if (hc >= hlo && hc <= hhi) mine.push_back(p);
        }
    }
    set_particles(mine);
    X0 = -0.5 * cfg.Lx; Y0 = -yoff;
}

// ---------------------------------------------------------------- time step
void Simulation::init() {
    load_bunch();
    s = 0; istep = 0;
    if (cfg.self_fields) {
        currents();
        comoving_field();
        add_grid_lasers();
        march(0.0);                 // E_z, B_z, L of the co-moving R; the sponge acts as in every step
        sponge_and_real_fields();
        const long n = long(ns + 1) * nk;
        cpx* a0 = j0x.data(); cpx* b0 = j0y.data(); const cpx* a1 = jkx.data(); const cpx* b1 = jky.data();
        par_for(n, [=] HD(long i) { a0[i] = a1[i]; b0[i] = b1[i]; });
        device_sync();
    }
    window_shift_post();
    comm.wait_sends();
}

void Simulation::step() {
    window_shift_start();
    push(s, s + cfg.dz, 0, np);
    exchange_particles(s + cfg.dz);
    if (cfg.plasma_on) collect_restreaming();
    s += cfg.dz; ++istep;
    if (cfg.self_fields && cfg.plasma_on) {
        currents();
        march_plasma(s - cfg.dz, s);
        sponge_and_real_fields();
        const long n = long(ns + 1) * nk;
        cpx* a0 = j0x.data(); cpx* b0 = j0y.data(); const cpx* a1 = jkx.data(); const cpx* b1 = jky.data();
        par_for(n, [=] HD(long i) { a0[i] = a1[i]; b0[i] = b1[i]; });
        device_sync();
    } else if (cfg.self_fields) solve_fields(0.5 * cfg.dz);
    window_shift_post();
    comm.wait_sends();
}

// The transverse window follows the bunch centroid in whole cells (in x; in y only with Fourier y). The
// centroid of step k is reduced over the ranks without blocking and used at step k + window_lag, so the ranks
// need not wait for each other.
void Simulation::window_shift_post() {
    if (!cfg.move_x && !(yf && cfg.move_y)) return;
    Buf<double> acc(3);
    double* A = acc.data();
    const Particle* P = parts.data();
    const double x0 = X0, y0 = Y0;
    par_for(np, [=] HD(long i) {
        if (P[i].alive != 1 || P[i].sp != SP_BUNCH) return;
        atomic_add(&A[0], P[i].w * (P[i].x + x0)); atomic_add(&A[1], P[i].w * (P[i].y + y0)); atomic_add(&A[2], P[i].w);
    });
    device_sync();
    cent_.emplace_back(0, acc.host());
    cent_.back().first = comm.iallreduce_sum(cent_.back().second.data(), 3);
}
void Simulation::window_shift_start() {
    if (!cfg.move_x && !(yf && cfg.move_y)) return;
    if (long(cent_.size()) < cfg.window_lag) return;       // centroid of step istep + 1 - lag not yet due
    comm.wait(cent_.front().first);
    const std::vector<double> c = cent_.front().second;
    cent_.pop_front();
    if (c[2] <= 0) return;
    const int nxs = cfg.move_x ? int(std::lround((c[0] / c[2] - (X0 + 0.5 * cfg.Lx)) / dx)) : 0;
    const int nys = yf && cfg.move_y ? int(std::lround((c[1] / c[2] - (Y0 + 0.5 * cfg.Ly)) / dy)) : 0;
    if (std::abs(nxs) >= 2 || std::abs(nys) >= 2) shift_window(nxs, nys);
}

// A shift is an exact phase factor in Fourier space and an index roll in real space.
void Simulation::shift_window(int nxs, int nys) {
    X0 += nxs * dx; Y0 += nys * dy;
    const double sx = nxs * dx, sy = nys * dy;
    Particle* P = parts.data();
    par_for(np, [=] HD(long i) { P[i].x -= sx; P[i].y -= sy; });
    for (auto& p : joining_) { p.x -= sx; p.y -= sy; }
    if (cfg.self_fields) {
        const long nkk = nk;
        const int nx = cfg.nx, ny = cfg.ny;
        const double* kx = kxs.data(); const double* ky = yf ? kys.data() : nullptr;
        auto rot = [&](Buf<cpx>& A) {
            cpx* p = A.data();
            par_for(long(A.size()), [=] HD(long idx) {
                const long i = idx % nkk;
                const double ph = kx[i % nx] * sx + (nys ? ky[i / nx] * sy : 0.0);
                p[idx] = p[idx] * C(::cos(ph), ::sin(ph));
            });
        };
        rot(Rx); rot(Ry); rot(Ez); rot(Bz); rot(j0x); rot(j0y);
        if (cfg.plasma_on) { rot(js0x); rot(js0y); rot(js0z); }
        Buf<float> tmp(F.size());
        float* t = tmp.data(); const float* f = F.data();
        par_for(long(F.size()), [=] HD(long idx) {
            const long i = idx % nkk, base = idx - i;
            const int ixx = int(i % nx), j = int(i / nx);
            const int sxx = ((ixx + nxs) % nx + nx) % nx, syy = ((j + nys) % ny + ny) % ny;
            t[idx] = f[base + long(syy) * nx + sxx];
        });
        device_sync();
        F = std::move(tmp);
    }
    device_sync();
}
