// Plasma: streaming electrons, trapping, grid lasers, the march slice by slice (single rank).
#include "sim_internal.hpp"
#include "sim_kernels.hpp"

// ================================================================ plasma (streaming particles, as in GEM-PIC-RZ)
// Per step s0 -> s1 with plasma (single rank):
//  1. push the beam list (bunch + trapped electrons) to s1 in the fields of level s0; particles slower than
//     vz_min leave it and stream in this march from their (z, zeta)
//  2. deposit the beam current of level s1 (all half-cells), spectra
//  3. load fresh plasma at the front of the window; march from the front to the rear. Per half-cell n, in
//     `iterations` passes: push the streaming particles from zeta_{n+1} to zeta_n (Boris, dt = h/(1 - v_z)) in
//     the fields of slice n+1 (final, level s1) and slice n (level s0 in the first pass, the last solution
//     after that), deposit their current W = q w dt/Delta, smooth it (plasma.smooth_length), solve slice n.
//     The current of half-cell n-1 needed by R at slice n is predicted as beam(s1) + streaming(s0); once
//     half-cell n is final, R at slice n+1 is corrected for the difference.
//  4. streaming particles that reach z = s1 + (trap_advance - 1) dz are trapped: they join the beam list if
//     v_z >= vz_beam, otherwise they stream again in the next step from their zeta
//  5. sponge, real fields
namespace {
HD inline void boris(double& ux, double& uy, double& uz, const double E[3], const double B[3], double qm, double dt) {
    const double qh = 0.5 * qm * dt;
    const double umx = ux + qh * E[0], umy = uy + qh * E[1], umz = uz + qh * E[2];
    const double gm = ::sqrt(1.0 + umx * umx + umy * umy + umz * umz);
    const double tx = qh * B[0] / gm, ty = qh * B[1] / gm, tz = qh * B[2] / gm;
    const double t2 = tx * tx + ty * ty + tz * tz;
    const double upx = umx + (umy * tz - umz * ty), upy = umy + (umz * tx - umx * tz), upz = umz + (umx * ty - umy * tx);
    const double sf = 2.0 / (1.0 + t2);
    const double vx = umx + sf * (upy * tz - upz * ty), vy = umy + sf * (upz * tx - upx * tz), vz = umz + sf * (upx * ty - upy * tx);
    ux = vx + qh * E[0]; uy = vy + qh * E[1]; uz = vz + qh * E[2];
}

struct StreamCtx {
    PushCtx pc;                 // fields (gather), laser, grids
    const double* zs;           // slice positions
    double* J;                  // real current plane of the half-cell: jx, jy, jz (3 nk)
    const double *ia, *id;      // 1/A_j, 1/Delta_j (Fourier y: 1/dy)
    double Delta, s1, sTrap, vz_beam;
    int n;
};

// TSC deposit of q w (dt / Delta) v / (h dx dy) at (x, y) into the half-cell plane (box coordinates)
HD inline void stream_deposit(const StreamCtx& c, double x, double y, double f, double vx, double vy, double vz) {
    const PushCtx& P = c.pc;
    int ix_[3], jn[3], jh[3]; double wx[3], wn[3], wh[3];
    tsc(x, P.dx, P.nx, ix_, wx);
    const long nk = P.nk;
    if (P.yf) {
        tsc(y, P.dy, P.ny, jn, wn);
        for (int v = 0; v < 3; ++v)
            for (int u = 0; u < 3; ++u) {
                const long k = long(jn[v]) * P.nx + ix_[u];
                const double w = f * wx[u] * wn[v] * c.ia[0];
                atomic_add(&c.J[k], w * vx); atomic_add(&c.J[nk + k], w * vy); atomic_add(&c.J[2 * nk + k], w * vz);
                atomic_add(&c.J[3 * nk + k], w);
            }
        return;
    }
    tsc_y(P.yax, y, jn, wn, jh, wh);
    for (int v = 0; v < 3; ++v) {
        const bool nod = jn[v] >= 1 && jn[v] <= P.ny - 1, hal = jh[v] >= 0 && jh[v] <= P.ny - 1;
        for (int u = 0; u < 3; ++u) {
            if (nod) {
                const long k = long(jn[v]) * P.nx + ix_[u];
                const double w = f * wx[u] * wn[v] * c.ia[jn[v]];
                atomic_add(&c.J[k], w * vx); atomic_add(&c.J[2 * nk + k], w * vz); atomic_add(&c.J[3 * nk + k], w);
            }
            if (hal) {
                const long k = long(jh[v]) * P.nx + ix_[u];
                atomic_add(&c.J[nk + k], f * wx[u] * wh[v] * c.id[jh[v]] * vy);
            }
        }
    }
}

// one streaming particle through half-cell n (from its zeta, normally zeta_{n+1}, to zeta_n)
HD inline void stream_one(const StreamCtx& c, Particle& p) {
    const int n = c.n;
    const double zn = c.zs[n], zn1 = c.zs[n + 1], h = zn1 - zn;
    if (p.alive == 6) {             // waiting: joins the march inside the half-cell that holds its zeta
        if (p.zeta > zn) p.alive = 1;
        else return;
    }
    if (p.alive != 1) return;
    const double hl = p.zeta >= zn1 - 1e-9 * h ? h : (p.zeta - zn > 0 ? p.zeta - zn : 0.0);
    const PushCtx& P = c.pc;
    const double g0 = ::sqrt(1 + p.px * p.px + p.py * p.py + p.pz * p.pz);
    const double omv0 = (1 + p.px * p.px + p.py * p.py) / (g0 * (g0 + p.pz));    // 1 - v_z, no cancellation
    const double dt0 = hl / omv0;
    const double xm = p.x + 0.5 * dt0 * p.px / g0, ym = p.y + 0.5 * dt0 * p.py / g0;
    const double zm = p.z + 0.5 * dt0 * p.pz / g0, zetam = p.zeta - 0.5 * hl;
    double E[3], B[3], Ee[3], Be[3];
    gather(P, ::fmod(::fmod(xm, P.Lx) + P.Lx, P.Lx), P.yf ? ::fmod(::fmod(ym, P.Ly) + P.Ly, P.Ly) : ym, zetam, E, B);
    P.laser.field(xm + P.X0, zm, zm - (zetam - P.zc), Ee, Be);
    for (int k = 0; k < 3; ++k) { E[k] += Ee[k]; B[k] += Be[k]; }
    double ux = p.px, uy = p.py, uz = p.pz;
    boris(ux, uy, uz, E, B, p.qm, dt0);
    double g1 = ::sqrt(1 + ux * ux + uy * uy + uz * uz);
    double dt = 2 * hl / (omv0 + (1 + ux * ux + uy * uy) / (g1 * (g1 + uz)));    // time for |dzeta| = hl
    ux = p.px; uy = p.py; uz = p.pz;
    boris(ux, uy, uz, E, B, p.qm, dt);
    g1 = ::sqrt(1 + ux * ux + uy * uy + uz * uz);
    double vx = 0.5 * (p.px / g0 + ux / g1), vy = 0.5 * (p.py / g0 + uy / g1), vz = 0.5 * (p.pz / g0 + uz / g1);
    double zeta_end = zn;
    bool trapped = false;
    if (p.z + vz * dt >= c.sTrap) {    // advanced trap_advance steps inside this half-cell: trapped
        const double dtp = (c.sTrap - p.z) / (vz > 1e-3 ? vz : 1e-3) > 0 ? (c.sTrap - p.z) / (vz > 1e-3 ? vz : 1e-3) : 0.0;
        ux = p.px; uy = p.py; uz = p.pz;
        boris(ux, uy, uz, E, B, p.qm, dtp);
        g1 = ::sqrt(1 + ux * ux + uy * uy + uz * uz);
        vx = 0.5 * (p.px / g0 + ux / g1); vy = 0.5 * (p.py / g0 + uy / g1); vz = 0.5 * (p.pz / g0 + uz / g1);
        dt = dtp;
        zeta_end = p.zeta - (1 - vz) * dtp;
        trapped = true;
    }
    const double xo = p.x, yo = p.y;
    p.x += vx * dt; p.y += vy * dt;
    p.z = trapped ? c.s1 : p.z + vz * dt;
    p.px = ux; p.py = uy; p.pz = uz;
    p.zeta = zeta_end;
    // streaming current: q w dt / Delta per unit time, into half-cell n
    const double f = p.q * p.w * dt / (c.Delta * h * P.dx);
    stream_deposit(c, 0.5 * (xo + p.x), 0.5 * (yo + p.y), f, vx, vy, vz);
    if (!P.yf && (p.y < P.ylo || p.y > P.yhi)) { p.alive = 0; return; }
    if (trapped) {
        p.alive = 5;
        const double vze = uz / g1;
        if (vze >= c.vz_beam)        // joins the beam list: its beam current of level s1 from now on
            stream_deposit(c, p.x, p.y, p.q * p.w / (h * P.dx * vze), ux / g1, uy / g1, vze);
    }
}
}  // namespace

void Simulation::setup_plasma() {
    if (!cfg.plasma_on) return;
    if (comm.size > 1) throw std::runtime_error("plasma: MPI along zeta is not available with plasma yet (run on one rank)");
    const size_t pl = size_t(nk);
    for (Buf<cpx>* p : {&js0x, &js0y, &js0z}) p->resize((ns + 1) * pl);
    jsr.resize(4 * pl);
    rho_lines.resize(size_t(N) * (cfg.nx + cfg.ny));
    bool ne = false;                             // openPMD n_e: the charge density of every half-cell
    for (const auto& g : cfg.groups) ne = ne || (g.openpmd && (g.fields >> 6 & 1));
    if (ne) rho_full.resize(size_t(N - 1) * pl);
    jsn.resize(3 * pl); jsin.resize(3 * pl); bk.resize(4 * pl); sp3.resize(3 * pl);
    jpred.resize(3 * pl); jtmp.resize(3 * pl); jsi0.resize(3 * pl); dlt.resize(3 * pl);
    if (!yf && scp.size() == 0) {
        for (Buf<cpx>* p : {&sRx, &sRy, &sLx, &sLy, &srhs}) p->resize(pl);
        scp.resize(pl);
    }
}

// fresh plasma at the front of the window (a regular lattice of ppc_x x ppc_y particles per cell), and the
// particles that join the march at their zeta
void Simulation::load_plasma(double s0, double s1, const std::vector<Particle>& joining) {
    std::vector<Particle> v;
    const double zmid = 0.5 * (s0 + s1), Delta = s1 - s0;
    const int nx = cfg.nx, ny = cfg.ny, px = cfg.ppc_x, py = cfg.ppc_y;
    const double zf = cfg.zeta[N - 1];
    for (int j = 0; j < ny; ++j) {
        double y0, hy;
        if (yf) { y0 = j * dy; hy = dy; }
        else {
            if (j == 0 || j == ny - 1) continue;          // no plasma in the cells at the walls
            y0 = cfg.ynode[j]; hy = cfg.ynode[j + 1] - cfg.ynode[j];
        }
        for (int i = 0; i < nx; ++i)
            for (int b2 = 0; b2 < py; ++b2)
                for (int a2 = 0; a2 < px; ++a2) {
                    Particle p{};
                    p.x = (i + (a2 + 0.5) / px) * dx;
                    p.y = y0 + (b2 + 0.5) / py * hy;
                    const double n = cfg.plasma_density(p.x + X0, p.y + Y0, zmid);
                    if (n <= 0) continue;
                    p.w = n * dx * hy * Delta / (px * py);
                    p.q = -1; p.qm = -1; p.sp = SP_PLASMA; p.alive = 1;
                    p.zeta = zf; p.z = s0;
                    v.push_back(p);
                }
    }
    fresh_last = long(v.size());
    for (Particle p : joining) { p.alive = 6; v.push_back(p); }
    nstrm = long(v.size());
    if (strm.size() < v.size()) { strm.resize(v.size() + v.size() / 8 + 64); strm_bk.resize(strm.size()); }
    strm.from_host(v.data(), 0, v.size());
}

// spectra of the streaming current of one half-cell (jsr) into jsn: transverse filters as for the beam, the
// k = 0 part kept, and the smoothing (1 - a^2 lap_perp)^-1 of plasma.smooth_length
void Simulation::stream_spectra() {
    const long nkk = nk;
    const int nx = cfg.nx, ny = cfg.ny;
    const double* J = jsr.data();
    cpx* pk = sp3.data();                       // plane 0: jx + i jy, plane 1: jz
    par_for(nk, [=] HD(long i) { pk[i] = C(J[i], J[nkk + i]); pk[nkk + i] = C(J[2 * nkk + i], 0); });
    fft_->run(pk, 2, false);
    cpx* ox = jsn.data(); cpx* oy = ox + nkk; cpx* oz = ox + 2 * nkk;
    const double* fl = filt.data();
    const double* kxs_ = kxs.data();
    const double a2 = cfg.smooth_length * cfg.smooth_length;
    if (yf) {
        const double* fly = filty.data(); const double* kys_ = kys.data();
        par_for(nk, [=] HD(long i) {
            const int ixx = int(i % nx), j = int(i / nx);
            const long m = long((ny - j) % ny) * nx + (nx - ixx) % nx;
            const cpx z = pk[i], zm = conj(pk[m]);
            const double k2 = kxs_[ixx] * kxs_[ixx] + kys_[j] * kys_[j];
            const double f = fl[ixx] * fly[j] / (1 + a2 * k2);
            ox[i] = f * (0.5 * (z + zm)); oy[i] = f * (0.5 * C((z - zm).im, -(z - zm).re)); oz[i] = f * pk[nkk + i];
        });
        device_sync();
        return;
    }
    par_for(nk, [=] HD(long i) {
        const int ixx = int(i % nx);
        const long m = i - ixx + (nx - ixx) % nx;
        const cpx z = pk[i], zm = conj(pk[m]);
        ox[i] = 0.5 * (z + zm); oy[i] = 0.5 * C((z - zm).im, -(z - zm).re); oz[i] = pk[nkk + i];
    });
    const double* ia = iA.data(); const double* id = iD.data();
    const bool kf = cfg.k_filter;
    cpx* rhs = srhs.data(); double* cpv = scp.data();
    par_for(nx, [=] HD(long line) {
        const int ixx = int(line);
        const long o = ixx;
        const double fx = fl[ixx];
        if (kf) {                                         // (1,2,1)/4 on the cell charges, as for the beam
            cpx* const nodal[2] = {ox, oz};
            for (int t = 0; t < 2; ++t) {
                cpx* A = nodal[t];
                cpx qm = C(0), q0 = A[o + nx] / ia[1];
                for (int j = 1; j < ny; ++j) {
                    const cpx qp = j + 1 < ny ? A[o + long(j + 1) * nx] / ia[j + 1] : C(0);
                    A[o + long(j) * nx] = (0.25 * qm + 0.5 * q0 + 0.25 * qp) * ia[j];
                    qm = q0; q0 = qp;
                }
            }
            cpx qm = oy[o] / id[0], q0 = qm;
            for (int j = 0; j < ny; ++j) {
                const cpx qp = j + 1 < ny ? oy[o + long(j + 1) * nx] / id[j + 1] : q0;
                oy[o + long(j) * nx] = (0.25 * qm + 0.5 * q0 + 0.25 * qp) * id[j];
                qm = q0; q0 = qp;
            }
        }
        ox[o] = C(0); oz[o] = C(0);
        for (int j = 0; j < ny; ++j) {
            const long k = o + long(j) * nx;
            ox[k] = fx * ox[k]; oy[k] = fx * oy[k]; oz[k] = fx * oz[k];
        }
        if (a2 <= 0) return;
        const double kx2 = kxs_[ixx] * kxs_[ixx];
        // nodes (j_x, j_z): (1 + a^2 (k_x^2 - b- b+)) f = g, f = 0 at the walls (as E_z in the march)
        cpx* const nodal[2] = {ox, oz};
        for (int t = 0; t < 2; ++t) {
            cpx* A = nodal[t];
            double cp = 0; cpx dp = C(0);
            for (int j = 1; j < ny; ++j) {
                const double lo = j > 1 ? -a2 * ia[j] * id[j - 1] : 0, up = j < ny - 1 ? -a2 * ia[j] * id[j] : 0;
                const double den = 1 + a2 * kx2 + a2 * ia[j] * (id[j - 1] + id[j]) - lo * cp;
                dp = (A[o + long(j) * nx] - lo * dp) / den;
                cp = up / den;
                rhs[o + long(j) * nx] = dp; cpv[o + long(j) * nx] = cp;
            }
            cpx next = C(0);
            for (int j = ny - 1; j >= 1; --j) { next = rhs[o + long(j) * nx] - cpv[o + long(j) * nx] * next; A[o + long(j) * nx] = next; }
        }
        // half-nodes (j_y): (1 + a^2 (k_x^2 - b+ b-)) f = g (as B_z in the march)
        double cp = 0; cpx dp = C(0);
        for (int j = 0; j < ny; ++j) {
            const double al = j > 0 ? ia[j] : 0, au = j < ny - 1 ? ia[j + 1] : 0;
            const double lo = -a2 * id[j] * al, up = -a2 * id[j] * au;
            const double den = 1 + a2 * kx2 + a2 * id[j] * (al + au) - lo * cp;
            dp = (oy[o + long(j) * nx] - lo * dp) / den;
            cp = up / den;
            rhs[o + long(j) * nx] = dp; cpv[o + long(j) * nx] = cp;
        }
        cpx next = C(0);
        for (int j = ny - 1; j >= 0; --j) { next = rhs[o + long(j) * nx] - cpv[o + long(j) * nx] * next; oy[o + long(j) * nx] = next; }
    });
    device_sync();
}

// the march of one slice n (all modes or k_x lines)
void Simulation::march_slice(int n, double D) {
    MarchCtx c{Rx.data(), Ry.data(), Ez.data(), Bz.data(), Lx.data(), Ly.data(), bnd.data(), j0x.data(), j0y.data(),
               jkx.data(), jky.data(), jkz.data(), sRx.data(), sRy.data(), sLx.data(), sLy.data(), srhs.data(), scp.data(),
               kxd.data(), yf ? kyd.data() : nullptr, iD.data(), iA.data(), hcg.data(), wlo.data(), whi.data(), D, a, b, N,
               cfg.nx, cfg.ny, long(nk)};
    if (yf) par_for(nk, [=] HD(long i) { march_point(c, i, n); });
    else par_for(cfg.nx, [=] HD(long i) { march_line(c, int(i), n); });
    device_sync();
}

// E and B of slice n in real space (F plane fl = n - a), from its spectra
void Simulation::real_fields_slice(int n) {
    const long nkk = nk, l = n - a;
    const cpx *rx = Rx.data() + l * nkk, *ry = Ry.data() + l * nkk, *lx = Lx.data() + l * nkk, *ly = Ly.data() + l * nkk;
    const cpx *ez = Ez.data() + l * nkk, *bz = Bz.data() + l * nkk;
    cpx* t = sp3.data();
    par_for(nk, [=] HD(long i) {
        t[i] = C(rx[i].re - ry[i].im, rx[i].im + ry[i].re);
        t[nkk + i] = C(lx[i].re - ly[i].im, lx[i].im + ly[i].re);
        t[2 * nkk + i] = C(ez[i].re - bz[i].im, ez[i].im + bz[i].re);
    });
    fft_->run(t, 3, true);
    float* Fp = F.data() + l * nkk;
    const long fpl = long(ns + 2) * nkk;
    par_for(nk, [=] HD(long i) {
        const double r1 = t[i].re, r2 = t[i].im, l1 = t[nkk + i].re, l2 = t[nkk + i].im;
        Fp[0 * fpl + i] = float(0.5 * (r1 + l1)); Fp[1 * fpl + i] = float(0.5 * (r2 + l2));
        Fp[3 * fpl + i] = float(0.5 * (l2 - r2)); Fp[4 * fpl + i] = float(0.5 * (r1 - l1));
        Fp[2 * fpl + i] = float(t[2 * nkk + i].re); Fp[5 * fpl + i] = float(t[2 * nkk + i].im);
    });
    device_sync();
}

// the march of a step with streaming plasma (single rank: a = 0, b = N; half-cell m at plane m + 1).
// Prediction of the streaming current of half-cell n-1 (needed by R at slice n before the particles get there):
// its value of level s0 plus the change from s0 to s1, extrapolated linearly from the half-cells n and n+1
// (numerics.predictor = extrapolate, default) or without the change (previous, as in GEM-PIC-RZ). The change of
// a laser-driven current is the rotation of the carrier phase in s; without it the error of R at slice n is
// about (omega_p^2 dz / 2 omega)^2 of the laser field.
void Simulation::march_plasma(double s0, double s1) {
    const double Delta = s1 - s0, D = 0.5 * Delta;
    const long nkk = nk;
    load_plasma(s0, s1, joining_);
    joining_.clear();
    const bool extrap = cfg.predictor_extrapolate;
    std::vector<double> cen(N - 1);
    for (int m = 0; m < N - 1; ++m) cen[m] = 0.5 * (cfg.zeta[m] + cfg.zeta[m + 1]);
    auto copy3 = [&](cpx* dst, const cpx* x, const cpx* y, const cpx* z) {
        par_for(nk, [=] HD(long i) { dst[i] = x[i]; dst[nkk + i] = y[i]; dst[2 * nkk + i] = z[i]; });
    };
    // set the streaming part of jk(m) from `old` (the part it holds now) to `nw`; old <- nw
    auto replace = [&](int m, cpx* old, const cpx* nw) {
        cpx* jx = jkx.data() + (m + 1) * nkk; cpx* jy = jky.data() + (m + 1) * nkk; cpx* jz = jkz.data() + (m + 1) * nkk;
        par_for(nk, [=] HD(long i) {
            jx[i] = jx[i] + nw[i] - old[i]; jy[i] = jy[i] + nw[nkk + i] - old[nkk + i]; jz[i] = jz[i] + nw[2 * nkk + i] - old[2 * nkk + i];
            old[i] = nw[i]; old[nkk + i] = nw[nkk + i]; old[2 * nkk + i] = nw[2 * nkk + i];
        });
    };
    cpx* pin = jpred.data();         // streaming part now inside jk(n-1)
    cpx* tmp = jtmp.data();
    cpx* dl = dlt.data();            // change s0 -> s1 of half-cell n+1 (extrapolation)
    // front: half-cell N-2 predicted with its value of level s0
    jpred.zero();
    copy3(tmp, js0x.data() + (N - 1) * nkk, js0y.data() + (N - 1) * nkk, js0z.data() + (N - 1) * nkk);
    replace(N - 2, pin, tmp);
    march_slice(N - 1, D);
    real_fields_slice(N - 1);

    StreamCtx sc;
    {
        PushCtx& c = sc.pc;
        c.F = F.data(); c.a = a; c.ns = ns; c.N = N; c.nx = cfg.nx; c.ny = cfg.ny; c.nk = nk; c.self = 1;
        c.yf = yf; c.dx = dx; c.dy = dy; c.Lx = cfg.Lx; c.Ly = cfg.Ly; c.zax = zax; c.yax = yax; c.laser = laser; c.X0 = X0;
        c.zc = cfg.zeta_center; c.dz_push = cfg.dz_push; c.push_phase = cfg.push_phase; c.zhand = -1e300;
        c.zmax = cfg.zeta[N - 1]; c.s1 = s1; c.vz_min = 0; c.freeze = 0;
        c.ylo = yf ? -1e300 : cfg.ynode[1]; c.yhi = yf ? 1e300 : cfg.ynode[cfg.ny - 1];
    }
    sc.zs = zs.data(); sc.J = jsr.data(); sc.ia = iA.data(); sc.id = iD.data();
    sc.Delta = Delta; sc.s1 = s1; sc.sTrap = s1 + (cfg.trap_advance - 1) * Delta; sc.vz_beam = cfg.vz_beam;
    const int passes = cfg.iterations;
    bool have_dl = false;
    for (int n = N - 2; n >= 0; --n) {
        const long hp = (n + 1) * nkk;           // plane of half-cell n
        cpx* sin_ = jsin.data();                  // streaming part inside jk(n): the prediction so far
        par_for(3 * nkk, [=] HD(long i) { sin_[i] = pin[i]; });
        cpx* si0 = jsi0.data();                   // ... as used by the solve of slice n+1 (for the correction)
        par_for(3 * nkk, [=] HD(long i) { si0[i] = pin[i]; });
        if (n >= 1) {                             // start of the prediction of half-cell n-1: its level s0
            jpred.zero();
            copy3(tmp, js0x.data() + n * nkk, js0y.data() + n * nkk, js0z.data() + n * nkk);
            replace(n - 1, pin, tmp);
        }
        if (passes > 1) {
            int k = 0;
            for (Buf<cpx>* p : {&Rx, &Ry, &Ez, &Bz}) {
                cpx* dst = bk.data() + (k++) * nkk; const cpx* src = p->data() + n * nkk;
                par_for(nk, [=] HD(long i) { dst[i] = src[i]; });
            }
            Particle* d = strm_bk.data(); const Particle* sp_ = strm.data();
            par_for(nstrm, [=] HD(long i) { d[i] = sp_[i]; });
        }
        // weights of the linear extrapolation of the change to half-cell n-1
        const double wex = (n >= 1 && n + 1 <= N - 2 && have_dl) ? (cen[n] - cen[n - 1]) / (cen[n + 1] - cen[n]) : 0.0;
        for (int pass = 0; pass < passes; ++pass) {
            if (pass > 0) {
                int k = 0;
                for (Buf<cpx>* p : {&Rx, &Ry, &Ez, &Bz}) {
                    cpx* dst = p->data() + n * nkk; const cpx* src = bk.data() + (k++) * nkk;
                    par_for(nk, [=] HD(long i) { dst[i] = src[i]; });
                }
                Particle* d = strm.data(); const Particle* sp_ = strm_bk.data();
                par_for(nstrm, [=] HD(long i) { d[i] = sp_[i]; });
            }
            jsr.zero();
            sc.n = n;
            Particle* P = strm.data();
            const StreamCtx scc = sc;
            par_for(nstrm, [=] HD(long i) { stream_one(scc, P[i]); });
            device_sync();
            stream_spectra();
            replace(n, sin_, jsn.data());
            if (n >= 1 && extrap) {               // prediction of half-cell n-1 = s0 + change, extrapolated
                const cpx *u = jsn.data(), *sx = js0x.data() + hp, *sy = js0y.data() + hp, *sz = js0z.data() + hp;
                const cpx *px_ = js0x.data() + n * nkk, *py_ = js0y.data() + n * nkk, *pz_ = js0z.data() + n * nkk;
                const double w = wex;
                par_for(nk, [=] HD(long i) {
                    const cpx d0 = u[i] - sx[i], d1 = u[nkk + i] - sy[i], d2 = u[2 * nkk + i] - sz[i];
                    tmp[i] = px_[i] + d0 + w * (d0 - dl[i]);
                    tmp[nkk + i] = py_[i] + d1 + w * (d1 - dl[nkk + i]);
                    tmp[2 * nkk + i] = pz_[i] + d2 + w * (d2 - dl[2 * nkk + i]);
                });
                replace(n - 1, pin, tmp);
            }
            march_slice(n, D);
            real_fields_slice(n);
        }
        {   // electron density of half-cell n on the lines y = 0 and x = 0 (frames)
            const int nx = cfg.nx, ny = cfg.ny;
            const int i0 = ((int(std::lround(-X0 / dx)) % nx) + nx) % nx;
            const int j0 = yf ? ((int(std::lround(-Y0 / dy)) % ny) + ny) % ny : ny / 2;
            const double* J = jsr.data() + 3 * nkk; double* rl = rho_lines.data() + long(n) * (nx + ny);
            par_for(nx + ny, [=] HD(long k) { rl[k] = k < nx ? J[long(j0) * nx + k] : J[(k - nx) * nx + i0]; });
            if (rho_full.size()) {               // and in the whole half-cell (openPMD n_e)
                float* rf = rho_full.data() + long(n) * nkk;
                par_for(nk, [=] HD(long i) { rf[i] = float(J[i]); });
            }
        }
        {   // half-cell n is final: correct R at slice n+1 for the prediction it used, keep the change for the
            // extrapolation and the current as the prediction of the next step
            const double q = D * cfg_wlo_[n + 1];
            cpx* rx = Rx.data() + (n + 1) * nkk; cpx* ry = Ry.data() + (n + 1) * nkk;
            cpx *sx = js0x.data() + hp, *sy = js0y.data() + hp, *sz = js0z.data() + hp;
            const cpx* u = jsn.data();
            par_for(nk, [=] HD(long i) {
                rx[i] = rx[i] - q * (u[i] - si0[i]); ry[i] = ry[i] - q * (u[nkk + i] - si0[nkk + i]);
                dl[i] = u[i] - sx[i]; dl[nkk + i] = u[nkk + i] - sy[i]; dl[2 * nkk + i] = u[2 * nkk + i] - sz[i];
                sx[i] = u[i]; sy[i] = u[nkk + i]; sz[i] = u[2 * nkk + i];
            });
            device_sync();
            have_dl = true;
        }
    }
    // trapped particles: fast ones join the beam list, slow ones stream again in the next step
    std::vector<Particle> sv(static_cast<size_t>(nstrm));
    strm.to_host(sv.data(), 0, sv.size());
    std::vector<Particle> add;
    for (auto& p : sv) {
        if (p.alive != 5) continue;
        const double g = std::sqrt(1 + p.px * p.px + p.py * p.py + p.pz * p.pz);
        p.z = s1;
        if (p.pz / g >= cfg.vz_beam) { p.alive = 1; p.sp = SP_TRAPPED; add.push_back(p); ++trapped_total; }
        else { p.alive = 6; joining_.push_back(p); }
    }
    if (!add.empty()) {
        std::vector<Particle> all = host_particles(), keep;
        keep.reserve(all.size() + add.size());
        for (auto& p : all) if (p.alive == 1) keep.push_back(p);
        keep.insert(keep.end(), add.begin(), add.end());
        set_particles(keep);
    }
    // the sponge damps R - R_b with the bound field of the beam list only (the plasma is not a v = c source)
    cpx* jz = jkz.data(); const cpx* sz = js0z.data();
    par_for(long(ns + 1) * nk, [=] HD(long i) { jz[i] = jz[i] - sz[i]; });
    device_sync();
}

// the beam particles that became slower than vz_min stream in the current march
void Simulation::collect_restreaming() {
    std::vector<Particle> all = host_particles(), keep;
    bool any = false;
    for (auto& p : all) if (p.alive == 4) { any = true; break; }
    if (!any) return;
    for (auto& p : all) {
        if (p.alive == 4) { Particle q = p; q.alive = 6; joining_.push_back(q); ++restreamed_total; }
        else if (p.alive == 1) keep.push_back(p);
    }
    set_particles(keep);
}

// a Gaussian laser pulse on the grid: R = 2 E_perp of a paraxial beam (complex beam parameter), focus at z_focus
void Simulation::add_grid_lasers() {
    if (cfg.glasers.empty()) return;
    const int nx = cfg.nx, ny = cfg.ny;
    const size_t pl = size_t(nk);
    std::vector<cpx> pk(pl);
    for (const auto& Lz : cfg.glasers) {
        const double amp = Lz.a0 * Lz.kL;                       // field in m c omega0 / e
        const double zR = 0.5 * Lz.kL * Lz.waist * Lz.waist;
        const std::complex<double> qb(1.0, (0.0 - Lz.z_focus) / zR);
        const std::complex<double> rot(std::cos(Lz.pol_angle), std::sin(Lz.pol_angle));
        for (int n = a; n < b; ++n) {
            const double d = cfg.zeta[n] - Lz.zeta_c;
            const double env = std::exp(-2 * std::log(2.0) * d * d / (Lz.tau * Lz.tau));
            if (env < 1e-12) continue;
            const std::complex<double> car = std::exp(std::complex<double>(0, Lz.kL * d + Lz.cep));
            auto field = [&](double xp, double yp, double& ex, double& ey) {
                const std::complex<double> E = amp * env * std::exp(-(xp * xp + yp * yp) / (Lz.waist * Lz.waist) / qb) / qb * car;
                if (Lz.circular) { ex = E.real(); ey = Lz.helicity * E.imag(); }
                else { ex = E.real() * rot.real(); ey = E.real() * rot.imag(); }
            };
            for (int j = 0; j < ny; ++j)
                for (int i = 0; i < nx; ++i) {
                    const double xp = i * dx + X0;
                    double ex, ey, ex2, ey2;
                    if (yf) {
                        field(xp, j * dy + Y0, ex, ey);
                        pk[size_t(j) * nx + i] = C(2 * ex, 2 * ey);                         // R_x + i R_y
                    } else {                                                           // R_x on nodes, R_y on half-nodes
                        field(xp, cfg.ynode[j], ex, ey);
                        field(xp, 0.5 * (cfg.ynode[j] + cfg.ynode[j + 1]), ex2, ey2);
                        pk[size_t(j) * nx + i] = C(j == 0 ? 0.0 : 2 * ex, 2 * ey2);
                    }
                }
            sp3.from_host(pk.data(), 0, pl);
            cpx* t = sp3.data();
            fft_->run(t, 1, false);
            const long l = n - a, nkk = nk;
            cpx* rx = Rx.data() + l * nkk; cpx* ry = Ry.data() + l * nkk;
            const bool yfo = yf;
            par_for(nk, [=] HD(long idx) {
                const int ixx = int(idx % nx), j = int(idx / nx);
                const long m = yfo ? long((ny - j) % ny) * nx + (nx - ixx) % nx : idx - ixx + (nx - ixx) % nx;
                const cpx z = t[idx], zm = conj(t[m]);
                rx[idx] = rx[idx] + 0.5 * (z + zm); ry[idx] = ry[idx] + 0.5 * C((z - zm).im, -(z - zm).re);
            });
            device_sync();
        }
    }
}
