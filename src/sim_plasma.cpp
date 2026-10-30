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
#include "sim_stream.hpp"

void Simulation::setup_plasma() {
    if (!cfg.plasma_on) return;
    const size_t pl = size_t(nk);
    for (Buf<cpx>* p : {&js0x, &js0y, &js0z}) p->resize((ns + 1) * pl);
    jsr.resize(5 * pl);
    rho_lines.resize(size_t(N) * (cfg.nx + cfg.ny));
    if (cfg.mobile_ions()) rho_lines_i.resize(size_t(N) * (cfg.nx + cfg.ny));
    bool ne = false, ni = false;                 // openPMD n_e, n_i: the charge densities of every half-cell
    for (const auto& g : cfg.groups) { ne = ne || (g.openpmd && (g.fields >> 6 & 1)); ni = ni || (g.openpmd && (g.fields >> 7 & 1)); }
    if (ne) rho_full.resize(size_t(N - 1) * pl);
    if (ni && cfg.mobile_ions()) rho_full_i.resize(size_t(N - 1) * pl);
    jsn.resize(3 * pl); jsin.resize(3 * pl); bk.resize(4 * pl); sp3.resize(3 * pl);
    jpred.resize(3 * pl); jtmp.resize(3 * pl); jsi0.resize(3 * pl); dlt.resize(3 * pl);
    if (!yf && scp.size() == 0) {
        for (Buf<cpx>* p : {&sRx, &sRy, &sLx, &sLy, &srhs}) p->resize(pl);
        scp.resize(pl);
    }
}

// fresh plasma at the front of the window (a regular lattice of ppc_x x ppc_y particles per cell; the mobile ions
// of every ion block at the positions of the electrons: neutral pairs), and the particles that join the march at
// their zeta
void Simulation::load_plasma(double s0, double s1, const std::vector<Particle>& joining, const std::vector<Particle>* arriving) {
    std::vector<Particle> v;
    if (arriving) v = *arriving;                           // MPI, below the top rank: the particles from above
    long cap = 0;                                          // electrons that the ionization can free in this step
    const unsigned long long id0 = (unsigned long long)(istep + 1) << 32;
    const double zmid = 0.5 * (s0 + s1), Delta = s1 - s0;
    const int nx = cfg.nx, ny = cfg.ny, px = cfg.ppc_x, py = cfg.ppc_y;
    const double zf = cfg.zeta[N - 1];
    for (int j = 0; j < (arriving ? 0 : ny); ++j) {
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
                    if (cfg.free_electrons) { p.id = id0 + v.size(); v.push_back(p); }
                    for (size_t k = 0; k < cfg.ions.size(); ++k) {
                        const Config::IonSpecies& is = cfg.ions[k];
                        if (!is.loaded()) continue;
                        Particle q = p;
                        q.q = is.Z0; q.qm = is.Z0 / is.mass; q.w = is.atoms * p.w; q.sp = SP_ION + int(k);
                        q.id = id0 + v.size();
                        v.push_back(q);
                    }
                }
    }
    fresh_last = long(v.size());
    for (Particle p : joining) { p.alive = 6; v.push_back(p); }
    for (const auto& p : v)
        if (is_ion(p.sp) && cfg.ions[p.sp - SP_ION].ionization) cap += cfg.ions[p.sp - SP_ION].Zmax - std::lround(p.q);
    nstrm = long(v.size());
    born_cap = cap;
    const size_t need = v.size() + size_t(cap);
    if (strm.size() < need) { strm.resize(need + need / 8 + 64); strm_bk.resize(strm.size()); }
    if (cap > 0 && born.size() < size_t(cap)) born.resize(size_t(cap));
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
    const size_t pl = size_t(nk);
    const bool top = comm.rank == comm.size - 1, bottom = comm.rank == 0;
    const int nhi = top ? N - 2 : b - 1;               // the half-cells of this rank: nhi down to a
    auto hpl = [&](int m) { return long(m - a + 1) * nkk; };   // plane of half-cell m (jk, js0)
    auto spl = [&](int n) { return long(n - a) * nkk; };       // plane of slice n (R, E_z, B_z)
    std::vector<cpx> pk;                                // package from above: bnd (6 planes), prediction, change
    if (top) load_plasma(s0, s1, joining_, nullptr);
    else {
        const std::vector<Particle> arr = comm.recv_vec<Particle>(comm.rank + 1, TAG_PPART);
        pk.resize(12 * pl);
        comm.recv(comm.rank + 1, TAG_PDOWN, pk.data(), pk.size());
        std::vector<float> fb(6 * pl);
        comm.recv(comm.rank + 1, TAG_PDOWNF, fb.data(), fb.size());
        for (int c = 0; c < 6; ++c) F.from_host(&fb[c * pl], (size_t(c) * (ns + 2) + ns) * pl, pl);   // ghost slice b
        bnd.from_host(pk.data(), 0, 6 * pl);
        load_plasma(s0, s1, joining_, &arr);
    }
    joining_.clear();
    const bool extrap = cfg.predictor_extrapolate;
    std::vector<double> cen(N - 1);
    for (int m = 0; m < N - 1; ++m) cen[m] = 0.5 * (cfg.zeta[m] + cfg.zeta[m + 1]);
    auto copy3 = [&](cpx* dst, const cpx* x, const cpx* y, const cpx* z) {
        par_for(nk, [=] HD(long i) { dst[i] = x[i]; dst[nkk + i] = y[i]; dst[2 * nkk + i] = z[i]; });
    };
    // set the streaming part of jk(m) from `old` (the part it holds now) to `nw`; old <- nw
    auto replace = [&](int m, cpx* old, const cpx* nw) {
        cpx* jx = jkx.data() + hpl(m); cpx* jy = jky.data() + hpl(m); cpx* jz = jkz.data() + hpl(m);
        par_for(nk, [=] HD(long i) {
            jx[i] = jx[i] + nw[i] - old[i]; jy[i] = jy[i] + nw[nkk + i] - old[nkk + i]; jz[i] = jz[i] + nw[2 * nkk + i] - old[2 * nkk + i];
            old[i] = nw[i]; old[nkk + i] = nw[nkk + i]; old[2 * nkk + i] = nw[2 * nkk + i];
        });
    };
    cpx* pin = jpred.data();         // streaming part now inside jk(n-1)
    cpx* tmp = jtmp.data();
    cpx* dl = dlt.data();            // change s0 -> s1 of half-cell n+1 (extrapolation)
    bool have_dl = false;
    jpred.zero();
    if (top) {                       // front: half-cell N-2 predicted with its value of level s0
        copy3(tmp, js0x.data() + hpl(N - 2), js0y.data() + hpl(N - 2), js0z.data() + hpl(N - 2));
        replace(N - 2, pin, tmp);
        march_slice(N - 1, D);
        real_fields_slice(N - 1);
    } else {                         // slice b was solved above with this prediction of half-cell b-1
        dlt.from_host(&pk[9 * pl], 0, 3 * pl);
        jtmp.from_host(&pk[6 * pl], 0, 3 * pl);
        replace(b - 1, pin, tmp);
        have_dl = true;
    }

    StreamCtx sc;
    {
        PushCtx& c = sc.pc;
        c.F = F.data(); c.a = a; c.ns = ns; c.N = N; c.nx = cfg.nx; c.ny = cfg.ny; c.nk = nk; c.self = 1;
        c.yf = yf; c.dx = dx; c.dy = dy; c.Lx = cfg.Lx; c.Ly = cfg.Ly; c.zax = zax; c.yax = yax; c.laser = laser; c.X0 = X0;
        c.zc = cfg.zeta_center; c.dz_push = cfg.dz_push; c.push_phase = cfg.push_phase; c.zhand = -1e300;
        c.zmax = cfg.zeta[N - 1]; c.s1 = s1; c.vz_min = 0; c.freeze = 0; c.hmin = 1e300; c.Y0 = Y0; c.ext = external();
        c.ylo = yf ? -1e300 : cfg.ynode[1]; c.yhi = yf ? 1e300 : cfg.ynode[cfg.ny - 1];
    }
    sc.zs = zs.data(); sc.J = jsr.data(); sc.ia = iA.data(); sc.id = iD.data();
    sc.Delta = Delta; sc.s1 = s1; sc.sTrap = s1 + (cfg.trap_advance - 1) * Delta; sc.vz_beam = cfg.vz_beam;
    sc.ions = ion_dev.data(); sc.adk = adk_dev.data(); sc.born = born.data(); sc.bcnt = bcnt.data(); sc.bcap = 0; sc.key = 0;
    const bool ionz = born_cap > 0;
    const uint64_t key_step = ion_key_step(cfg.seed, istep);
    const int passes = cfg.iterations;
    for (int n = nhi; n >= a; --n) {
        const long hp = hpl(n);                   // plane of half-cell n
        cpx* sin_ = jsin.data();                  // streaming part inside jk(n): the prediction so far
        par_for(3 * nkk, [=] HD(long i) { sin_[i] = pin[i]; });
        cpx* si0 = jsi0.data();                   // ... as used by the solve of slice n+1 (for the correction)
        par_for(3 * nkk, [=] HD(long i) { si0[i] = pin[i]; });
        if (n >= 1) {                             // start of the prediction of half-cell n-1: its level s0
            jpred.zero();
            copy3(tmp, js0x.data() + hpl(n - 1), js0y.data() + hpl(n - 1), js0z.data() + hpl(n - 1));
            replace(n - 1, pin, tmp);
        }
        if (passes > 1) {
            int k = 0;
            for (Buf<cpx>* p : {&Rx, &Ry, &Ez, &Bz}) {
                cpx* dst = bk.data() + (k++) * nkk; const cpx* src = p->data() + spl(n);
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
                    cpx* dst = p->data() + spl(n); const cpx* src = bk.data() + (k++) * nkk;
                    par_for(nk, [=] HD(long i) { dst[i] = src[i]; });
                }
                Particle* d = strm.data(); const Particle* sp_ = strm_bk.data();
                par_for(nstrm, [=] HD(long i) { d[i] = sp_[i]; });
            }
            jsr.zero();
            sc.n = n;
            if (ionz) {                           // electrons freed in this pass (only the last pass counts)
                bcnt.zero();
                sc.key = ion_key_slice(key_step, n, pass);
                sc.bcap = int(std::min<long>(long(born.size()), long(strm.size()) - nstrm));
            }
            Particle* P = strm.data();
            const StreamCtx scc = sc;
            par_for(nstrm, [=] HD(long i) { stream_one(scc, P[i]); });
            device_sync();
            stream_spectra();
            replace(n, sin_, jsn.data());
            if (n >= 1 && extrap) {               // prediction of half-cell n-1 = s0 + change, extrapolated
                const cpx *u = jsn.data(), *sx = js0x.data() + hp, *sy = js0y.data() + hp, *sz = js0z.data() + hp;
                const cpx *px_ = js0x.data() + hpl(n - 1), *py_ = js0y.data() + hpl(n - 1), *pz_ = js0z.data() + hpl(n - 1);
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
        if (ionz) {                               // the freed electrons stream from zeta_n on
            int cnt = 0;
            bcnt.to_host(&cnt, 0, 1);
            if (cnt > sc.bcap) throw std::runtime_error("ionization: list of freed electrons overflowed");
            Particle* d = strm.data() + nstrm; const Particle* bp = born.data();
            par_for(cnt, [=] HD(long i) { d[i] = bp[i]; });
            nstrm += cnt;
        }
        {   // electron and ion density of half-cell n on the lines y = 0 and x = 0 (frames)
            const int nx = cfg.nx, ny = cfg.ny;
            const int i0 = ((int(std::lround(-X0 / dx)) % nx) + nx) % nx;
            const int j0 = yf ? ((int(std::lround(-Y0 / dy)) % ny) + ny) % ny : ny / 2;
            for (int sgn = 0; sgn < 2; ++sgn) {
                Buf<double>& lines = sgn ? rho_lines_i : rho_lines;
                Buf<float>& full = sgn ? rho_full_i : rho_full;
                const double* J = jsr.data() + (3 + sgn) * nkk;
                if (lines.size()) {
                    double* rl = lines.data() + long(n) * (nx + ny);
                    par_for(nx + ny, [=] HD(long k) { rl[k] = k < nx ? J[long(j0) * nx + k] : J[(k - nx) * nx + i0]; });
                }
                if (full.size()) {               // and in the whole half-cell (openPMD n_e, n_i)
                    float* rf = full.data() + long(n) * nkk;
                    par_for(nk, [=] HD(long i) { rf[i] = float(J[i]); });
                }
            }
        }
        {   // half-cell n is final: correct R at slice n+1 for the prediction it used, keep the change for the
            // extrapolation and the current as the prediction of the next step
            // (slice b belongs to the rank above: its correction and the current go up)
            const double q = D * cfg_wlo_[n + 1];
            const bool own = n + 1 < b;
            cpx* rx = own ? Rx.data() + spl(n + 1) : tmp; cpx* ry = own ? Ry.data() + spl(n + 1) : tmp + nkk;
            if (!own) jtmp.zero();
            cpx *sx = js0x.data() + hp, *sy = js0y.data() + hp, *sz = js0z.data() + hp;
            const cpx* u = jsn.data();
            par_for(nk, [=] HD(long i) {
                rx[i] = rx[i] - q * (u[i] - si0[i]); ry[i] = ry[i] - q * (u[nkk + i] - si0[nkk + i]);
                dl[i] = u[i] - sx[i]; dl[nkk + i] = u[nkk + i] - sy[i]; dl[2 * nkk + i] = u[2 * nkk + i] - sz[i];
                sx[i] = u[i]; sy[i] = u[nkk + i]; sz[i] = u[2 * nkk + i];
            });
            device_sync();
            have_dl = true;
            if (!own) {                           // package up: -correction of R_x, R_y at slice b; the current
                std::vector<cpx> up(5 * pl);
                jtmp.to_host(up.data(), 0, 2 * pl);
                jsn.to_host(&up[2 * pl], 0, 3 * pl);
                comm.isend(comm.rank + 1, TAG_PUP, up.data(), up.size());
            }
        }
    }
    std::vector<Particle> trap_below;             // electrons trapped in half-cell a: beam list of the rank below
    if (!bottom) {                                // package down: particles at zeta_a, slice a, prediction, change
        std::vector<Particle> sv(static_cast<size_t>(nstrm)), keep, down;
        strm.to_host(sv.data(), 0, sv.size());
        for (const auto& p : sv) (p.alive == 1 ? down : keep).push_back(p);
        comm.isend(comm.rank - 1, TAG_PPART, down.data(), down.size());
        std::vector<cpx> pd(12 * pl);
        int k = 0;
        for (Buf<cpx>* p : {&Rx, &Ry, &Ez, &Bz, &Lx, &Ly}) p->to_host(&pd[pl * k++], 0, pl);
        jpred.to_host(&pd[6 * pl], 0, 3 * pl);
        dlt.to_host(&pd[9 * pl], 0, 3 * pl);
        comm.isend(comm.rank - 1, TAG_PDOWN, pd.data(), pd.size());
        std::vector<float> fd(6 * pl);
        for (int c = 0; c < 6; ++c) F.to_host(&fd[c * pl], size_t(c) * (ns + 2) * pl, pl);
        comm.isend(comm.rank - 1, TAG_PDOWNF, fd.data(), fd.size());
        nstrm = long(keep.size());
        strm.from_host(keep.data(), 0, keep.size());
        // package from below: the correction of R at slice a and the final current of half-cell a-1
        std::vector<cpx> up(5 * pl);
        comm.recv(comm.rank - 1, TAG_PUP, up.data(), up.size());
        jtmp.from_host(&up[2 * pl], 0, 3 * pl);
        replace(a - 1, pin, tmp);                 // jk(a-1): beam + final streaming current
        cpx *gx = js0x.data(), *gy = js0y.data(), *gz = js0z.data();
        par_for(nk, [=] HD(long i) { gx[i] = tmp[i]; gy[i] = tmp[nkk + i]; gz[i] = tmp[2 * nkk + i]; });
        jtmp.from_host(up.data(), 0, 2 * pl);
        cpx* rx = Rx.data(); cpx* ry = Ry.data();
        par_for(nk, [=] HD(long i) { rx[i] = rx[i] + tmp[i]; ry[i] = ry[i] + tmp[nkk + i]; });
        device_sync();
    }
    // trapped particles: fast ones join the beam list, slow ones stream again in the next step
    std::vector<Particle> sv(static_cast<size_t>(nstrm));
    strm.to_host(sv.data(), 0, sv.size());
    std::vector<Particle> add;
    for (auto& h : ion_stats) h.clear();
    for (auto& p : sv) {
        if (is_ion(p.sp) && cfg.ions[p.sp - SP_ION].ionization) {   // charge states leaving the window, freed charge
            const auto& is = cfg.ions[p.sp - SP_ION];
            std::vector<double>& h = ion_stats[p.sp - SP_ION];
            if (h.empty()) h.assign(is.Zmax + 2, 0.0);
            const int Z = int(std::lround(p.q));
            if (p.alive == 1) h[Z] += p.w;
            h[is.Zmax + 1] += (Z - is.Z0) * p.w;
        }
        if (p.alive != 5) continue;
        const double g = std::sqrt(1 + p.px * p.px + p.py * p.py + p.pz * p.pz);
        p.z = s1;
        if (p.pz / g >= cfg.vz_beam) {
            p.alive = 1; p.sp = trapped_species(p.sp); ++trapped_total;
            (!bottom && p.zeta < cfg.zeta[a + 1] ? trap_below : add).push_back(p);   // beam half-cells [a+1, b]
        }
        else { p.alive = 6; joining_.push_back(p); }
    }
    if (!bottom) comm.isend(comm.rank - 1, TAG_PTRAP, trap_below.data(), trap_below.size());
    if (!top) for (const auto& p : comm.recv_vec<Particle>(comm.rank + 1, TAG_PTRAP)) add.push_back(p);
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

// the beam particles that became slower than vz_min stream in the current march (MPI: those in half-cell b of
// the beam list of this rank are marched by the rank above; collective with the neighbours)
void Simulation::collect_restreaming() {
    std::vector<Particle> all = host_particles(), keep, up;
    bool any = false;
    for (auto& p : all) if (p.alive == 4) { any = true; break; }
    const bool top = comm.rank == comm.size - 1;
    if (any) {
        for (auto& p : all) {
            if (p.alive == 4) {
                Particle q = p; q.alive = 6; ++restreamed_total;
                (!top && q.zeta >= cfg.zeta[b] ? up : joining_).push_back(q);
            } else if (p.alive == 1) keep.push_back(p);
        }
        set_particles(keep);
    }
    if (comm.size == 1) return;
    if (!top) comm.isend(comm.rank + 1, TAG_RESTR, up.data(), up.size());
    if (comm.rank > 0) for (const auto& p : comm.recv_vec<Particle>(comm.rank - 1, TAG_RESTR)) joining_.push_back(p);
}

// The densities of the half-cells of all ranks on rank 0 (collective; each rank marches the half-cells a ... b-1,
// the top rank up to N-2): the lines y = 0 and x = 0 [N][nx + ny], and the whole planes [N-1][nk].
std::vector<double> Simulation::halfcell_lines(const Buf<double>& lines) {
    if (comm.size == 1 || !lines.size()) return lines.host();
    const size_t L = size_t(cfg.nx + cfg.ny);
    const int nhi = comm.rank == comm.size - 1 ? N - 2 : b - 1;
    std::vector<double> mine(size_t(nhi - a + 1) * L);
    lines.to_host(mine.data(), size_t(a) * L, mine.size());
    std::vector<char> all = comm.gather_bytes(mine.data(), mine.size() * sizeof(double));
    std::vector<double> v(size_t(N) * L, 0.0);
    if (comm.rank == 0) std::memcpy(v.data(), all.data(), all.size());
    return v;
}
std::vector<float> Simulation::halfcell_planes(const Buf<float>& full) {
    if (comm.size == 1 || !full.size()) return full.host();
    const int nhi = comm.rank == comm.size - 1 ? N - 2 : b - 1;
    std::vector<float> mine(size_t(nhi - a + 1) * nk);
    full.to_host(mine.data(), size_t(a) * nk, mine.size());
    std::vector<char> all = comm.gather_bytes(mine.data(), mine.size() * sizeof(float));
    std::vector<float> v(size_t(N - 1) * nk, 0.0f);
    if (comm.rank == 0) std::memcpy(v.data(), all.data(), all.size());
    return v;
}
