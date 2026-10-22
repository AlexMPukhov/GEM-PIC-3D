// Fields: the march over the slices, the co-moving field of the bunch, sponge and real-space fields.
#include "sim_internal.hpp"
#include "sim_kernels.hpp"

// The march of this block: every mode (Fourier y) or k_x line (free-hand y) is an independent recurrence over
// the slices, from b-1 down to a. GPU: one thread per mode or line; CPU: slice by slice, OpenMP over them.
void Simulation::march(double D) {
    const size_t pl = size_t(nk);
    if (comm.rank < comm.size - 1) {
        std::vector<cpx> buf(6 * pl);
        comm.recv(comm.rank + 1, TAG_BND, buf.data(), buf.size());
        bnd.from_host(buf.data(), 0, buf.size());
    }
    MarchCtx c{Rx.data(), Ry.data(), Ez.data(), Bz.data(), Lx.data(), Ly.data(), bnd.data(), j0x.data(), j0y.data(),
               jkx.data(), jky.data(), jkz.data(), sRx.data(), sRy.data(), sLx.data(), sLy.data(), srhs.data(), scp.data(),
               kxd.data(), yf ? kyd.data() : nullptr, iD.data(), iA.data(), hcg.data(), wlo.data(), whi.data(), D, a, b, N,
               cfg.nx, cfg.ny, long(nk)};
    const int lo = a, hi = b;
    if (yf) {
#if GEMPIC_GPU
        par_for(nk, [=] HD(long i) { for (int n = hi - 1; n >= lo; --n) march_point(c, i, n); });
#else
        for (int n = hi - 1; n >= lo; --n) par_for(nk, [=](long i) { march_point(c, i, n); });
#endif
    } else {
#if GEMPIC_GPU
        par_for(cfg.nx, [=] HD(long i) { for (int n = hi - 1; n >= lo; --n) march_line(c, int(i), n); });
#else
        for (int n = hi - 1; n >= lo; --n) par_for(cfg.nx, [=](long i) { march_line(c, int(i), n); });
#endif
    }
    device_sync();
    if (comm.rank > 0) {           // slice a of level s1 is the boundary of the block below
        std::vector<cpx> buf(6 * pl);
        int k = 0;
        for (Buf<cpx>* p : {&Rx, &Ry, &Ez, &Bz, &Lx, &Ly}) p->to_host(&buf[pl * k++], 0, pl);
        comm.isend(comm.rank - 1, TAG_BND, buf.data(), buf.size());
    }
}

// Co-moving field of the bunch: -(lap_perp + gamma^-2 d_zeta^2) psi = rho, R = (1 + beta) E_perp,
// E_perp = -grad psi. In the eigenvectors of -Lap_y (eigenvalue mu) each (k_x, mu) is one tridiagonal (Thomas)
// solve along the free-hand zeta grid, pipelined over the ranks.
void Simulation::comoving_field() {
    double sums[2] = {0, 0};
    for (auto& p : host_particles())
        if (p.alive == 1) { sums[0] += p.w * p.pz / std::sqrt(1 + p.px * p.px + p.py * p.py + p.pz * p.pz); sums[1] += p.w; }
    comm.allreduce_sum(sums, 2);
    const double beta = sums[1] > 0 ? sums[0] / sums[1] : 1;
    const double gm2 = 1 - beta * beta;
    beta_b = beta; gm2_b = gm2;
    const size_t pl = size_t(nk);
    const long nkk = nk;
    const int nx = cfg.nx, ny = cfg.ny;
    // free-hand y: half-cell charges to the eigen-coefficients (slot s = eigenvalue s - 1; slot 0 stays 0).
    // Fourier y: the spectra themselves (slot = iy, eigenvalue k_y^2).
    const bool yfo = yf;
    Buf<cpx> E1(yf ? 0 : (ns + 1) * pl);
    if (!yf) {
        cpx* e1 = E1.data(); const cpx* jz = jkz.data(); const double* ef = evf.data();
        par_for(long(ns + 1) * nk, [=] HD(long idx) {
            const long i = idx % nkk, base = idx - i;
            const int s = int(i / nx), ixx = int(i % nx);
            cpx v = C(0);
            if (s > 0) for (int j = 1; j < ny; ++j) v = v + ef[long(s) * ny + j] * jz[base + long(j) * nx + ixx];
            e1[idx] = v;
        });
        device_sync();
    }
    // forward elimination: cp in S1, dp in S2; the last row of the block below arrives in bnd[0], bnd[1]
    if (comm.rank > 0) {
        std::vector<cpx> buf(2 * pl);
        comm.recv(comm.rank - 1, TAG_TRIF, buf.data(), buf.size());
        bnd.from_host(buf.data(), 0, buf.size());
    }
    cpx* cp = S1.data(); cpx* dp = S2.data();
    const cpx* e1 = yf ? jkz.data() : E1.data(); const cpx* bd = bnd.data();
    const double* kx = kxd.data(); const double* ev = evl.data();
    const double* hc = hcg.data(); const double* wl = wlo.data(); const double* wu = whi.data();
    const int a_ = a, ns_ = ns, first = comm.rank == 0;
    par_for(nk, [=] HD(long i) {
        const int s = int(i / nx), ixx = int(i % nx);
        cpx cprev = first ? C(0) : bd[i], dprev = first ? C(0) : bd[nkk + i];
        for (int l = 0; l < ns_; ++l) {
            const long o = l * nkk + i;
            const int n = a_ + l;
            const double hm = hc[n + 1], hp = hc[n + 2];
            const double lo = -gm2 * 2 / ((hm + hp) * hm), up = -gm2 * 2 / ((hm + hp) * hp);
            const double dg = kx[ixx] * kx[ixx] + ev[s] + gm2 * 2 / (hm * hp);
            if ((!yfo && s == 0) || dg == 0) { cp[o] = C(0); dp[o] = C(0); continue; }
            const cpx rhs = (1 / beta) * (wl[n] * e1[o] + wu[n] * e1[o + nkk]);
            if (n == 0) { cprev = C(0); dprev = C(0); }
            const cpx den = C(dg) - lo * cprev;
            cp[o] = C(up) / den; dp[o] = (rhs - lo * dprev) / den;
            cprev = cp[o]; dprev = dp[o];
        }
    });
    device_sync();
    if (comm.rank < comm.size - 1) {
        std::vector<cpx> buf(2 * pl);
        S1.to_host(buf.data(), (ns - 1) * pl, pl); S2.to_host(&buf[pl], (ns - 1) * pl, pl);
        comm.isend(comm.rank + 1, TAG_TRIF, buf.data(), buf.size());
    }
    // back substitution, psi overwrites dp; psi of slice b arrives from the block above
    std::vector<cpx> topv(pl, C(0));
    if (comm.rank < comm.size - 1) comm.recv(comm.rank + 1, TAG_TRIB, topv.data(), pl);
    bnd.from_host(topv.data(), 0, pl);
    par_for(nk, [=] HD(long i) {
        cpx next = bd[i];
        for (int l = ns_ - 1; l >= 0; --l) {
            next = dp[l * nkk + i] - cp[l * nkk + i] * next;
            dp[l * nkk + i] = next;
        }
    });
    device_sync();
    if (comm.rank > 0) {
        std::vector<cpx> buf(pl);
        S2.to_host(buf.data(), 0, pl);
        comm.isend(comm.rank - 1, TAG_TRIB, buf.data(), pl);
    }
    // back to the nodes (psi in S1), then R
    if (yf) {
        cpx* t = cp;
        par_for(long(ns) * nk, [=] HD(long idx) { t[idx] = dp[idx]; });
        device_sync();
    } else {
        const double* eb = evb.data();
        par_for(long(ns) * nk, [=] HD(long idx) {
            const long i = idx % nkk, base = idx - i;
            const int j = int(i / nx), ixx = int(i % nx);
            cpx v = C(0);
            if (j > 0) for (int s = 1; s < ny; ++s) v = v + eb[long(j) * ny + s] * dp[base + long(s) * nx + ixx];
            cp[idx] = v;
        });
        device_sync();
    }
    cpx* rx = Rx.data(); cpx* ry = Ry.data(); cpx* ez = Ez.data(); cpx* bz = Bz.data();
    const double* id = iD.data(); const double* ky = yf ? kyd.data() : nullptr;
    par_for(long(ns) * nk, [=] HD(long idx) {
        const long i = idx % nkk;
        const int j = int(i / nx), ixx = int(i % nx);
        rx[idx] = -(1 + beta) * (ix(kx[ixx]) * cp[idx]);
        if (yfo) ry[idx] = -(1 + beta) * (ix(ky[j]) * cp[idx]);
        else {
            const cpx up = j + 1 < ny ? cp[idx + nx] : C(0);
            ry[idx] = -(1 + beta) * ((up - cp[idx]) * id[j]);
        }
        ez[idx] = C(0); bz[idx] = C(0);
    });
    device_sync();
}

// Sponge on the radiation part R - R_b of the forward field (R_b = (1 + beta) E_perp of the charge of the
// slice, T psi = rho with T = k_x^2 + k_y^2 (Fourier y) or k_x^2 - Lap_y (free-hand y), plus the first
// 1/gamma^2 correction), then E and B of the own slices in real space, then the ghost slices.
void Simulation::sponge_and_real_fields() {
    const long nkk = nk;
    const int nx = cfg.nx, ny = cfg.ny;
    cpx* rx = Rx.data(); cpx* ry = Ry.data();
    cpx* s1 = S1.data(); cpx* s2 = S2.data();
    const cpx* jz = jkz.data();
    const double* kx = kxd.data(); const double* id = iD.data(); const double* ia = iA.data();
    if (cfg.sponge > 0 || cfg.sponge_y > 0) {
        const double* pc = pcp.data(); const double* pd = pid.data();
        const double* wl = wlo.data(); const double* wu = whi.data();
        const int a_ = a;
        // psi of the slice charge per y-line (LU factors of T = k_x^2 - Lap_y precomputed), into S2; R - R_b
        // packed into S1. The bound field includes the first correction in 1/gamma^2 of the co-moving field,
        // psi = T^-1 (rho + gamma^-2 T^-1 d_zeta^2 rho), and R_b = (1 + beta) E_perp.
        const double* hc = hcg.data();
        const double g2 = gm2_b, opb = 1 + beta_b;
        const cpx* jg = jzg.data();
        const int ns_ = ns, Ng = N;
        const bool yfo = yf;
        const double* ky = yf ? kyd.data() : nullptr;
        if (yfo) par_for(long(ns) * nk, [=] HD(long idx) {   // Fourier y: T = k^2 is a number
            const long i = idx % nkk;
            const int ixx = int(i % nx), j = int(i / nx), l = int(idx / nkk), n = a_ + l;
            auto half = [&](int hk) -> cpx {
                if (hk < 0) return jg[i];
                if (hk > ns_) return jg[nkk + i];
                return jz[long(hk) * nkk + i];
            };
            auto rho_at = [&](int dl) -> cpx {
                const int nn = n + dl;
                if (nn < 0 || nn > Ng - 1) return C(0);
                return wl[nn] * half(l + dl) + wu[nn] * half(l + dl + 1);
            };
            const double hm = hc[n + 1], hp = hc[n + 2];
            const double cm = 2 / ((hm + hp) * hm), cp_ = 2 / ((hm + hp) * hp);
            const double k2 = kx[ixx] * kx[ixx] + ky[j] * ky[j];
            cpx psi = C(0);
            if (k2 > 0) {
                const cpx d2 = g2 * (cp_ * rho_at(1) - (cm + cp_) * rho_at(0) + cm * rho_at(-1));
                psi = (rho_at(0) + d2 / k2) / k2;
            }
            s2[idx] = psi;
            const cpx bx = -opb * (ix(kx[ixx]) * psi), by = -opb * (ix(ky[j]) * psi);
            const cpx u = rx[idx] - bx, v = ry[idx] - by;
            s1[idx] = C(u.re - v.im, u.im + v.re);
        });
        else par_for(long(ns) * nx, [=] HD(long line) {
            const int ixx = int(line % nx), l = int(line / nx), n = a_ + l;
            const long o = long(l) * nkk + ixx;
            auto half = [&](int hk, long jo) -> cpx {        // half-cell plane hk = -1 .. ns+1 (ghosts a-2 and b)
                if (hk < 0) return jg[ixx + jo];
                if (hk > ns_) return jg[nkk + ixx + jo];
                return jz[long(hk) * nkk + ixx + jo];
            };
            auto rho_at = [&](int dl, long jo) -> cpx {     // charge of slice n + dl at y-index offset jo
                const int nn = n + dl;
                if (nn < 0 || nn > Ng - 1) return C(0);
                return wl[nn] * half(l + dl, jo) + wu[nn] * half(l + dl + 1, jo);
            };
            const double hm = hc[n + 1], hp = hc[n + 2];
            const double cm = 2 / ((hm + hp) * hm), cp_ = 2 / ((hm + hp) * hp);
            auto solve = [&](bool second) {               // T^-1 of (rho or its d_zeta^2 / gamma^2 + ...) into s2
                cpx dpv = C(0);
                for (int j = 1; j < ny; ++j) {
                    const long jo = long(j) * nx, k = o + jo;
                    cpx r;
                    if (!second) r = g2 * (cp_ * rho_at(1, jo) - (cm + cp_) * rho_at(0, jo) + cm * rho_at(-1, jo));
                    else r = rho_at(0, jo) + s2[k];
                    const double lo = j > 1 ? -ia[j] * id[j - 1] : 0;
                    dpv = (r - lo * dpv) * pd[long(j) * nx + ixx];
                    s2[k] = dpv;
                }
                cpx next = C(0);
                for (int j = ny - 1; j >= 1; --j) { next = s2[o + long(j) * nx] - pc[long(j) * nx + ixx] * next; s2[o + long(j) * nx] = next; }
            };
            solve(false);
            solve(true);
            s2[o] = C(0);
            const cpx a = ix(kx[ixx]);
            for (int j = 0; j < ny; ++j) {
                const long k = o + long(j) * nx;
                const cpx pu = j + 1 < ny ? s2[k + nx] : C(0);
                const cpx bx = -opb * (a * s2[k]), by = -opb * ((pu - s2[k]) * id[j]);
                const cpx u = rx[k] - bx, v = ry[k] - by;
                s1[k] = C(u.re - v.im, u.im + v.re);           // u + i v
            }
        });
        fft_->run(s1, ns, true);
        const double* mx = maskx.data(); const double* mn = masky_n.data(); const double* mh = masky_h.data();
        par_for(long(ns) * nk, [=] HD(long idx) {
            const long i = idx % nkk;
            const int j = int(i / nx), ixx = int(i % nx);
            s1[idx] = C(mx[ixx] * mn[j] * s1[idx].re, mx[ixx] * mh[j] * s1[idx].im);
        });
        fft_->run(s1, ns, false);
        if (yfo) par_for(long(ns) * nk, [=] HD(long idx) {
            const long i = idx % nkk, base = idx - i;
            const int j = int(i / nx), ixx = int(i % nx);
            const long m = base + long((ny - j) % ny) * nx + (nx - ixx) % nx;
            const cpx z = s1[idx], zm = conj(s1[m]);
            rx[idx] = 0.5 * (z + zm) - opb * (ix(kx[ixx]) * s2[idx]);
            ry[idx] = 0.5 * C((z - zm).im, -(z - zm).re) - opb * (ix(ky[j]) * s2[idx]);
        });
        else par_for(long(ns) * nk, [=] HD(long idx) {
            const long i = idx % nkk;
            const int j = int(i / nx), ixx = int(i % nx);
            const long m = idx - ixx + (nx - ixx) % nx;
            const cpx z = s1[idx], zm = conj(s1[m]);
            const cpx pu = j + 1 < ny ? s2[idx + nx] : C(0);
            const cpx bx = -opb * (ix(kx[ixx]) * s2[idx]), by = -opb * ((pu - s2[idx]) * id[j]);
            rx[idx] = j == 0 ? C(0) : 0.5 * (z + zm) + bx;
            ry[idx] = 0.5 * C((z - zm).im, -(z - zm).re) + by;
        });
    }
    // real fields: S1 <- inverse(R_x + i R_y), S2 <- inverse(L_x + i L_y)
    const cpx* lx = Lx.data(); const cpx* ly = Ly.data();
    par_for(long(ns) * nk, [=] HD(long idx) {
        s1[idx] = C(rx[idx].re - ry[idx].im, rx[idx].im + ry[idx].re);
        s2[idx] = C(lx[idx].re - ly[idx].im, lx[idx].im + ly[idx].re);
    });
    fft_->run(s1, ns, true);
    fft_->run(s2, ns, true);
    float* Fp = F.data();
    const long fpl = long(ns + 2) * nkk;      // stride between the components of F
    par_for(long(ns) * nk, [=] HD(long idx) {
        const double r1 = s1[idx].re, r2 = s1[idx].im, l1 = s2[idx].re, l2 = s2[idx].im;
        Fp[0 * fpl + idx] = float(0.5 * (r1 + l1));      // E_x  (nodes)
        Fp[1 * fpl + idx] = float(0.5 * (r2 + l2));      // E_y  (half-nodes of the free-hand y)
        Fp[3 * fpl + idx] = float(0.5 * (l2 - r2));      // B_x  (half-nodes of the free-hand y)
        Fp[4 * fpl + idx] = float(0.5 * (r1 - l1));      // B_y  (nodes)
    });
    const cpx* ez = Ez.data(); const cpx* bz = Bz.data();
    par_for(long(ns) * nk, [=] HD(long idx) { s1[idx] = C(ez[idx].re - bz[idx].im, ez[idx].im + bz[idx].re); });
    fft_->run(s1, ns, true);
    par_for(long(ns) * nk, [=] HD(long idx) {
        Fp[2 * fpl + idx] = float(s1[idx].re);           // E_z  (nodes)
        Fp[5 * fpl + idx] = float(s1[idx].im);           // B_z  (half-nodes of the free-hand y)
    });
    device_sync();
    // ghost slices: the two lowest slices of this block go to the block below; receive b, b+1 from above
    const size_t pl = size_t(nk);
    if (comm.rank > 0) {
        std::vector<float> buf(12 * pl);
        for (int comp = 0; comp < 6; ++comp) F.to_host(&buf[comp * 2 * pl], size_t(comp) * (ns + 2) * pl, 2 * pl);
        comm.isend(comm.rank - 1, TAG_GHOST, buf.data(), buf.size());
    }
    if (comm.rank < comm.size - 1) {
        std::vector<float> buf(12 * pl);
        comm.recv(comm.rank + 1, TAG_GHOST, buf.data(), buf.size());
        for (int comp = 0; comp < 6; ++comp) F.from_host(&buf[comp * 2 * pl], (size_t(comp) * (ns + 2) + ns) * pl, 2 * pl);
    }
}

void Simulation::solve_fields(double D) {
    currents();
    march(D);
    sponge_and_real_fields();
    const long n = long(ns + 1) * nk;
    cpx* a0 = j0x.data(); cpx* b0 = j0y.data(); const cpx* a1 = jkx.data(); const cpx* b1 = jky.data();
    par_for(n, [=] HD(long i) { a0[i] = a1[i]; b0[i] = b1[i]; });
    device_sync();
}
