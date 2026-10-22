// Bunch particles: push to the next plane, MPI handover, current deposition and its spectra.
#include "sim_internal.hpp"
#include "sim_kernels.hpp"

// Push the particles [i0, i1) to the plane s1: own particles from s0, adopted ones (alive = 3) for their rest.
void Simulation::push(double s0, double s1, long i0, long i1) {
    PushCtx c;
    c.F = F.data(); c.a = a; c.ns = ns; c.N = N; c.nx = cfg.nx; c.ny = cfg.ny; c.nk = nk; c.self = cfg.self_fields;
    c.yf = yf; c.dx = dx; c.dy = dy; c.Lx = cfg.Lx; c.Ly = cfg.Ly; c.zax = zax; c.yax = yax; c.laser = laser; c.X0 = X0;
    c.zc = cfg.zeta_center; c.dz_push = cfg.dz_push; c.push_phase = cfg.push_phase;
    c.zhand = comm.rank == 0 ? -1e300 : cfg.zeta[a + 1];
    c.zmax = cfg.zeta[N - 1]; c.s1 = s1; c.vz_min = cfg.plasma_on ? cfg.vz_min : 0; c.freeze = cfg.freeze_bunch;
    c.ylo = yf ? -1e300 : cfg.ynode[1]; c.yhi = yf ? 1e300 : cfg.ynode[cfg.ny - 1];
    Particle* P = parts.data();
    par_for(i1 - i0, [=] HD(long k) {
        Particle& p = P[i0 + k];
        if (p.alive == 1) push_one(c, p, s0);
        else if (p.alive == 3) push_one(c, p, s1 - p.rem);
    });
    device_sync();
}

// Hand the particles that entered the half-cells of the rank below to it, adopt those of the rank above
// and finish their push. Lost particles are removed here.
void Simulation::exchange_particles(double s1) {
    if (comm.size == 1) return;
    std::vector<Particle> all = host_particles(), keep, hand;
    keep.reserve(all.size());
    for (auto& p : all) {
        if (p.alive == 1) keep.push_back(p);
        else if (p.alive == 2) hand.push_back(p);
        else ++lost_total;
    }
    if (comm.rank > 0) comm.isend(comm.rank - 1, TAG_HAND, hand.data(), hand.size());
    else if (!hand.empty()) throw std::runtime_error("handover on rank 0");
    const long n_own = long(keep.size());
    if (comm.rank < comm.size - 1) {
        std::vector<Particle> got = comm.recv_vec<Particle>(comm.rank + 1, TAG_HAND);
        for (auto& p : got) { p.alive = 3; keep.push_back(p); }
    }
    set_particles(keep);
    if (np > n_own) push(s1, s1, n_own, np);
    for (auto& p : host_particles())
        if (p.alive == 2) throw std::runtime_error("an adopted particle crossed a whole block in one step (blocks too small)");
}

// ---------------------------------------------------------------- currents
// Deposit (a particle on the plane z = s carries q w / v_z per unit zeta, j = q w v / v_z: NGP over the
// half-cell in zeta, TSC transverse; j_x and j_z on the y-nodes, j_y on the half-nodes), exchange of the
// half-cells at the block edges, (1,2,1)/4 filter along zeta on the half-cell charges, spectra in x, the
// transverse filters (cos^2 in k_x, (1,2,1)/4 on the cell charges along y), and no TEM current between the
// walls (the y-mean of j_y at k_x = 0).
void Simulation::currents() {
    jw.zero();
    const int nh = ns + 3, a_ = a, nx = cfg.nx, ny = cfg.ny, Ng = N;
    const long nkk = nk;
    const double dx_ = dx, dy_ = dy;
    const bool yfo = yf;
    const Axis za = zax, ya = yax;
    const double* hc = hcg.data(); const double* ia = iA.data(); const double* id = iD.data();
    double* J = jw.data();
    const Particle* P = parts.data();
    par_for(np, [=] HD(long ip) {
        const Particle& p = P[ip];
        if (p.alive != 1) return;
        int m; double fz;
        za.locate(p.zeta, m, fz);
        const int hw = m - (a_ - 2);
        if (hw < 0 || hw >= nh || m < 0 || m > Ng - 2) return;
        const double g = ::sqrt(1 + p.px * p.px + p.py * p.py + p.pz * p.pz);
        const double vx = p.px / g, vy = p.py / g, vz = p.pz / g;
        const double f = p.q * p.w / (hc[m + 2] * dx_);
        int ix[3], jn[3], jh[3]; double wx[3], wn[3], wh[3];
        tsc(p.x, dx_, nx, ix, wx);
        if (yfo) {                                   // Fourier y: all components on the periodic nodes
            tsc(p.y, dy_, ny, jn, wn);
            for (int v = 0; v < 3; ++v)
                for (int u = 0; u < 3; ++u) {
                    const long k = long(jn[v]) * nx + ix[u];
                    const double w = f * wx[u] * wn[v] * ia[0];
                    atomic_add(&J[(0L * nh + hw) * nkk + k], w * vx / vz);
                    atomic_add(&J[(1L * nh + hw) * nkk + k], w * vy / vz);
                    atomic_add(&J[(2L * nh + hw) * nkk + k], w);
                }
            return;
        }
        tsc_y(ya, p.y, jn, wn, jh, wh);
        for (int v = 0; v < 3; ++v) {
            const bool nod = jn[v] >= 1 && jn[v] <= ny - 1, hal = jh[v] >= 0 && jh[v] <= ny - 1;
            for (int u = 0; u < 3; ++u) {
                if (nod) {
                    const long k = long(jn[v]) * nx + ix[u];
                    const double w = f * wx[u] * wn[v] * ia[jn[v]];
                    atomic_add(&J[(0L * nh + hw) * nkk + k], w * vx / vz);
                    atomic_add(&J[(2L * nh + hw) * nkk + k], w);
                }
                if (hal) {
                    const long k = long(jh[v]) * nx + ix[u];
                    atomic_add(&J[(1L * nh + hw) * nkk + k], f * wx[u] * wh[v] * id[jh[v]] * vy / vz);
                }
            }
        }
    });
    device_sync();
    // the top three half-cells of this block are the bottom three ghosts of the block above
    const size_t pl = size_t(nk);
    if (comm.rank < comm.size - 1) {
        std::vector<double> buf(9 * pl);
        for (int comp = 0; comp < 3; ++comp) jw.to_host(&buf[comp * 3 * pl], (size_t(comp) * nh + ns) * pl, 3 * pl);
        comm.isend(comm.rank + 1, TAG_CUR, buf.data(), buf.size());
    }
    if (comm.rank > 0) {
        std::vector<double> buf(9 * pl);
        comm.recv(comm.rank - 1, TAG_CUR, buf.data(), buf.size());
        for (int comp = 0; comp < 3; ++comp) jw.from_host(&buf[comp * 3 * pl], (size_t(comp) * nh + 0) * pl, 3 * pl);
    }
    // filter along zeta (on the charges h_m j_m) into the half-cells a-1 .. b-1; jx + i jy packed into S1, jz into jkz
    const bool zf = cfg.zeta_filter, zu = zax.uniform;
    cpx* s1 = S1.data(); cpx* jz = jkz.data();
    par_for(long(ns + 1) * nk, [=] HD(long idx) {
        const int hk = int(idx / nkk);
        const long i = idx % nkk;
        const int hc_ = a_ - 1 + hk, hw = hk + 1;
        double v[3] = {0, 0, 0};
        if (hc_ >= 0 && hc_ <= Ng - 2)
            for (int comp = 0; comp < 3; ++comp) {
                const double* Jc = J + size_t(comp) * nh * nkk;
                if (zf && zu) v[comp] = 0.25 * Jc[(hw - 1) * nkk + i] + 0.5 * Jc[hw * nkk + i] + 0.25 * Jc[(hw + 1) * nkk + i];
                else v[comp] = zf ? (0.25 * hc[hc_ + 1] * Jc[(hw - 1) * nkk + i] + 0.5 * hc[hc_ + 2] * Jc[hw * nkk + i] +
                                0.25 * hc[hc_ + 3] * Jc[(hw + 1) * nkk + i]) / hc[hc_ + 2]
                             : Jc[hw * nkk + i];
            }
        s1[idx] = C(v[0], v[1]);
        jz[idx] = C(v[2], 0);
    });
    fft_->run(S1.data(), ns + 1, false);
    fft_->run(jkz.data(), ns + 1, false);
    cpx* jx = jkx.data(); cpx* jy = jky.data();
    const double* fl = filt.data();
    const bool keep0 = cfg.plasma_on;     // with plasma the k = 0 current is a 1D plasma wave, not a net current
    if (yfo) {
        // Fourier y: unpack the pair, apply the transverse filter (zero at the Nyquist modes, and at k = 0 in vacuum)
        const double* fly = filty.data();
        par_for(long(ns + 1) * nk, [=] HD(long idx) {
            const long i = idx % nkk, base = idx - i;
            const int ixx = int(i % nx), j = int(i / nx);
            const long m = base + long((ny - j) % ny) * nx + (nx - ixx) % nx;
            const cpx z = s1[idx], zm = conj(s1[m]);
            const double f = (ixx == 0 && j == 0 && !keep0) ? 0.0 : fl[ixx] * fly[j];
            jx[idx] = f * (0.5 * (z + zm)); jy[idx] = f * (0.5 * C((z - zm).im, -(z - zm).re)); jz[idx] = f * jz[idx];
        });
        device_sync();
        exchange_jz_ghosts();
        return;
    }
    // free-hand y: unpack the pair
    par_for(long(ns + 1) * nk, [=] HD(long idx) {
        const long i = idx % nkk;
        const int ixx = int(i % nx);
        const long m = idx - ixx + (nx - ixx) % nx;
        const cpx z = s1[idx], zm = conj(s1[m]);
        jx[idx] = 0.5 * (z + zm); jy[idx] = 0.5 * C((z - zm).im, -(z - zm).re);
    });
    // transverse filters per y-line
    const bool kf = cfg.k_filter;
    const double Ly = cfg.Ly;
    par_for(long(ns + 1) * nx, [=] HD(long line) {
        const int ixx = int(line % nx);
        const long o = (line / nx) * nkk + ixx;          // element j at o + j*nx
        const double fx = fl[ixx];
        if (kf) {
            cpx* const nodal[2] = {jx, jz};
            for (int t = 0; t < 2; ++t) {                // nodes: charges Q_j = A_j J_j, Q = 0 at the walls
                cpx* A = nodal[t];
                cpx qm = C(0), q0 = A[o + nx] / ia[1];
                for (int j = 1; j < ny; ++j) {
                    const cpx qp = j + 1 < ny ? A[o + long(j + 1) * nx] / ia[j + 1] : C(0);
                    A[o + long(j) * nx] = (0.25 * qm + 0.5 * q0 + 0.25 * qp) * ia[j];
                    qm = q0; q0 = qp;
                }
            }
            cpx qm = jy[o] / id[0], q0 = qm;             // half-nodes: Q_j = Delta_j J_j, mirrored at the walls
            for (int j = 0; j < ny; ++j) {
                const cpx qp = j + 1 < ny ? jy[o + long(j + 1) * nx] / id[j + 1] : q0;
                jy[o + long(j) * nx] = (0.25 * qm + 0.5 * q0 + 0.25 * qp) * id[j];
                qm = q0; q0 = qp;
            }
        }
        cpx mean = C(0);
        if (ixx == 0 && !keep0) {
            for (int j = 0; j < ny; ++j) mean = mean + jy[o + long(j) * nx] / id[j];
            mean = mean / Ly;
        }
        jx[o] = C(0); jz[o] = C(0);
        for (int j = 0; j < ny; ++j) {
            const long k = o + long(j) * nx;
            jx[k] = fx * jx[k]; jz[k] = fx * jz[k]; jy[k] = fx * (jy[k] - mean);
        }
    });
    device_sync();
    exchange_jz_ghosts();
}

// filtered j_z of the half-cells a-2 and b from the neighbours (bound field of the sponge, d_zeta^2 rho)
void Simulation::exchange_jz_ghosts() {
    const size_t pl = size_t(nk);
    std::vector<cpx> lo(pl, C(0)), hi(pl, C(0));
    if (comm.rank < comm.size - 1) {
        std::vector<cpx> buf(pl);
        jkz.to_host(buf.data(), size_t(ns - 1) * pl, pl);
        comm.isend(comm.rank + 1, TAG_JZU, buf.data(), pl);
    }
    if (comm.rank > 0) {
        std::vector<cpx> buf(pl);
        jkz.to_host(buf.data(), pl, pl);
        comm.isend(comm.rank - 1, TAG_JZD, buf.data(), pl);
        comm.recv(comm.rank - 1, TAG_JZU, lo.data(), pl);
    }
    if (comm.rank < comm.size - 1) comm.recv(comm.rank + 1, TAG_JZD, hi.data(), pl);
    jzg.from_host(lo.data(), 0, pl); jzg.from_host(hi.data(), pl, pl);
}
