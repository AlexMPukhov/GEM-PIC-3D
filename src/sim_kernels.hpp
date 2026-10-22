// Device code shared by the Simulation sources: TSC weights, field gather and the RK4 push in z of the bunch
// (src/sim_particles.cpp; the gather is also used by the plasma), and the per-mode march of a slice
// (src/sim_fields.cpp, src/sim_plasma.cpp). Header-only: the kernels are inlined into every caller.
#pragma once
#include "sim_internal.hpp"

// ---------------------------------------------------------------- particles (device code)
namespace {
struct PushCtx {
    const float* F; int a, ns, N, nx, ny, nk, self, yf; double dx, dy, Lx, Ly;
    Axis zax, yax;
    Laser laser; double X0, zc, dz_push, push_phase, zhand, zmax, ylo, yhi, s1, vz_min; int freeze;
};
struct State { double x, y, zeta, px, py, pz; };

// Quadratic (TSC) weights around the nearest point of the index coordinate xi (points at integers).
HD inline void tsc_index(double xi, int idx[3], double w[3]) {
    const int i0 = int(::floor(xi + 0.5));
    const double f = xi - i0;
    w[0] = 0.5 * (0.5 - f) * (0.5 - f); w[1] = 0.75 - f * f; w[2] = 0.5 * (0.5 + f) * (0.5 + f);
    for (int k = 0; k < 3; ++k) idx[k] = i0 - 1 + k;
}
// TSC on the periodic uniform x grid with nodes at i*d.
HD inline void tsc(double x, double d, int n, int idx[3], double w[3]) {
    tsc_index(x / d, idx, w);
    for (int k = 0; k < 3; ++k) idx[k] = (idx[k] % n + n) % n;
}
// TSC on the free-hand y grid, in the index coordinate xi(y) (linear within a cell): nodes at the integers,
// half-nodes at the half-integers (stored at index j for j + 1/2). Points outside 0..ny-1 are walls or beyond.
HD inline void tsc_y(const Axis& ax, double y, int jn[3], double wn[3], int jh[3], double wh[3]) {
    int i; double f;
    ax.locate(y, i, f);
    const double xi = i + f;
    tsc_index(xi, jn, wn);
    tsc_index(xi - 0.5, jh, wh);
}

// Grid fields of level s0 at (x, y, zeta): TSC transverse, linear in zeta between the slices of this rank
// (own slices and the two ghost slices of the rank above).
HD inline void gather(const PushCtx& c, double x, double y, double zeta, double E[3], double B[3]) {
    E[0] = E[1] = E[2] = B[0] = B[1] = B[2] = 0;
    if (!c.self) return;
    int n; double fz;
    c.zax.locate(zeta, n, fz);
    const int fl = n - c.a;
    if (fl < 0 || fl > c.ns) return;
    int ix[3], jn[3], jh[3]; double wx[3], wn[3], wh[3];
    tsc(x, c.dx, c.nx, ix, wx);
    if (c.yf) tsc(y, c.dy, c.ny, jn, wn);           // Fourier y: all components on the nodes, periodic
    else tsc_y(c.yax, y, jn, wn, jh, wh);
    for (int comp = 0; comp < 6; ++comp) {
        const float* p0 = c.F + (size_t(comp) * (c.ns + 2) + fl) * c.nk;
        const float* p1 = p0 + c.nk;
        const bool hf = !c.yf && is_half(comp);
        const int* jj = hf ? jh : jn;
        const double* wy = hf ? wh : wn;
        double v = 0;
        for (int j = 0; j < 3; ++j) {
            if (jj[j] < 0 || jj[j] >= c.ny) continue;
            for (int i = 0; i < 3; ++i) {
                const int k = jj[j] * c.nx + ix[i];
                v += wx[i] * wy[j] * ((1 - fz) * p0[k] + fz * p1[k]);
            }
        }
        if (comp < 3) E[comp] = v; else B[comp - 3] = v;
    }
}
// Characteristics with z as the independent variable: dr/dz = p/p_z, dzeta/dz = -(gamma - p_z)/p_z,
// dp/dz = q (gamma E + p x B)/p_z, with gamma - p_z = (1 + p_perp^2)/(gamma + p_z).
HD inline bool rhs(const PushCtx& c, const State& Y, double z, double qm, State& d) {
    const double g = ::sqrt(1 + Y.px * Y.px + Y.py * Y.py + Y.pz * Y.pz);
    if (Y.pz < 0.3 * g) return false;
    const double gm = (1 + Y.px * Y.px + Y.py * Y.py) / (g + Y.pz);
    double E[3], B[3], Ee[3], Be[3];
    gather(c, ::fmod(::fmod(Y.x, c.Lx) + c.Lx, c.Lx), c.yf ? ::fmod(::fmod(Y.y, c.Ly) + c.Ly, c.Ly) : Y.y, Y.zeta, E, B);
    c.laser.field(Y.x + c.X0, z, z - (Y.zeta - c.zc), Ee, Be);
    for (int k = 0; k < 3; ++k) { E[k] += Ee[k]; B[k] += Be[k]; }
    const double ip = 1 / Y.pz;
    d.x = Y.px * ip; d.y = Y.py * ip; d.zeta = -gm * ip;
    d.px = qm * (g * E[0] + Y.py * B[2] - Y.pz * B[1]) * ip;
    d.py = qm * (g * E[1] + Y.pz * B[0] - Y.px * B[2]) * ip;
    d.pz = qm * (g * E[2] + Y.px * B[1] - Y.py * B[0]) * ip;
    return true;
}
HD inline State axpy(const State& a, double h, const State& b) {
    return State{a.x + h * b.x, a.y + h * b.y, a.zeta + h * b.zeta, a.px + h * b.px, a.py + h * b.py, a.pz + h * b.pz};
}
// RK4 from z0 to the plane s1. A sub-step is at most dz_push and at most push_phase of the laser phase.
// A particle that enters the half-cells of the rank below (zeta < zhand) at the start of a sub-step is
// handed over with the rest of the step.
HD inline void push_one(const PushCtx& c, Particle& p, double z0) {
    if (c.freeze && p.sp == SP_BUNCH) { p.z = c.s1; return; }   // numerics.freeze_bunch: rigid, (x, y, zeta, p) fixed
    State Y{p.x, p.y, p.zeta, p.px, p.py, p.pz};
    double z = z0;
    while (z < c.s1 - 1e-12 * ::fabs(c.s1)) {
        if (Y.zeta < c.zhand) {
            p.x = Y.x; p.y = Y.y; p.zeta = Y.zeta; p.px = Y.px; p.py = Y.py; p.pz = Y.pz;
            p.alive = 2; p.rem = c.s1 - z;
            return;
        }
        if (c.vz_min > 0 && Y.pz < c.vz_min * ::sqrt(1 + Y.px * Y.px + Y.py * Y.py + Y.pz * Y.pz)) {
            // too slow for the beam list: streams in the current march from (z, zeta)
            p.x = Y.x; p.y = Y.y; p.zeta = Y.zeta; p.px = Y.px; p.py = Y.py; p.pz = Y.pz; p.z = z;
            p.alive = 4;
            return;
        }
        double ds = c.dz_push < c.s1 - z ? c.dz_push : c.s1 - z;
        if (c.laser.on) {
            const double g = ::sqrt(1 + Y.px * Y.px + Y.py * Y.py + Y.pz * Y.pz);
            const double hl = g - Y.px * c.laser.kx - Y.pz * c.laser.kz;
            const double lim = c.push_phase * Y.pz / (c.laser.kL * (hl > 1e-12 ? hl : 1e-12));
            if (lim < ds) ds = (c.s1 - z) / ::ceil((c.s1 - z) / lim);
        }
        State k1, k2, k3, k4;
        const double qm = p.qm;
        if (!(rhs(c, Y, z, qm, k1) && rhs(c, axpy(Y, 0.5 * ds, k1), z + 0.5 * ds, qm, k2) &&
              rhs(c, axpy(Y, 0.5 * ds, k2), z + 0.5 * ds, qm, k3) && rhs(c, axpy(Y, ds, k3), z + ds, qm, k4))) {
            p.alive = 0;
            return;
        }
        const double w = ds / 6;
        Y.x += w * (k1.x + 2 * k2.x + 2 * k3.x + k4.x); Y.y += w * (k1.y + 2 * k2.y + 2 * k3.y + k4.y);
        Y.zeta += w * (k1.zeta + 2 * k2.zeta + 2 * k3.zeta + k4.zeta);
        Y.px += w * (k1.px + 2 * k2.px + 2 * k3.px + k4.px); Y.py += w * (k1.py + 2 * k2.py + 2 * k3.py + k4.py);
        Y.pz += w * (k1.pz + 2 * k2.pz + 2 * k3.pz + k4.pz);
        z += ds;
    }
    p.x = Y.x; p.y = Y.y; p.zeta = Y.zeta; p.px = Y.px; p.py = Y.py; p.pz = Y.pz; p.z = c.s1;   // continuous coordinates
    // lost: out of the window in zeta, or within one cell of a wall in y
    p.alive = (p.zeta < 0 || p.zeta >= c.zmax || p.y < c.ylo || p.y > c.yhi) ? 0 : 1;
    // a particle that ends the step in the half-cells of the rank below belongs to that rank
    if (p.alive == 1 && p.zeta < c.zhand) { p.alive = 2; p.rem = 0; }
}
}  // namespace

// ---------------------------------------------------------------- field march (device code)
// Per k_x the y-line of slice n is coupled through the finite differences of y:
//   b+ f_j = (f_{j+1} - f_j)/Delta_j  (nodes -> half-node j+1/2, f = 0 at the walls j = 0, ny)
//   b- g_j = (g_j - g_{j-1})/A_j      (half-nodes -> node j; set to 0 on the wall nodes)
// a = i k_x is a number and commutes with b+-, so the cross terms cancel exactly as in the Fourier form, and
//   [1 + (h/4)(D + h/4)(k_x^2 - b- b+)] E_z = E_z(n+1) - (h/2) Q(n+1) - (h/4) S_E - h j_z   (nodes, Dirichlet)
//   [1 + (h/4)(D + h/4)(k_x^2 - b+ b-)] B_z = B_z(n+1) - (h/2) C(n+1) - (h/4) S_B           (half-nodes, Neumann)
// are two tridiagonal systems along y (Thomas). R and L follow by back-substitution.
namespace {
struct MarchCtx {
    cpx *Rx, *Ry, *Ez, *Bz, *Lx, *Ly;
    const cpx *bnd, *j0x, *j0y, *jkx, *jky, *jkz;
    cpx *sRx, *sRy, *sLx, *sLy, *srhs; double* scp;
    const double *kx, *ky, *iD, *iA, *hcg, *wlo, *whi;
    double D; int a, b, N, nx, ny; long nk;
};

// Fourier y. Slice n of level s1 for the mode i: R by the trapezoidal rule in s, L, E_z, B_z by the
// trapezoidal rule over the half-cell [n, n+1] of length h_n; B_z and E_z decouple and follow in closed
// form, then R and L. The values of slice n+1 (level s1) come from the arrays or, at the top of the block,
// from `bnd`.
HD inline void march_point(const MarchCtx& c, long i, int n) {
    const int l = n - c.a;
    const long o = long(l) * c.nk + i;
    const int ixx = int(i % c.nx), j = int(i / c.nx);
    const cpx a = ix(c.kx[ixx]), b = ix(c.ky[j]);
    const double k2 = c.kx[ixx] * c.kx[ixx] + c.ky[j] * c.ky[j], D = c.D, wl = c.wlo[n], wu = c.whi[n];
    const long od = o, ou = o + c.nk;                // half-cells n-1 and n
    const cpx Jx = wl * (c.j0x[od] + c.jkx[od]) + wu * (c.j0x[ou] + c.jkx[ou]);
    const cpx Jy = wl * (c.j0y[od] + c.jky[od]) + wu * (c.j0y[ou] + c.jky[ou]);
    const cpx Rsx = c.Rx[o] + D * (a * c.Ez[o] + b * c.Bz[o]) - D * Jx;
    const cpx Rsy = c.Ry[o] + D * (b * c.Ez[o] - a * c.Bz[o]) - D * Jy;
    if (n == c.N - 1) {             // front of the window: L = E_z = B_z = 0
        c.Rx[o] = Rsx; c.Ry[o] = Rsy; c.Ez[o] = c.Bz[o] = c.Lx[o] = c.Ly[o] = C(0);
        return;
    }
    cpx pRx, pRy, pEz, pBz, pLx, pLy;
    if (n == c.b - 1) {
        pRx = c.bnd[0 * c.nk + i]; pRy = c.bnd[1 * c.nk + i]; pEz = c.bnd[2 * c.nk + i];
        pBz = c.bnd[3 * c.nk + i]; pLx = c.bnd[4 * c.nk + i]; pLy = c.bnd[5 * c.nk + i];
    } else {
        const long p = o + c.nk;
        pRx = c.Rx[p]; pRy = c.Ry[p]; pEz = c.Ez[p]; pBz = c.Bz[p]; pLx = c.Lx[p]; pLy = c.Ly[p];
    }
    const double h = c.hcg[n + 2];
    const cpx upx = c.jkx[ou], upy = c.jky[ou], upz = c.jkz[ou];
    const cpx Lsx = pLx - 0.25 * h * (a * pEz - b * pBz) - 0.5 * h * upx;
    const cpx Lsy = pLy - 0.25 * h * (a * pBz + b * pEz) - 0.5 * h * upy;
    const cpx Cp = 0.5 * (a * (pRy + pLy) - b * (pRx + pLx));
    const cpx Qp = -0.5 * (a * (pRx - pLx)) + 0.5 * (b * (pLy - pRy));
    const cpx SB = a * Rsy + a * Lsy - b * Rsx - b * Lsx;
    const cpx SE = -(a * Rsx) + a * Lsx + b * Lsy - b * Rsy;
    const double den = 1 + 0.25 * h * (D + 0.25 * h) * k2;
    const cpx Bzn = (pBz - 0.5 * h * Cp - 0.25 * h * SB) / den;
    const cpx Ezn = (pEz - 0.5 * h * Qp - 0.25 * h * SE - h * upz) / den;
    c.Ez[o] = Ezn; c.Bz[o] = Bzn;
    c.Rx[o] = Rsx + D * (a * Ezn + b * Bzn); c.Ry[o] = Rsy + D * (b * Ezn - a * Bzn);
    c.Lx[o] = Lsx - 0.25 * h * (a * Ezn - b * Bzn); c.Ly[o] = Lsy - 0.25 * h * (a * Bzn + b * Ezn);
}

// Free-hand y: the y-line of slice n for k_x index ik.
HD inline void march_line(const MarchCtx& c, int ik, int n) {
    const int nx = c.nx, ny = c.ny;
    const long P0 = long(n - c.a) * c.nk + ik;      // slice n of the block = half-cell n-1 (current planes)
    const long P1 = P0 + c.nk;                      // slice n+1 = half-cell n
    const cpx a = ix(c.kx[ik]);
    const double kx2 = c.kx[ik] * c.kx[ik], D = c.D, wl = c.wlo[n], wu = c.whi[n];
    const double* iD = c.iD; const double* iA = c.iA;
    auto S = [&](cpx* A, int j) -> cpx& { return A[long(j) * nx + ik]; };   // scratch line
    // known part of R (level s0 and the currents of both levels)
    for (int j = 0; j < ny; ++j) {
        const long o = P0 + long(j) * nx;
        if (j == 0) S(c.sRx, 0) = C(0);
        else {
            const cpx Jx = wl * (c.j0x[o] + c.jkx[o]) + wu * (c.j0x[o + c.nk] + c.jkx[o + c.nk]);
            const cpx bm = (c.Bz[o] - c.Bz[o - nx]) * iA[j];
            S(c.sRx, j) = c.Rx[o] + D * (a * c.Ez[o] + bm) - D * Jx;
        }
        const cpx Jy = wl * (c.j0y[o] + c.jky[o]) + wu * (c.j0y[o + c.nk] + c.jky[o + c.nk]);
        const cpx ezu = j + 1 < ny ? c.Ez[o + nx] : C(0);
        S(c.sRy, j) = c.Ry[o] + D * ((ezu - c.Ez[o]) * iD[j] - a * c.Bz[o]) - D * Jy;
    }
    if (n == c.N - 1) {             // front of the window: L = E_z = B_z = 0
        for (int j = 0; j < ny; ++j) {
            const long o = P0 + long(j) * nx;
            c.Rx[o] = S(c.sRx, j); c.Ry[o] = S(c.sRy, j); c.Ez[o] = c.Bz[o] = c.Lx[o] = c.Ly[o] = C(0);
        }
        return;
    }
    const bool top = n == c.b - 1;  // slice n+1 is the boundary slice from the rank above
    auto nxt = [&](int comp, const cpx* A, int j) -> cpx {
        if (j < 0 || j >= ny) return C(0);
        return top ? c.bnd[comp * c.nk + long(j) * nx + ik] : A[P1 + long(j) * nx];
    };
    auto pRx = [&](int j) { return nxt(0, c.Rx, j); }; auto pRy = [&](int j) { return nxt(1, c.Ry, j); };
    auto pEz = [&](int j) { return j == 0 ? C(0) : nxt(2, c.Ez, j); }; auto pBz = [&](int j) { return nxt(3, c.Bz, j); };
    auto pLx = [&](int j) { return nxt(4, c.Lx, j); }; auto pLy = [&](int j) { return nxt(5, c.Ly, j); };
    const double h = c.hcg[n + 2], cc = 0.25 * h * (D + 0.25 * h);
    // known part of L
    for (int j = 0; j < ny; ++j) {
        const long u = P1 + long(j) * nx;           // half-cell n, level s1
        if (j == 0) S(c.sLx, 0) = C(0);
        else {
            const cpx bm = (pBz(j) - pBz(j - 1)) * iA[j];
            S(c.sLx, j) = pLx(j) - 0.25 * h * (a * pEz(j) - bm) - 0.5 * h * c.jkx[u];
        }
        const cpx bp = (pEz(j + 1) - pEz(j)) * iD[j];
        S(c.sLy, j) = pLy(j) - 0.25 * h * (a * pBz(j) + bp) - 0.5 * h * c.jky[u];
    }
    // E_z on the nodes 1..ny-1
    double cp = 0;
    cpx dp = C(0);
    for (int j = 1; j < ny; ++j) {
        const cpx Qp = 0.5 * (-(a * (pRx(j) - pLx(j))) + ((pLy(j) - pRy(j)) - (pLy(j - 1) - pRy(j - 1))) * iA[j]);
        const cpx SE = -(a * (S(c.sRx, j) - S(c.sLx, j))) +
                       ((S(c.sLy, j) - S(c.sRy, j)) - (S(c.sLy, j - 1) - S(c.sRy, j - 1))) * iA[j];
        const cpx r = pEz(j) - 0.5 * h * Qp - 0.25 * h * SE - h * c.jkz[P1 + long(j) * nx];
        const double lo = j > 1 ? -cc * iA[j] * iD[j - 1] : 0, up = j < ny - 1 ? -cc * iA[j] * iD[j] : 0;
        const double den = 1 + cc * kx2 + cc * iA[j] * (iD[j - 1] + iD[j]) - lo * cp;
        dp = (r - lo * dp) / den;
        cp = up / den;
        S(c.srhs, j) = dp; c.scp[long(j) * nx + ik] = cp;
    }
    {
        cpx next = C(0);
        for (int j = ny - 1; j >= 1; --j) { next = S(c.srhs, j) - c.scp[long(j) * nx + ik] * next; c.Ez[P0 + long(j) * nx] = next; }
        c.Ez[P0] = C(0);
    }
    // B_z on the half-nodes 0..ny-1
    cp = 0; dp = C(0);
    for (int j = 0; j < ny; ++j) {
        const cpx fu = j + 1 < ny ? pRx(j + 1) + pLx(j + 1) : C(0), f0 = pRx(j) + pLx(j);
        const cpx Cp = 0.5 * (a * (pRy(j) + pLy(j)) - (fu - f0) * iD[j]);
        const cpx su = j + 1 < ny ? S(c.sRx, j + 1) + S(c.sLx, j + 1) : C(0), s0 = S(c.sRx, j) + S(c.sLx, j);
        const cpx SB = a * (S(c.sRy, j) + S(c.sLy, j)) - (su - s0) * iD[j];
        const cpx r = pBz(j) - 0.5 * h * Cp - 0.25 * h * SB;
        const double al = j > 0 ? iA[j] : 0, au = j < ny - 1 ? iA[j + 1] : 0;
        const double lo = -cc * iD[j] * al, up = -cc * iD[j] * au;
        const double den = 1 + cc * kx2 + cc * iD[j] * (al + au) - lo * cp;
        dp = (r - lo * dp) / den;
        cp = up / den;
        S(c.srhs, j) = dp; c.scp[long(j) * nx + ik] = cp;
    }
    {
        cpx next = C(0);
        for (int j = ny - 1; j >= 0; --j) { next = S(c.srhs, j) - c.scp[long(j) * nx + ik] * next; c.Bz[P0 + long(j) * nx] = next; }
    }
    // R and L
    for (int j = 0; j < ny; ++j) {
        const long o = P0 + long(j) * nx;
        if (j == 0) { c.Rx[o] = C(0); c.Lx[o] = C(0); }
        else {
            const cpx bm = (c.Bz[o] - c.Bz[o - nx]) * iA[j];
            c.Rx[o] = S(c.sRx, j) + D * (a * c.Ez[o] + bm);
            c.Lx[o] = S(c.sLx, j) - 0.25 * h * (a * c.Ez[o] - bm);
        }
        const cpx ezu = j + 1 < ny ? c.Ez[o + nx] : C(0);
        const cpx bp = (ezu - c.Ez[o]) * iD[j];
        c.Ry[o] = S(c.sRy, j) + D * (bp - a * c.Bz[o]);
        c.Ly[o] = S(c.sLy, j) - 0.25 * h * (a * c.Bz[o] + bp);
    }
}
}  // namespace
