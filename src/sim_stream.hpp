// Device code of the streaming plasma (included by src/sim_plasma.cpp): Boris push, the current deposit of a
// streaming particle, the field ionization and the push of one particle through a half-cell.
#pragma once
#include "sim_internal.hpp"
#include "sim_kernels.hpp"

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
    // ions: species constants, ADK levels, the list of electrons freed in this pass (count, capacity), random key
    const IonDev* ions; const AdkLevel* adk; Particle* born; int* bcnt; int bcap; uint64_t key;
};

// TSC deposit of q w (dt / Delta) v / (h dx dy) at (x, y) into the half-cell plane (box coordinates); the charge
// goes to plane 3 (negative particles) or 4 (positive: ions, positrons)
HD inline void stream_deposit(const StreamCtx& c, double x, double y, double f, double vx, double vy, double vz) {
    const PushCtx& P = c.pc;
    int ix_[3], jn[3], jh[3]; double wx[3], wn[3], wh[3];
    tsc(x, P.dx, P.nx, ix_, wx);
    const long nk = P.nk, cq = (f < 0 ? 3 : 4) * nk;
    if (P.yf) {
        tsc(y, P.dy, P.ny, jn, wn);
        for (int v = 0; v < 3; ++v)
            for (int u = 0; u < 3; ++u) {
                const long k = long(jn[v]) * P.nx + ix_[u];
                const double w = f * wx[u] * wn[v] * c.ia[0];
                atomic_add(&c.J[k], w * vx); atomic_add(&c.J[nk + k], w * vy); atomic_add(&c.J[2 * nk + k], w * vz);
                atomic_add(&c.J[cq + k], w);
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
                atomic_add(&c.J[k], w * vx); atomic_add(&c.J[2 * nk + k], w * vz); atomic_add(&c.J[cq + k], w);
            }
            if (hal) {
                const long k = long(jh[v]) * P.nx + ix_[u];
                atomic_add(&c.J[nk + k], f * wx[u] * wh[v] * c.id[jh[v]] * vy);
            }
        }
    }
}

// |E| (grid and analytic laser) at (x, y) at the lab position z and the slice position zeta
HD inline double field_abs(const StreamCtx& c, double x, double y, double z, double zeta) {
    const PushCtx& P = c.pc;
    double E[3], B[3], Ee[3], Be[3];
    gather(P, ::fmod(::fmod(x, P.Lx) + P.Lx, P.Lx), P.yf ? ::fmod(::fmod(y, P.Ly) + P.Ly, P.Ly) : y, zeta, E, B);
    P.laser.field(x + P.X0, z, z - (zeta - P.zc), Ee, Be);
    for (int k = 0; k < 3; ++k) E[k] += Ee[k];
    return ::sqrt(E[0] * E[0] + E[1] * E[1] + E[2] * E[2]);
}

// Monte Carlo field ionization of an ion macroparticle during dt (as GEM-PIC-RZ): the ADK rate of its level is
// averaged over the fields at the two ends of the half-cell, (w(E1) + w(E0))/2, and sequential ionization
// k -> k+1 -> ... is sampled with exponential waiting times. Each event frees one electron per represented atom,
// at the ion with its velocity.
HD inline void ionize(const StreamCtx& c, Particle& p, double E1, double E0, double dt) {
    const IonDev& I = c.ions[p.sp - SP_ION];
    if (!I.ionizing || (E1 < I.Emin && E0 < I.Emin)) return;
    const uint64_t base = ion_key_particle(c.key, p.id);
    int Z = int(::floor(p.q + 0.5));
    const int Z0 = Z;
    unsigned draw = 0;
    double tleft = dt;
    while (Z < I.Zmax) {
        const AdkLevel& L = c.adk[I.off + Z];
        const double w = 0.5 * (adk_rate(L, E1) + adk_rate(L, E0));
        if (!(w > 0)) break;
        const double tw = -::log(1.0 - uniform01(base + draw++)) / w;
        if (tw > tleft) break;
        tleft -= tw; ++Z;
        Particle e = p;
        e.q = -1; e.qm = -1; e.sp = SP_FREED + (p.sp - SP_ION); e.alive = 1;
        const int j = atomic_inc(c.bcnt);
        if (j < c.bcap) c.born[j] = e;
    }
    if (Z != Z0) { p.q = Z; p.qm = Z / I.mass; }
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
    const bool ion = is_ion(p.sp);
    if (ion && !c.ions[p.sp - SP_ION].mobile) {     // fixed ion (ionizing): its charge, then the ionization
        const double f = p.q * p.w * hl / (c.Delta * h * P.dx);
        if (f != 0) stream_deposit(c, p.x, p.y, f, 0, 0, 0);
        const double E1 = field_abs(c, p.x, p.y, p.z, p.zeta), E0 = field_abs(c, p.x, p.y, p.z, zn);
        p.zeta = zn;
        ionize(c, p, E1, E0, hl);
        return;
    }
    const double zeta0 = p.zeta;
    const double g0 = ::sqrt(1 + p.px * p.px + p.py * p.py + p.pz * p.pz);
    const double omv0 = (1 + p.px * p.px + p.py * p.py) / (g0 * (g0 + p.pz));    // 1 - v_z, no cancellation
    const double dt0 = hl / omv0;
    const double xm = p.x + 0.5 * dt0 * p.px / g0, ym = p.y + 0.5 * dt0 * p.py / g0;
    const double zm = p.z + 0.5 * dt0 * p.pz / g0, zetam = p.zeta - 0.5 * hl;
    double E[3], B[3], Ee[3], Be[3];
    gather(P, ::fmod(::fmod(xm, P.Lx) + P.Lx, P.Lx), P.yf ? ::fmod(::fmod(ym, P.Ly) + P.Ly, P.Ly) : ym, zetam, E, B);
    P.laser.field(xm + P.X0, zm, zm - (zetam - P.zc), Ee, Be);
    for (int k = 0; k < 3; ++k) { E[k] += Ee[k]; B[k] += Be[k]; }
    if (P.ext.any()) add_external(P.ext, xm + P.X0, ym + P.Y0, zm, B);
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
    if (f != 0) stream_deposit(c, 0.5 * (xo + p.x), 0.5 * (yo + p.y), f, vx, vy, vz);
    if (!P.yf && (p.y < P.ylo || p.y > P.yhi)) { p.alive = 0; return; }
    if (ion && !trapped && c.ions[p.sp - SP_ION].ionizing)
        ionize(c, p, field_abs(c, xm, ym, zm, zeta0), field_abs(c, xm, ym, zm, zn), dt);
    if (trapped) {
        p.alive = 5;
        const double vze = uz / g1;
        if (vze >= c.vz_beam)        // joins the beam list: its beam current of level s1 from now on
            stream_deposit(c, p.x, p.y, p.q * p.w / (h * P.dx * vze), ux / g1, uy / g1, vze);
    }
}
}  // namespace
