// GEM-PIC-3D: Galilean electromagnetic PIC in three dimensions.
// Transverse directions: Fourier modes (periodic box with absorbing sponge); longitudinal: zeta = z - t,
// marched from the front of the window to the rear; slow coordinate s = z (GT2 for the particles).
// Units: c = 1, lengths 1/k0, times 1/omega0, momenta m c, fields m c omega0/e, densities n_c (of lambda0).
//
// Parallelism: every kernel runs as an OpenMP loop or a CUDA kernel (portable.hpp); MPI decomposes the
// window along zeta into blocks of slices, rank P-1 at the front.
#pragma once
#include <cstdio>
#include <deque>
#include <memory>
#include <string>
#include <vector>
#include "comm.hpp"
#include "deck.hpp"
#include "fftmany.hpp"
#include "portable.hpp"

struct Config {
    double lambda0 = 0, Lu = 0, Tu = 0, nc = 0;       // reference wavelength [m], code length [m], code time [s], n_c [m^-3]
    int nx = 0, ny = 0, nzeta = 0;
    double Lx = 0, Ly = 0, Lwin = 0, h = 0;
    double dz = 0, z_end = 0, dz_push = 0;
    double sponge = 0, sponge_rate = 0.1;
    bool move_x = true, move_y = true;
    int window_lag = 4;                                // the window shift of step k uses the centroid of step k - lag
    bool self_fields = true, zeta_filter = true, k_filter = true;
    double push_phase = 0.05;
    bool laser_on = false;
    double a0 = 0, kL = 1, theta = 0, tau = 1, cep = 0;
    std::string pol = "s";
    double start_fwhm = 3.5, stop_fwhm = 3.5;
    double charge = 0, gamma = 1, sigx = 0, sigy = 0, sigz = 0, emitx = 0, emity = 0, espread = 0;
    double zeta_center = 0;
    long nmacro = 0;
    std::string noise = "random";
    double noise_lambda = 0; int beamlet = 4; unsigned long seed = 1;
    std::string dir = "out";
    int every = 10;
    bool dump_particles = true;

    static Config from_deck(const Deck& d);
    static Config base_units(double lambda0);          // units only (used by the tests)
    void summary(FILE* f, int ranks) const;
};

// Plane-wave pulse of infinite transverse size: A = a0 g(eta) cos(kL (eta - eta_c) + cep) e_pol,
// eta = t - khat . r, khat = (sin theta, 0, cos theta), E = -dA/deta, B = khat x E.
struct Laser {
    int on = 0, pol = 0;                // pol: 0 s (E along y), 1 p (E in the x-z plane), 2 circular
    double a0 = 0, kL = 1, tau = 1, cep = 0, eta_c = 0, kx = 0, kz = 1;
    HD double eta(double x, double z, double t) const { return t - kx * x - kz * z; }
    HD double envelope(double et) const { const double x = et - eta_c; return ::exp(-1.3862943611198906 * x * x / (tau * tau)); }
    HD void field(double x, double z, double t, double E[3], double B[3]) const {
        E[0] = E[1] = E[2] = B[0] = B[1] = B[2] = 0;
        if (!on) return;
        const double xi = eta(x, z, t) - eta_c;
        if (::fabs(xi) > 5 * tau) return;
        const double g = ::exp(-1.3862943611198906 * xi * xi / (tau * tau));     // 2 ln 2
        const double gp = -2.7725887222397811 * xi / (tau * tau) * g;            // 4 ln 2
        const double ph = kL * xi + cep, c = ::cos(ph), s = ::sin(ph);
        const double d1 = a0 * (gp * c - kL * g * s), d2 = a0 * (gp * s + kL * g * c);
        if (pol == 0) { E[1] = -d1; }
        else if (pol == 1) { E[0] = -d1 * kz; E[2] = d1 * kx; }
        else { E[0] = -d1 * kz; E[2] = d1 * kx; E[1] = -d2; }
        B[0] = -kz * E[1]; B[1] = kz * E[0] - kx * E[2]; B[2] = kx * E[1];     // khat x E
    }
};

struct Particle { double x, y, zeta, px, py, pz, w, rem; int alive, pad; };
// alive: 0 lost, 1 own, 2 handed to the rank below during the push (rem = z still to go), 3 adopted

class Simulation {
public:
    Simulation(const Config& c, Comm& comm);
    void init();
    void step();
    bool finished() const { return s >= cfg.z_end - 1e-9 || stop_; }
    void diagnostics(FILE* f, bool header);        // collective; f is used on rank 0 only
    void dump_particles(const std::string& file);  // collective

    // fields of level s1 (march with D = dz/2), or E_z, B_z, L of the stored R (D = 0)
    void solve_fields(double D);

    // decomposition: this rank owns the slices [a, b) and the particles of the half-cells [a+1, b]
    // (rank 0 from half-cell 0)
    Config cfg;
    Comm& comm;
    Laser laser;
    int N = 0, a = 0, b = 0, ns = 0, nk = 0;
    double dx = 0, dy = 0, s = 0, X0 = 0, Y0 = 0;
    long istep = 0;
    double field_energy = 0, W_macro_ne = 0;

    // device arrays of nk-point planes: slice planes l = n - a (0..ns-1); half-cell planes hk = m - (a-1)
    // (0..ns); unfiltered current planes hw = m - (a-2) (0..ns+2); real-field planes fl = n - a (0..ns+1:
    // own slices and the ghost slices b, b+1 of the rank above)
    Buf<cpx> Rx, Ry, Ez, Bz, Lx, Ly;        // ns planes
    Buf<cpx> bnd;                           // slice b of level s1 from the rank above: Rx Ry Ez Bz Lx Ly
    Buf<cpx> j0x, j0y, jkx, jky, jkz;       // filtered spectra of the half-cell currents, levels s0 and s1
    Buf<double> jw;                         // unfiltered deposit, 3 components x (ns+3) planes
    Buf<float> F;                           // E, B in real space: F[(c*(ns+2) + fl)*nk + i]
    Buf<cpx> S1, S2;                        // FFT scratch, ns+1 planes each
    Buf<double> kxd, kyd, kxs, kys, filt, mask;
    Buf<Particle> parts;
    long np = 0;                            // particles in `parts`
    long lost_total = 0;

    double total_field_energy();          // collective, code units
    std::vector<Particle> host_particles() const;
    void set_particles(const std::vector<Particle>& v);

    // internal steps (public because nvcc forbids device lambdas in private member functions)
    void load_bunch();
    void comoving_field();
    void push(double s0, double s1, long i0, long i1);
    void exchange_particles(double s1);
    void currents();             // deposit, exchange with the rank below/above, zeta filter, spectra
    void march(double D);
    void sponge_and_real_fields();
    void window_shift_start();
    void window_shift_post();
    void shift_window(int nxs, int nys);
    double local_field_energy();

private:
    std::unique_ptr<FftMany> fft_;
    std::deque<std::pair<int, std::vector<double>>> cent_;   // pending centroid reductions (handle, sums)
    bool stop_ = false;
};
