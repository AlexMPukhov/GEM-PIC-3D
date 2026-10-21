// GEM-PIC-3D: Galilean electromagnetic PIC in three dimensions.
// x: Fourier modes (periodic, absorbing sponge, window that follows the bunch). y: Fourier modes as well
// (default), or optionally finite differences on a free-hand staggered grid between conducting walls
// (absorbing sponge). zeta = z - t: uniform or free-hand slices,
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
#include "grid.hpp"
#include "portable.hpp"

struct Config {
    double lambda0 = 0, Lu = 0, Tu = 0, nc = 0;       // reference wavelength [m], code length [m], code time [s], n_c [m^-3]
    int nx = 0, ny = 0, nzeta = 0;                    // ny: modes (Fourier) or cells between the walls (free-hand)
    double Lx = 0, Ly = 0, Lwin = 0;                  // Ly: period (Fourier) or wall to wall (free-hand)
    bool y_fourier = true;                            // y: Fourier modes (default) or the free-hand grid
    std::vector<double> zeta;                         // slices, zeta[0] = 0 (rear) ... zeta[nzeta-1] = Lwin
    std::vector<double> ynode;                        // y nodes (ny+1). Fourier: j*dy - Ly/2 at the start;
                                                      // free-hand: ynode[0] = -Ly/2 and ynode[ny] = Ly/2 are the walls
    double zeta_change = 0, y_change = 0;             // largest change of the step from one cell to the next
    double dz = 0, z_end = 0, dz_push = 0;
    double sponge = 0, sponge_y = 0, sponge_rate = 0.1;
    bool move_x = true, move_y = true;                // move_y: Fourier y only
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
    // grid lasers (laser blocks with type = gaussian): paraxial Gaussian pulses on the grid
    struct GridLaser {
        double a0 = 1, kL = 1, waist = 10, tau = 10, zeta_c = 0, z_focus = 0, cep = 0, pol_angle = 0;
        bool circular = false; int helicity = 1;
    };
    std::vector<GridLaser> glasers;
    // plasma (code units): n = n0 f(z) g(r) for r < r_max; f: table (z_points, z_values; constant beyond its
    // ends; 1 without a table); g: 1 (uniform) or 1 + channel_depth r^2 / r_channel^2 (parabolic)
    bool plasma_on = false;
    double n0 = 0, r_max = 1e300, channel_depth = 0, r_channel = 1, smooth_length = 0;
    std::vector<double> pz_points, pz_values;
    bool parabolic = false;
    int ppc_x = 2, ppc_y = 2;
    // streaming particles (as in GEM-PIC-RZ): passes per slice, trapping and return to streaming
    int iterations = 2;
    double vz_min = 0.8, vz_beam = 0.9, trap_advance = 1.0;
    bool predictor_extrapolate = true;                // streaming current of half-cell n-1: s0 + extrapolated change
    double plasma_density(double x, double y, double z) const;   // x, y relative to the axis (lab)
    std::string dir = "out";
    int every = 10;
    bool dump_particles = true;
    bool frames = false;                               // output.frames: frame_NNNNN.bin every output step (GUI)

    static Config from_deck(const Deck& d);
    static Config base_units(double lambda0);          // units only (used by the tests)
    void summary(FILE* f, int ranks) const;
    // grids (code units): uniform, or a free-hand step table (points >= 0; for y measured from the axis,
    // mirrored, walls at +-y_max rounded up to a whole cell; for zeta from the rear, window rounded up)
    void zeta_uniform(int n, double Lw);
    void zeta_table(const std::vector<double>& pts, const std::vector<double>& steps, double Lw);
    void y_uniform(int n, double Ly_);                 // Fourier modes
    void y_table(const std::vector<double>& pts, const std::vector<double>& steps, double ymax);
    double dzeta(int m) const { return zeta[m + 1] - zeta[m]; }
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

// w: number of real particles (code units of n_c / k0^3), q: charge number (-1 electron), qm = q / mass (in m_e),
// z: lab position (beam list: the plane z = s after a push; streaming particles move in z within a level),
// sp: species (0 bunch, 1 trapped plasma electron, 2 plasma electron streaming).
struct Particle { double x, y, zeta, px, py, pz, w, rem, z, q, qm; int alive, sp; };
// beam list, alive: 0 lost, 1 own, 2 handed to the rank below during the push (rem = z still to go), 3 adopted,
// 4 slower than numerics.vz_min: leaves the beam list and streams in the current march from its (z, zeta)
// streaming list, alive: 1 streaming, 0 lost, 5 trapped (reached the next level), 6 waiting (joins the march at
// its zeta)
enum Species { SP_BUNCH = 0, SP_TRAPPED = 1, SP_PLASMA = 2 };

class Simulation {
public:
    Simulation(const Config& c, Comm& comm);
    void init();
    void step();
    bool finished() const { return s >= cfg.z_end - 1e-9 || stop_; }
    void diagnostics(FILE* f, bool header);        // collective; f is used on rank 0 only
    void dump_particles(const std::string& file);  // collective
    void write_frame(const std::string& file);     // collective: fields in the planes y = 0 and x = 0 (GUI)

    // fields of level s1 (march with D = dz/2), or E_z, B_z, L of the stored R (D = 0)
    void solve_fields(double D);

    // decomposition: this rank owns the slices [a, b) and the particles of the half-cells [a+1, b]
    // (rank 0 from half-cell 0)
    Config cfg;
    Comm& comm;
    Laser laser;
    int N = 0, a = 0, b = 0, ns = 0, nk = 0;          // nk = nx * ny points of a plane, index j*nx + ix
    double dx = 0, dy = 0, s = 0, X0 = 0, Y0 = 0;     // dy, Y0: Fourier y only (free-hand y: Y0 = 0)
    bool yf = true;                                    // cfg.y_fourier
    long istep = 0;
    double field_energy = 0, W_macro_ne = 0;
    double beta_b = 1, gm2_b = 0;           // mean beta of the bunch at the start, 1 - beta^2 (bound field of the sponge)

    // device arrays of nk-point planes: slice planes l = n - a (0..ns-1); half-cell planes hk = m - (a-1)
    // (0..ns); unfiltered current planes hw = m - (a-2) (0..ns+2); real-field planes fl = n - a (0..ns+1:
    // own slices and the ghost slices b, b+1 of the rank above)
    Buf<cpx> Rx, Ry, Ez, Bz, Lx, Ly;        // ns planes
    Buf<cpx> bnd;                           // slice b of level s1 from the rank above: Rx Ry Ez Bz Lx Ly
    Buf<cpx> j0x, j0y, jkx, jky, jkz;       // filtered spectra of the half-cell currents, levels s0 and s1
    Buf<cpx> jzg;                           // filtered j_z of the half-cells a-2 and b (from the neighbours)
    Buf<double> jw;                         // unfiltered deposit, 3 components x (ns+3) planes
    Buf<float> F;                           // E, B in real space: F[(c*(ns+2) + fl)*nk + i]
    Buf<cpx> S1, S2;                        // FFT scratch, ns+1 planes each
    Buf<cpx> sRx, sRy, sLx, sLy, srhs;      // march scratch of the free-hand y: one plane (a y-line per k_x)
    Buf<double> scp;
    // x: k_x per ix (Nyquist zeroed for derivatives), k_x for the window shift, current filter, sponge
    Buf<double> kxd, kxs, filt, maskx;
    // Fourier y: the same per iy (kyd, kys, filty); masky_n = masky_h is the sponge, evl = kyd^2, iD = iA = 1/dy.
    // Free-hand y: node positions (ny+1), 1/Delta_j of the half-cell j between nodes j and j+1, 1/A_j of the dual
    // cell of node j (between the half-nodes j-1/2 and j+1/2), sponge profiles on nodes and half-nodes, the LU
    // factors of k_x^2 - Lap_y on the nodes (bound field), the eigenvectors of -Lap_y (co-moving field)
    Buf<double> kyd, kys, filty;
    Buf<double> yn, iD, iA, masky_n, masky_h, pcp, pid, evf, evb, evl;
    // zeta: slices, half-cell lengths (guarded: hcg[m+2]), the weights of the half-cells below and above a slice
    Buf<double> zs, hcg, wlo, whi, wsl;
    Axis zax, yax;
    std::vector<double> hDel, hA;           // host copies of Delta_j, A_j
    Buf<Particle> parts;
    long np = 0;                            // particles in `parts`
    long lost_total = 0;
    // plasma: streaming particles of the current march, the streaming current of level s0 per half-cell
    // (spectra; the prediction of the next step), the current of one half-cell (real space, spectra), scratch
    Buf<Particle> strm, strm_bk;
    long nstrm = 0, fresh_last = 0, trapped_total = 0, restreamed_total = 0;
    Buf<cpx> js0x, js0y, js0z, jsn, jsin, bk, sp3, jpred, jtmp, jsi0, dlt;
    Buf<double> jsr;                         // jx, jy, jz, rho of one half-cell (real space)
    Buf<double> rho_lines;                   // electron charge density of the streaming particles on the lines
                                             // y = 0 (nx) and x = 0 (ny) of every half-cell [N][nx + ny]
    std::vector<Particle> joining_;          // join the next march at their zeta (box coordinates)
    std::vector<double> cfg_wlo_;            // host copy of wlo

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
    void setup_plasma();
    void load_plasma(double s0, double s1, const std::vector<Particle>& joining);
    void stream_spectra();
    void march_slice(int n, double D);
    void real_fields_slice(int n);
    void march_plasma(double s0, double s1);
    void collect_restreaming();
    void add_grid_lasers();

private:
    std::unique_ptr<FftMany> fft_;
    std::deque<std::pair<int, std::vector<double>>> cent_;   // pending centroid reductions (handle, sums)
    void setup_y();
    void setup_y_fourier();
    void exchange_jz_ghosts();
    void setup_zeta();
    bool stop_ = false;
};
