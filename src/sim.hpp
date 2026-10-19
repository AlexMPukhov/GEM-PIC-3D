// GEM-PIC-3D: Galilean electromagnetic PIC in three dimensions.
// Transverse directions: Fourier modes (periodic box with absorbing sponge),
// longitudinal: zeta = z - t marched from the front of the window to the rear,
// slow coordinate s = z (GT2 for the particles). Units: c = 1, lengths 1/k0,
// times 1/omega0, momenta m c, fields m c omega0 / e, densities n_c (of lambda0).
#pragma once
#include <cstdio>
#include <string>
#include <vector>
#include "deck.hpp"
#include "fft.hpp"

struct Config {
    // reference wavelength and units
    double lambda0 = 0;          // [m]
    double Lu = 0, Tu = 0;       // code length [m], code time [s]
    double nc = 0;               // critical density of lambda0 [m^-3]
    // grid
    int nx = 0, ny = 0, nzeta = 0;
    double Lx = 0, Ly = 0, Lwin = 0, h = 0;      // box and slice spacing
    double dz = 0, z_end = 0, dz_push = 0;       // field step, end, largest particle sub-step
    double sponge = 0, sponge_rate = 0.1;        // width of the absorbing layer, damping per step at the edge
    bool move_x = true, move_y = true;           // transverse window follows the bunch centroid
    // numerics
    bool self_fields = true, zeta_filter = true, k_filter = true;
    double push_phase = 0.05;                    // largest laser phase per particle sub-step [rad]
    // laser (analytic plane wave, external field)
    bool laser_on = false;
    double a0 = 0, kL = 1, theta = 0, tau = 1, cep = 0;  // tau: intensity FWHM [code time]
    std::string pol = "s";                               // s | p | circular
    double start_fwhm = 3.5, stop_fwhm = 3.5;
    // bunch
    double charge = 0, gamma = 1, sigx = 0, sigy = 0, sigz = 0, emitx = 0, emity = 0, espread = 0;
    double zeta_center = 0;
    long nmacro = 0;
    std::string noise = "random";
    double noise_lambda = 0; int beamlet = 4; unsigned long seed = 1;
    // output
    std::string dir = "out";
    int every = 10;
    bool dump_particles = true;

    static Config from_deck(const Deck& d);
    void summary(FILE* f) const;
};

// Plane-wave pulse of infinite transverse size: A = a0 g(eta) cos(kL (eta - eta_c) + cep) e_pol,
// eta = t - khat . r, khat = (sin theta, 0, cos theta). E = -dA/deta, B = khat x E.
struct Laser {
    bool on = false;
    double a0 = 0, kL = 1, tau = 1, cep = 0, eta_c = 0;
    double kx = 0, kz = 1;              // components of khat
    int pol = 0;                        // 0 s (E along y), 1 p (E in the x-z plane), 2 circular
    double eta(double x, double z, double t) const { return t - kx * x - kz * z; }
    void field(double x, double z, double t, double E[3], double B[3]) const;
    double envelope(double eta_) const;
};

struct Particle { double x, y, zeta, px, py, pz, w; int alive; };

class Simulation {
public:
    explicit Simulation(const Config& c);
    void init();                 // load the bunch, its co-moving field, level s = 0
    void step();                 // s -> s + dz
    bool finished() const;
    void diagnostics(FILE* f, bool header);
    void dump_particles(const std::string& file) const;

    // state (public for the tests)
    Config cfg;
    Laser laser;
    int N = 0, NH = 0, nk = 0;   // slices, half-cells, transverse points
    double dx = 0, dy = 0, s = 0, X0 = 0, Y0 = 0;
    long istep = 0, lost = 0;
    std::vector<Particle> parts;
    std::vector<double> kxd, kyd, kxs, kys, filt, mask;  // derivative and shift wavenumbers, current filter, sponge mask
    std::vector<cplx> Rx, Ry, Ez0, Bz0, j0x, j0y;       // spectra: level s0 (N slices / NH half-cells)
    std::vector<float> F;                                // real E, B of level s0: F[(c*N + n)*nk + i]
    std::vector<double> jr;                              // deposited currents: jr[(c*NH + m)*nk + i]
    double field_energy = 0;                             // code units
    double W_macro_ne = 0;                               // electrons per macroparticle (weight 1)

    void push(double s0, double ds);
    void deposit();
    void march(double D);
    void comoving_field();
    void shift_window();
    void gather(double x, double y, double zeta, double E[3], double B[3]) const;
    double particle_eta(const Particle& p) const;
    void load_bunch();
private:
    FFT2 fft_;
};
