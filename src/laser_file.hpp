// Laser pulses from LASY / openPMD envelope files (as GEM-PIC-RZ src/laser_file.*).
//
// File format (LASY, https://github.com/LASY-org/lasy; openPMD extension LaserEnvelope): mesh record
// "laserEnvelope" (a scalar, complex dataset) of one iteration, the complex envelope of the transverse electric
// field at a plane z = z_f, as a function of the lab time t at that plane and of (x, y) or (r, theta):
//     E_{x,y}(x, y, t) = Re[ p_{x,y} env(x, y, t) e^{-i omega_L t} ]
// attributes: angularFrequency (omega_L, rad/s), polarization (p_x, p_y, real or complex), envelopeField
// ("electric_field" [V/m] or "normalized_vector_potential"), geometry cartesian (axes t, y, x) or thetaMode
// (axes t, r; components m = 0, cos(m theta), sin(m theta): env = C_0 + sum_m C_m cos(m theta) + S_m sin(m theta)).
//
// Mapping onto GEM-PIC-3D (GT2, as GEM-PIC-RZ): a level s is the plane z = s, its slice zeta the time
// t = s - zeta. The file time t_center is put at zeta_center: at the start level s = 0 the slice zeta holds the
// retarded time tau = t_center + zeta_center - zeta of the file's time base. Between the file's plane and z = 0
// every plane wave (omega, k_x, k_y) propagates exactly in vacuum,
//     k_z = sqrt(omega^2 - k_perp^2),   envelope phase e^{i (k_z - omega)(0 - file_z)} at fixed tau,
// evanescent components (k_perp > omega) are dropped. The forward field of the code is R = 2 E_perp, L = 0.
#pragma once
#include <complex>
#include <string>
#include <vector>

using cdl = std::complex<double>;

// content of an envelope file, converted to code units (lengths 1/k0, times 1/omega0, fields m c omega0 / e)
struct LaserFileData {
    std::string path, mesh, software, field;     // field: envelopeField of the file
    long iteration = -1;
    bool cartesian = false;
    double omegaL = 1;                           // carrier frequency [omega0]
    cdl px = 1.0, py = 0.0;                      // polarization vector (normalized)
    int nt = 0, nr = 0, nx = 0, ny = 0, ncomp = 1;  // ncomp: thetaMode components 2 K + 1
    double t0 = 0, dt = 1, r0 = 0, dr = 1, x0 = 0, dx = 1, y0 = 0, dy = 1;
    std::vector<cdl> env;                        // cartesian [t][y][x]; thetaMode [comp][t][r]
    double peak = 0;                             // largest |env| (all theta)
};

// Reads the mesh `mesh` (empty: "laserEnvelope") of `iteration` (< 0: the last). Lu: code length [m] (1/k0),
// Tu: code time [s] (1/omega0). Needs the HDF5 build (make HDF5=1); throws std::runtime_error otherwise.
LaserFileData read_laser_file(const std::string& path, const std::string& mesh, long iteration, double Lu, double Tu);

// Writes an envelope file in the LASY layout (cartesian or thetaMode as d.cartesian, one iteration 0, dataOrder C);
// d.field = "normalized_vector_potential": env is a, else E in code units. For the tests; HDF5 build only.
void write_laser_file(const std::string& path, const LaserFileData& d, double Lu, double Tu);

// The envelope of a file laser at the start plane z = 0 on a uniform Cartesian grid, interpolated on demand.
class LaserFileField {
public:
    struct Params {
        double scale = 1;          // factor on the envelope (a0 of the deck)
        double zeta_center = 0;    // window position of the file time t_center
        double t_center = 0;       // [1/omega0], file time base
        double file_z = 0;         // plane of the file in the simulation coordinates
        int padding = 2;           // FFT length in t / length of the file's time axis
        int padding_xy = 1;        // the same transversally
    };
    LaserFileField(const LaserFileData& d, const Params& p);
    // E_x, E_y at (x, y) of the slice zeta at z = 0 (code units)
    void field(double x, double y, double zeta, double& ex, double& ey) const;
    double tau(double zeta) const { return P.t_center + P.zeta_center - zeta; }
    bool covers(double zeta) const { const double t = tau(zeta); return t >= t0 && t <= t0 + (nt - 1) * dt; }
    std::vector<std::string> info;   // summary lines
    double edge = 0;                 // largest |env| at the edges of the propagated grid, relative to the peak
    double omegaL = 1;
    cdl px = 1.0, py = 0.0;
    // envelope at z = 0 (without the factor scale), trilinear between the nodes, 0 outside
    cdl env(double x, double y, double t) const;
private:
    Params P;
    int nt = 0, ny = 0, nx = 0;
    double t0 = 0, dt = 1, x0 = 0, dx = 1, y0 = 0, dy = 1;
    std::vector<cdl> A;              // [t][y][x] at z = 0
};

// time of the intensity centroid of the file [1/omega0]
double laser_file_centroid(const LaserFileData& d);
