// ============================================================================
//  openpmd.cpp -- openPMD 1.1.0 output of the output groups (format = openpmd or both; build with make HDF5=1),
//  as in GEM-PIC-RZ; the groups are described in docs/output_spec.md
//
//  file-based series, basePath /data/%T/ (T = step), one file per output step of a group:
//    <group dir>/<openpmd_file>.h5     default openpmd/data_%06T: the selected fields and/or beams
//
//  meshes   E, B (components x, y, z), n_e (plasma electrons streaming through the window, with plasma):
//           geometry cartesian, data [zeta][y][x] (C order, axisLabels z, y, x), SI via unitSI.
//           Uniform grids (Fourier y, uniform zeta): the code's own nodes, exact. Free-hand y or zeta:
//           records interpolated to a uniform grid (openpmd_dy, openpmd_dzeta of the group; default the
//           smallest step) and/or native records native_E_x, ... on the code's own staggered nodes, with the node
//           positions in <basePath>gridCoordinates/... (the rectilinear-mesh proposal for openPMD 2.0 that
//           GEM-PIC-RZ and QUARZ use; geometry "other", geometryParameters "cartesian").
//  coordinates  a level s is shown at the time t = s/c with z = s + zeta (gridGlobalOffset z = s). In the
//           GT2 frame of the code the data of a level lie on the lab plane z = s at the times t = s - zeta.
//           x, y: lab positions (the transverse window offset is in gridGlobalOffset).
//  particles  "bunch" and "trapped" (plasma electrons in the beam list): position (z = zeta, positionOffset
//           z = s), momentum per real particle, weighting (real particles per macroparticle), constant charge
//           and mass, one particle patch with the bounding box.
//  The fields are those of the end of the step (after the sponge), as used by the next push.
// ============================================================================
#include "sim.hpp"

#ifdef GEMPIC_HDF5
#include <hdf5.h>
#include <algorithm>
#include <cmath>
#include <ctime>
#include <filesystem>
#include <stdexcept>
#include <string>
#include <memory>
#include <type_traits>
#include <vector>

namespace {
inline bool is_half(int c) { return c == 1 || c == 3 || c == 5; }   // E_y, B_x, B_z: half-nodes of the free-hand y
void check(herr_t e, const std::string& what) {
    if (e < 0) throw std::runtime_error("HDF5 error: " + what);
}
class Handle {
public:
    Handle(hid_t id, herr_t (*closer)(hid_t), const std::string& what) : id_(id), close_(closer) {
        if (id < 0) throw std::runtime_error("HDF5 error: " + what);
    }
    Handle(const Handle&) = delete;
    Handle& operator=(const Handle&) = delete;
    ~Handle() { if (id_ >= 0) close_(id_); }
    operator hid_t() const { return id_; }
private:
    hid_t id_;
    herr_t (*close_)(hid_t);
};

class Writer {
public:
    Writer(const std::string& path, bool single, int level) : file_(create(path), H5Fclose, "cannot create " + path), single_(single), level_(level) {}
    hid_t file() const { return file_; }
    static void attr(hid_t obj, const char* name, double v) {
        Handle sp(H5Screate(H5S_SCALAR), H5Sclose, "dataspace");
        Handle a(H5Acreate2(obj, name, H5T_IEEE_F64LE, sp, H5P_DEFAULT, H5P_DEFAULT), H5Aclose, name);
        check(H5Awrite(a, H5T_NATIVE_DOUBLE, &v), name);
    }
    static void attrU32(hid_t obj, const char* name, unsigned v) {
        Handle sp(H5Screate(H5S_SCALAR), H5Sclose, "dataspace");
        Handle a(H5Acreate2(obj, name, H5T_STD_U32LE, sp, H5P_DEFAULT, H5P_DEFAULT), H5Aclose, name);
        check(H5Awrite(a, H5T_NATIVE_UINT, &v), name);
    }
    static void attr(hid_t obj, const char* name, const std::string& v) {
        Handle t(H5Tcopy(H5T_C_S1), H5Tclose, "string type");
        check(H5Tset_size(t, v.empty() ? 1 : v.size()), "string size");
        check(H5Tset_strpad(t, H5T_STR_NULLTERM), "string pad");
        Handle sp(H5Screate(H5S_SCALAR), H5Sclose, "dataspace");
        Handle a(H5Acreate2(obj, name, t, sp, H5P_DEFAULT, H5P_DEFAULT), H5Aclose, name);
        const std::string s = v.empty() ? std::string(1, '\0') : v;
        check(H5Awrite(a, t, s.c_str()), name);
    }
    static void attr(hid_t obj, const char* name, const std::vector<double>& v) {
        const hsize_t n = v.size();
        Handle sp(H5Screate_simple(1, &n, nullptr), H5Sclose, "dataspace");
        Handle a(H5Acreate2(obj, name, H5T_IEEE_F64LE, sp, H5P_DEFAULT, H5P_DEFAULT), H5Aclose, name);
        check(H5Awrite(a, H5T_NATIVE_DOUBLE, v.data()), name);
    }
    static void attr(hid_t obj, const char* name, const std::vector<unsigned long long>& v) {
        const hsize_t n = v.size();
        Handle sp(H5Screate_simple(1, &n, nullptr), H5Sclose, "dataspace");
        Handle a(H5Acreate2(obj, name, H5T_STD_U64LE, sp, H5P_DEFAULT, H5P_DEFAULT), H5Aclose, name);
        check(H5Awrite(a, H5T_NATIVE_ULLONG, v.data()), name);
    }
    static void attr(hid_t obj, const char* name, const std::vector<std::string>& v) {   // fixed-length strings
        size_t len = 1;
        for (const auto& x : v) len = std::max(len, x.size() + 1);
        std::string buf(len * v.size(), '\0');
        for (size_t k = 0; k < v.size(); ++k) std::copy(v[k].begin(), v[k].end(), buf.begin() + long(k * len));
        Handle t(H5Tcopy(H5T_C_S1), H5Tclose, "string type");
        check(H5Tset_size(t, len), "string size");
        check(H5Tset_strpad(t, H5T_STR_NULLTERM), "string pad");
        const hsize_t n = v.size();
        Handle sp(H5Screate_simple(1, &n, nullptr), H5Sclose, "dataspace");
        Handle a(H5Acreate2(obj, name, t, sp, H5P_DEFAULT, H5P_DEFAULT), H5Aclose, name);
        check(H5Awrite(a, t, buf.data()), name);
    }
    hid_t groups(const std::string& path) {   // with all parents, returned open
        size_t p = 1;
        for (;;) {
            p = path.find('/', p);
            const std::string sub = path.substr(0, p);
            if (H5Lexists(file_, sub.c_str(), H5P_DEFAULT) <= 0) {
                Handle g(H5Gcreate2(file_, sub.c_str(), H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT), H5Gclose, "group " + sub);
            }
            if (p == std::string::npos) break;
            ++p;
        }
        return H5Gopen2(file_, path.c_str(), H5P_DEFAULT);
    }
    // float data (fields, particles); forceDouble: coordinates and scalars always in double precision
    template <class T> hid_t data(const std::string& path, const std::vector<hsize_t>& dims, const T* v, bool forceDouble = false) {
        const hid_t ft = (single_ && !forceDouble) ? H5T_IEEE_F32LE : H5T_IEEE_F64LE;
        const hid_t mt = std::is_same<T, float>::value ? H5T_NATIVE_FLOAT : H5T_NATIVE_DOUBLE;
        hid_t ds = create(path, dims, ft);
        if (ds < 0) throw std::runtime_error("HDF5 error: dataset " + path);
        check(H5Dwrite(ds, mt, H5S_ALL, H5S_ALL, H5P_DEFAULT, v), "write " + path);
        return ds;
    }
    void u64(const std::string& path, const unsigned long long v) {
        const hsize_t n = 1;
        Handle sp(H5Screate_simple(1, &n, nullptr), H5Sclose, "dataspace");
        Handle ds(H5Dcreate2(file_, path.c_str(), H5T_STD_U64LE, sp, H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT), H5Dclose, path);
        check(H5Dwrite(ds, H5T_NATIVE_ULLONG, H5S_ALL, H5S_ALL, H5P_DEFAULT, &v), "write " + path);
    }
    void text(const std::string& path, const std::string& s) {
        Handle t(H5Tcopy(H5T_C_S1), H5Tclose, "string type");
        check(H5Tset_size(t, s.empty() ? 1 : s.size()), "string size");
        Handle sp(H5Screate(H5S_SCALAR), H5Sclose, "dataspace");
        Handle ds(H5Dcreate2(file_, path.c_str(), t, sp, H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT), H5Dclose, path);
        const std::string v = s.empty() ? std::string(1, '\0') : s;
        check(H5Dwrite(ds, t, H5S_ALL, H5S_ALL, H5P_DEFAULT, v.c_str()), "write " + path);
    }
private:
    Handle file_;
    bool single_;
    int level_;
    static hid_t create(const std::string& path) {
        const hid_t fapl = H5Pcreate(H5P_FILE_ACCESS);     // HDF5 >= 1.8 format: attributes larger than 64 kB
        H5Pset_libver_bounds(fapl, H5F_LIBVER_V18, H5F_LIBVER_LATEST);
        const hid_t f = H5Fcreate(path.c_str(), H5F_ACC_TRUNC, H5P_DEFAULT, fapl);
        H5Pclose(fapl);
        return f;
    }
    hid_t create(const std::string& path, const std::vector<hsize_t>& dims, hid_t type) {
        Handle sp(H5Screate_simple(int(dims.size()), dims.data(), nullptr), H5Sclose, "dataspace " + path);
        Handle pl(H5Pcreate(H5P_DATASET_CREATE), H5Pclose, "property list");
        bool empty = false;
        for (hsize_t d : dims) empty = empty || d == 0;
        if (level_ > 0 && !empty) {     // chunks: one slice of a field, or 65536 particles
            std::vector<hsize_t> ch(dims);
            if (ch.size() == 1) ch[0] = std::min<hsize_t>(ch[0], 65536);
            else ch[0] = 1;
            check(H5Pset_chunk(pl, int(ch.size()), ch.data()), "chunk " + path);
            check(H5Pset_shuffle(pl), "shuffle " + path);
            check(H5Pset_deflate(pl, unsigned(level_)), "deflate " + path);
        }
        return H5Dcreate2(file_, path.c_str(), type, sp, H5P_DEFAULT, pl, H5P_DEFAULT);
    }
};

// linear interpolation weights from the (increasing) nodes x to the points q, clamped at the ends
void stencil(const std::vector<double>& x, const std::vector<double>& q, std::vector<int>& i0, std::vector<double>& w) {
    i0.resize(q.size()); w.resize(q.size());
    for (size_t k = 0; k < q.size(); ++k) {
        if (x.size() < 2 || q[k] <= x.front()) { i0[k] = 0; w[k] = 0; continue; }
        if (q[k] >= x.back()) { i0[k] = int(x.size()) - 2; w[k] = 1; continue; }
        const int j = int(std::upper_bound(x.begin(), x.end(), q[k]) - x.begin()) - 1;
        i0[k] = j; w[k] = (q[k] - x[j]) / (x[j + 1] - x[j]);
    }
}
}  // namespace

void Simulation::write_openpmd(const Config::OutGroup& g, bool fields, bool beams) {
    const int comps = fields ? g.fields : 0;
    const bool particles = beams && g.beams;
    namespace fs = std::filesystem;
    const int nx = cfg.nx, ny = cfg.ny;
    const long nkk = nk, fpl = long(ns + 2) * nk;
    // ---- gather the components to rank 0: [N][ny][nx] (box order: x index from X0, y index from Y0)
    const bool want_ne = (comps >> 6 & 1) && cfg.plasma_on && rho_full.size();
    std::vector<std::vector<float>> full(7);
    const std::vector<float> Fh = cfg.self_fields ? F.host() : std::vector<float>(F.size(), 0.0f);
    for (int c = 0; c < 6; ++c) {
        if (!(comps >> c & 1)) continue;
        std::vector<char> all = comm.gather_bytes(Fh.data() + c * fpl, size_t(ns) * nkk * sizeof(float));
        if (comm.rank == 0) {
            const float* A = reinterpret_cast<const float*>(all.data());
            full[c].assign(A, A + size_t(N) * nkk);
        }
    }
    std::vector<float> rh;
    if (want_ne && comm.rank == 0) rh = rho_full.host();      // plasma: one rank
    // ---- particles to rank 0
    std::vector<Particle> P;
    if (particles) {
        std::vector<Particle> mine;
        for (const auto& p : host_particles()) if (p.alive == 1) mine.push_back(p);
        std::vector<char> all = comm.gather_bytes(mine.data(), mine.size() * sizeof(Particle));
        if (comm.rank == 0) {
            const Particle* A = reinterpret_cast<const Particle*>(all.data());
            P.assign(A, A + all.size() / sizeof(Particle));
        }
    }
    if (comm.rank != 0) return;

    // file from the pattern openpmd_file (%T or %0<w>T = the step), relative to the group directory
    std::string pat = g.openpmd_file, fname;
    {
        const size_t pc = pat.find('%');
        const size_t te = pat.find('T', pc);
        int width = 0;
        if (pc + 1 < te) width = std::stoi(pat.substr(pc + 1, te - pc - 1));
        char num[32];
        std::snprintf(num, sizeof num, "%0*ld", width, istep);
        fname = pat.substr(0, pc) + num + pat.substr(te + 1);
    }
    const std::string name = g.dir + "/" + fname + ".h5";
    fs::create_directories(fs::path(name).parent_path());
    const std::string iterfmt = fs::path(pat).filename().string() + ".h5";
    H5Eset_auto2(H5E_DEFAULT, nullptr, nullptr);
    auto Wp = std::make_unique<Writer>(name + ".tmp", g.hdf5_single, g.hdf5_compression);
    Writer& W = *Wp;
    const double L = cfg.Lu;                                     // [m]
    const double E0 = 9.1093837015e-31 * 2.99792458e8 * 2.99792458e8 / (1.602176634e-19 * L);   // m c omega0 / e [V/m]
    const double B0 = E0 / 2.99792458e8;                         // [T]
    const double T0 = L / 2.99792458e8;                          // [s]
    const double ME = 9.1093837015e-31, QE = 1.602176634e-19, C0 = 2.99792458e8;
    auto unitDim = [](double l, double m, double t, double i) { return std::vector<double>{l, m, t, i, 0, 0, 0}; };

    const hid_t root = W.file();
    Writer::attr(root, "openPMD", std::string("1.1.0"));
    Writer::attrU32(root, "openPMDextension", 0u);
    Writer::attr(root, "basePath", std::string("/data/%T/"));
    if (comps) Writer::attr(root, "meshesPath", std::string("meshes/"));
    if (particles) Writer::attr(root, "particlesPath", std::string("particles/"));
    Writer::attr(root, "iterationEncoding", std::string("fileBased"));
    Writer::attr(root, "iterationFormat", iterfmt);
    Writer::attr(root, "software", std::string("GEM-PIC-3D"));
    Writer::attr(root, "softwareVersion", std::string("1"));
    Writer::attr(root, "author", std::string("GEM-PIC-3D user"));
    {
        char buf[64];
        const std::time_t t = std::time(nullptr);
        std::strftime(buf, sizeof buf, "%Y-%m-%d %H:%M:%S %z", std::localtime(&t));
        Writer::attr(root, "date", std::string(buf));
    }
    Writer::attr(root, "comment", std::string("GEM-PIC-3D (GT2 frame, s = z): a level s is shown at t = s/c with z = s + zeta; "
                                              "its data lie on the lab plane z = s at the times t = s - zeta. Deck in /data/<T>/input_deck"));
    const std::string base = "/data/" + std::to_string(istep);
    {
        Handle it(W.groups(base), H5Gclose, base);
        Writer::attr(it, "time", s);
        Writer::attr(it, "dt", cfg.dz);
        Writer::attr(it, "timeUnitSI", T0);
    }
    W.text(base + "/input_deck", cfg.deck_text);

    // ---- grids: the code's nodes per component, and the uniform output grid
    std::vector<double> xs(nx), yn(ny), yh(ny), zs(N), zh(N - 1);
    for (int i = 0; i < nx; ++i) xs[i] = i * dx;                        // relative to X0
    for (int j = 0; j < ny; ++j) {
        yn[j] = yf ? j * dy : cfg.ynode[j];                            // Fourier: relative to Y0
        yh[j] = yf ? j * dy : 0.5 * (cfg.ynode[j] + cfg.ynode[j + 1]);
    }
    for (int n = 0; n < N; ++n) zs[n] = cfg.zeta[n];
    for (int n = 0; n + 1 < N; ++n) zh[n] = 0.5 * (cfg.zeta[n] + cfg.zeta[n + 1]);
    // uniform zeta: equal steps to rounding (zeta_uniform sets zeta_n = n h, whose differences need not be equal bit
    // for bit); the records are then the slices themselves
    bool zu = zax.uniform;
    if (!zu && N > 2) {
        zu = true;
        const double h0 = zs[1] - zs[0];
        for (int n = 1; n + 1 < N && zu; ++n) zu = std::fabs(zs[n + 1] - zs[n] - h0) <= 1e-9 * h0;
    }
    const bool uniform_grid = yf && zu;
    const double yoff = yf ? Y0 : 0.0;
    const double dyo = g.openpmd_dy > 0 ? g.openpmd_dy : (yf ? dy : *std::min_element(hDel.begin(), hDel.end()));
    double dzmin = 1e300;
    for (int n = 0; n + 1 < N; ++n) dzmin = std::min(dzmin, zs[n + 1] - zs[n]);
    const int xs_ = g.xi_stride;
    const double dzo = (g.openpmd_dzeta > 0 ? g.openpmd_dzeta : dzmin) * xs_;
    // uniform output points: Fourier y: the nodes; free-hand y: -Ly/2 ... Ly/2 step dyo
    std::vector<double> yo, zo;
    if (yf) yo = yn;
    else for (int j = 0; ; ++j) { const double y = -0.5 * cfg.Ly + j * dyo; if (y > 0.5 * cfg.Ly + 1e-9 * dyo) break; yo.push_back(y); }
    if (zu) for (int n = 0; n < N; n += xs_) zo.push_back(zs[n]);
    else for (int k = 0; ; ++k) { const double z = k * dzo; if (z > zs.back() + 1e-9 * dzo) break; zo.push_back(z); }
    const std::vector<std::string> axes = {"z", "y", "x"};
    auto meshAttrs = [&](hid_t obj, const std::vector<double>& dim) {
        Writer::attr(obj, "geometry", std::string("cartesian"));
        Writer::attr(obj, "dataOrder", std::string("C"));
        Writer::attr(obj, "axisLabels", axes);
        Writer::attr(obj, "gridSpacing", std::vector<double>{zu ? (N > 1 ? (zs[1] - zs[0]) * xs_ : 1.0) : dzo, yf ? dy : dyo, dx});
        Writer::attr(obj, "gridGlobalOffset", std::vector<double>{s + zo.front(), yoff + yo.front(), X0});
        Writer::attr(obj, "gridUnitSI", L);
        Writer::attr(obj, "unitDimension", dim);
        Writer::attr(obj, "timeOffset", 0.0);
    };
    const std::string mp = base + "/meshes";
    if (comps) { Handle h(W.groups(mp), H5Gclose, mp); }
    // value of component c (or n_e: c = 6) at code slice n, code row j, column i
    auto value = [&](int c, int n, int j, int i) -> double {
        if (c < 6) return full[c][(size_t(n) * ny + j) * nx + i];
        // n_e / at slices: mean of the adjacent half-cells, -rho in units of n_c
        const double lo = n > 0 ? rh[(size_t(n - 1) * ny + j) * nx + i] : 0.0, hi = n < N - 1 ? rh[(size_t(n) * ny + j) * nx + i] : 0.0;
        return -0.5 * (lo + hi);
    };
    const char* cname[3] = {"x", "y", "z"};
    std::vector<float> buf;
    if (comps && (g.openpmd_uniform || uniform_grid)) {
        // interpolation (identity on uniform grids)
        std::vector<int> iz, iyn, iyh; std::vector<double> wz, wyn, wyh;
        stencil(zs, zo, iz, wz); stencil(yn, yo, iyn, wyn); stencil(yh, yo, iyh, wyh);
        auto field = [&](int c) {
            const bool half = !yf && c < 6 && is_half(c);
            const std::vector<int>& iy = half ? iyh : iyn;
            const std::vector<double>& wy = half ? wyh : wyn;
            buf.assign(zo.size() * yo.size() * nx, 0.0f);
            for (size_t k = 0; k < zo.size(); ++k)
                for (size_t q = 0; q < yo.size(); ++q)
                    for (int i = 0; i < nx; ++i) {
                        double v;
                        if (uniform_grid) v = value(c, int(k) * xs_, int(q), i);
                        else {
                            const int n = iz[k], j = iy[q];
                            const double a = wz[k], b = wy[q];
                            const int n1 = std::min(n + 1, N - 1), j1 = std::min(j + 1, ny - 1);
                            v = (1 - a) * ((1 - b) * value(c, n, j, i) + b * value(c, n, j1, i)) +
                                a * ((1 - b) * value(c, n1, j, i) + b * value(c, n1, j1, i));
                        }
                        buf[(k * yo.size() + q) * nx + i] = float(v);
                    }
        };
        const std::vector<hsize_t> dims = {hsize_t(zo.size()), hsize_t(yo.size()), hsize_t(nx)};
        for (int vec = 0; vec < 2; ++vec) {
            if (!(comps >> (3 * vec) & 7)) continue;
            const std::string rec = mp + (vec == 0 ? "/E" : "/B");
            {
                Handle h(W.groups(rec), H5Gclose, rec);
                meshAttrs(h, vec == 0 ? unitDim(1, 1, -3, -1) : unitDim(0, 1, -2, -1));
            }
            for (int k = 0; k < 3; ++k) {
                const int c = 3 * vec + k;
                if (!(comps >> c & 1)) continue;
                field(c);
                Handle ds(W.data(rec + "/" + cname[k], dims, buf.data()), H5Dclose, rec);
                Writer::attr(ds, "unitSI", vec == 0 ? E0 : B0);
                Writer::attr(ds, "position", std::vector<double>{0.0, 0.0, 0.0});
            }
        }
        if (want_ne) {
            field(6);
            Handle ds(W.data(mp + "/n_e", dims, buf.data()), H5Dclose, "n_e");
            meshAttrs(ds, unitDim(-3, 0, 0, 0));
            Writer::attr(ds, "unitSI", cfg.nc);
            Writer::attr(ds, "position", std::vector<double>{0.0, 0.0, 0.0});
        }
    }
    if (comps && g.openpmd_native && !uniform_grid) {
        // native records on the code's own nodes: E_x, E_z, B_y on the y nodes, E_y, B_x, B_z on the half-nodes
        // (free-hand y), the fields on the slices and n_e on the half-cells
        const std::string gc = base + "/gridCoordinates";
        { Handle h(W.groups(gc), H5Gclose, gc); }
        auto coords = [&](const char* nm, const std::vector<double>& v) {
            Handle h(W.data(gc + "/" + nm, {hsize_t(v.size())}, v.data(), true), H5Dclose, nm);
            return gc + "/" + nm;
        };
        std::vector<double> zss, zhs;
        for (int n = 0; n < N; n += xs_) zss.push_back(zs[n]);
        for (int n = 0; n + 1 < N; n += xs_) zhs.push_back(zh[n]);
        const std::string pZs = coords("zeta_slices", zss), pZh = coords("zeta_halfcells", zhs);
        std::vector<double> ynl(yn), yhl(yh);
        const std::string pYn = coords(yf ? "y" : "y_nodes", ynl);
        const std::string pYh = yf ? pYn : coords("y_halfnodes", yhl);
        auto minStep = [](const std::vector<double>& x) {
            double d = 1e300;
            for (size_t k = 1; k < x.size(); ++k) d = std::min(d, x[k] - x[k - 1]);
            return x.size() > 1 ? d : 1.0;
        };
        auto native = [&](const std::string& nm, int c, const std::vector<double>& dim, double unitSI) {
            const bool half = !yf && c < 6 && is_half(c);
            const std::vector<double>& zz = c < 6 ? zss : zhs;
            const std::vector<double>& yy = half ? yhl : ynl;
            buf.assign(zz.size() * ny * nx, 0.0f);
            for (size_t n = 0; n < zz.size(); ++n)
                for (int j = 0; j < ny; ++j)
                    for (int i = 0; i < nx; ++i)
                        buf[(n * ny + j) * nx + i] = c < 6 ? full[c][(n * xs_ * ny + j) * nx + i] : -rh[(n * xs_ * ny + j) * nx + i];
            Handle ds(W.data(mp + "/" + nm, {hsize_t(zz.size()), hsize_t(ny), hsize_t(nx)}, buf.data()), H5Dclose, nm);
            Writer::attr(ds, "geometry", std::string("other"));
            Writer::attr(ds, "geometryParameters", std::string("cartesian"));
            Writer::attr(ds, "gridType", std::string("rectilinear"));
            Writer::attr(ds, "dataOrder", std::string("C"));
            Writer::attr(ds, "axisLabels", axes);
            Writer::attr(ds, "gridSpacing", std::vector<double>{minStep(zz), minStep(yy), dx});
            Writer::attr(ds, "gridGlobalOffset", std::vector<double>{s, yoff, X0});
            Writer::attr(ds, "gridUnitSI", L);
            Writer::attr(ds, "gridCoordinates", std::vector<std::string>{c < 6 ? pZs : pZh, half ? pYh : pYn, ""});
            Writer::attr(ds, "unitDimension", dim);
            Writer::attr(ds, "timeOffset", 0.0);
            Writer::attr(ds, "position", std::vector<double>{0.0, 0.0, 0.0});
            Writer::attr(ds, "unitSI", unitSI);
        };
        const char* nn[6] = {"native_E_x", "native_E_y", "native_E_z", "native_B_x", "native_B_y", "native_B_z"};
        for (int c = 0; c < 6; ++c)
            if (comps >> c & 1) native(nn[c], c, c < 3 ? unitDim(1, 1, -3, -1) : unitDim(0, 1, -2, -1), c < 3 ? E0 : B0);
        if (want_ne) native("native_n_e", 6, unitDim(-3, 0, 0, 0), cfg.nc);
    }

    // ---- particles
    if (particles) {
        const std::string pp = base + "/particles";
        { Handle h(W.groups(pp), H5Gclose, pp); }
        const double Wsi = cfg.nc * L * L * L;                  // real particles per unit of w
        for (int sp = 0; sp < 2; ++sp) {
            std::vector<const Particle*> ps;
            if (!(g.beams >> sp & 1)) continue;
            long cnt = 0;
            for (const auto& p : P) if ((sp == 0) == (p.sp == SP_BUNCH) && cnt++ % g.particle_stride == 0) ps.push_back(&p);
            if (sp == 1 && ps.empty()) continue;
            const std::string sg = pp + (sp == 0 ? "/bunch" : "/trapped");
            { Handle h(W.groups(sg), H5Gclose, sg); }
            const hsize_t np = ps.size();
            std::vector<double> col(np);
            auto record = [&](const std::string& rec, const std::vector<double>& dim, unsigned mw, double wp) {
                Handle h(W.groups(sg + "/" + rec), H5Gclose, rec);
                Writer::attr(h, "unitDimension", dim);
                Writer::attr(h, "timeOffset", 0.0);
                Writer::attrU32(h, "macroWeighted", mw);
                Writer::attr(h, "weightingPower", wp);
            };
            auto component = [&](const std::string& path, double (*get)(const Particle&, double, double), double unitSI) {
                for (size_t k = 0; k < np; ++k) col[k] = get(*ps[k], X0, Y0);
                Handle ds(W.data(sg + "/" + path, {np}, col.data()), H5Dclose, path);
                Writer::attr(ds, "unitSI", unitSI);
            };
            auto constant = [&](const std::string& path, double value, double unitSI) {
                Handle h(W.groups(sg + "/" + path), H5Gclose, path);
                Writer::attr(h, "value", value);
                Writer::attr(h, "shape", std::vector<unsigned long long>{np});
                Writer::attr(h, "unitSI", unitSI);
            };
            record("position", unitDim(1, 0, 0, 0), 0u, 0.0);
            component("position/x", [](const Particle& p, double x0, double) { return p.x + x0; }, L);
            component("position/y", [](const Particle& p, double, double y0) { return p.y + y0; }, L);
            component("position/z", [](const Particle& p, double, double) { return p.zeta; }, L);
            record("positionOffset", unitDim(1, 0, 0, 0), 0u, 0.0);
            constant("positionOffset/x", 0.0, L);
            constant("positionOffset/y", 0.0, L);
            constant("positionOffset/z", s, L);
            record("momentum", unitDim(1, 1, -1, 0), 0u, 1.0);
            component("momentum/x", [](const Particle& p, double, double) { return p.px; }, ME * C0);
            component("momentum/y", [](const Particle& p, double, double) { return p.py; }, ME * C0);
            component("momentum/z", [](const Particle& p, double, double) { return p.pz; }, ME * C0);
            {
                for (size_t k = 0; k < np; ++k) col[k] = ps[k]->w * Wsi;
                Handle ds(W.data(sg + "/weighting", {np}, col.data(), true), H5Dclose, "weighting");
                Writer::attr(ds, "unitDimension", unitDim(0, 0, 0, 0));
                Writer::attr(ds, "timeOffset", 0.0);
                Writer::attrU32(ds, "macroWeighted", 1u);
                Writer::attr(ds, "weightingPower", 1.0);
                Writer::attr(ds, "unitSI", 1.0);
            }
            auto constRecord = [&](const char* nm, const std::vector<double>& dim, double v) {
                Handle h(W.groups(sg + "/" + nm), H5Gclose, nm);
                Writer::attr(h, "unitDimension", dim);
                Writer::attr(h, "timeOffset", 0.0);
                Writer::attrU32(h, "macroWeighted", 0u);
                Writer::attr(h, "weightingPower", 1.0);
                Writer::attr(h, "value", v);
                Writer::attr(h, "shape", std::vector<unsigned long long>{np});
                Writer::attr(h, "unitSI", 1.0);
            };
            constRecord("charge", unitDim(0, 0, 1, 1), -QE);
            constRecord("mass", unitDim(0, 1, 0, 0), ME);
            {   // one particle patch with the bounding box (lab z = zeta + s)
                double lo[3] = {0, 0, 0}, hi[3] = {0, 0, 0};
                for (size_t k = 0; k < np; ++k) {
                    const double v[3] = {ps[k]->x + X0, ps[k]->y + Y0, ps[k]->zeta + s};
                    for (int a = 0; a < 3; ++a) {
                        if (k == 0 || v[a] < lo[a]) lo[a] = v[a];
                        if (k == 0 || v[a] > hi[a]) hi[a] = v[a];
                    }
                }
                const std::string pg = sg + "/particlePatches";
                { Handle h(W.groups(pg), H5Gclose, pg); }
                W.u64(pg + "/numParticles", np);
                W.u64(pg + "/numParticlesOffset", 0);
                for (int which = 0; which < 2; ++which) {
                    const std::string rec = pg + (which == 0 ? "/offset" : "/extent");
                    {
                        Handle h(W.groups(rec), H5Gclose, rec);
                        Writer::attr(h, "unitDimension", unitDim(1, 0, 0, 0));
                    }
                    for (int a = 0; a < 3; ++a) {
                        const double v = which == 0 ? lo[a] : hi[a] - lo[a];
                        Handle ds(W.data(rec + "/" + cname[a], {1}, &v, true), H5Dclose, rec);
                        Writer::attr(ds, "unitSI", L);
                    }
                }
            }
        }
    }
    Wp.reset();                                     // close the file before it gets its final name
    fs::rename(name + ".tmp", name);
}
#else
void Simulation::write_openpmd(const Config::OutGroup&, bool, bool) {
    throw std::runtime_error("openPMD output needs HDF5: build with make HDF5=1");
}
#endif
