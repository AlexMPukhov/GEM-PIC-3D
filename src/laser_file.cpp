// Laser pulses from LASY / openPMD envelope files (see laser_file.hpp).
#include "laser_file.hpp"
#include "fft.hpp"
#include "h5read.hpp"
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <stdexcept>

namespace {
constexpr double kC = 299792458.0, kE = 1.602176634e-19, kMe = 9.1093837015e-31;
const cdl I1(0.0, 1.0);
std::string fmt(const char* f, double a, double b = 0, double c = 0, double d = 0) {
    char buf[256];
    std::snprintf(buf, sizeof buf, f, a, b, c, d);
    return buf;
}
}  // namespace

double laser_file_centroid(const LaserFileData& d) {
    double s0 = 0, s1 = 0;
    const size_t inner = d.cartesian ? size_t(d.nx) * d.ny : size_t(d.nr);
    const int outer = d.cartesian ? 1 : d.ncomp;
    for (int c = 0; c < outer; ++c)
        for (int l = 0; l < d.nt; ++l)
            for (size_t q = 0; q < inner; ++q) {
                // thetaMode: weight r (and 1/2 for the cos/sin components, their mean over theta)
                double w = d.cartesian ? 1.0 : (d.r0 + double(q) * d.dr) * (c == 0 ? 1.0 : 0.5);
                const double e = std::norm(d.env[(size_t(c) * d.nt + l) * inner + q]) * w;
                s0 += e; s1 += e * (d.t0 + l * d.dt);
            }
    return s0 > 0 ? s1 / s0 : d.t0 + 0.5 * (d.nt - 1) * d.dt;
}

LaserFileData read_laser_file(const std::string& path, const std::string& mesh, long iteration, double Lu, double Tu) {
#if !defined(GEMPIC_HDF5)
    (void)mesh; (void)iteration; (void)Lu; (void)Tu;
    throw std::runtime_error(path + ": reading laser files needs the HDF5 build (make HDF5=1)");
#else
    using namespace gem;
    H5Eset_auto2(H5E_DEFAULT, nullptr, nullptr);
    h5r::File F(path);
    LaserFileData D;
    D.path = path;
    D.mesh = mesh.empty() ? "laserEnvelope" : mesh;
    if (!F.hasAttr("/", "openPMD")) F.fail("not an openPMD file (no attribute 'openPMD')");
    std::string itRoot;
    D.iteration = F.iteration(iteration, itRoot);
    std::string mp = F.attrS("/", "meshesPath");
    if (mp.empty()) mp = "meshes/";
    if (mp.back() != '/') mp += '/';
    const std::string mroot = itRoot + std::to_string(D.iteration) + "/" + mp;
    const std::string p = mroot + D.mesh;
    if (!F.exists(p)) {
        std::string l;
        for (const auto& c : F.children(mroot)) l += (l.empty() ? "" : ", ") + c;
        F.fail("no mesh '" + D.mesh + "' in iteration " + std::to_string(D.iteration) + " (meshes: " +
               (l.empty() ? "none" : l) + "); a LASY envelope is 'laserEnvelope', else give file_mesh");
    }
    if (F.isGroup(p)) F.fail("mesh '" + D.mesh + "' is not a scalar record (an envelope is one complex dataset)");
    D.software = F.attrS("/", "software");
    const std::string geo = F.attrS(p, "geometry");
    if (geo == "cartesian") D.cartesian = true;
    else if (geo != "thetaMode") F.fail("mesh '" + D.mesh + "': geometry '" + geo + "' (need thetaMode or cartesian)");
    std::vector<std::string> lab = F.attrSV(p, "axisLabels");
    std::vector<double> sp = F.attrD(p, "gridSpacing"), off = F.attrD(p, "gridGlobalOffset");
    std::vector<double> pos(lab.size(), 0.0);
    if (F.hasAttr(p, "position")) pos = F.attrD(p, "position");
    if (F.attrS(p, "dataOrder") == "F") {
        std::reverse(lab.begin(), lab.end()); std::reverse(sp.begin(), sp.end());
        std::reverse(off.begin(), off.end()); std::reverse(pos.begin(), pos.end());
    }
    const size_t na = D.cartesian ? 3 : 2;
    if (lab.size() != na || sp.size() != na || off.size() != na || pos.size() != na)
        F.fail("mesh '" + D.mesh + "': expected " + std::to_string(na) + " axes (axisLabels, gridSpacing, gridGlobalOffset)");
    auto axis = [&](const char* name) {
        for (size_t k = 0; k < lab.size(); ++k)
            if (lab[k] == name) return int(k);
        if (std::string(name) == "t")
            F.fail("mesh '" + D.mesh + "': no time axis 't'; a field given in z must be converted to t first");
        F.fail(std::string("mesh '") + D.mesh + "': no axis '" + name + "'");
        return -1;
    };
    const double gu = F.attr1(p, "gridUnitSI", 1.0);
    const double E0 = kMe * kC * kC / (kE * Lu);       // m c omega0 / e [V/m]
    if (!F.hasAttr(p, "angularFrequency")) F.fail("mesh '" + D.mesh + "' has no attribute angularFrequency");
    D.omegaL = F.attrD(p, "angularFrequency")[0] * Tu;
    if (!(D.omegaL > 0)) F.fail("angularFrequency must be positive");
    if (F.hasAttr(p, "polarization")) {
        const auto v = F.attrC(p, "polarization");
        if (v.size() != 2) F.fail("polarization must have 2 entries");
        D.px = v[0]; D.py = v[1];
    }
    const double pn = std::sqrt(std::norm(D.px) + std::norm(D.py));
    if (!(pn > 0)) F.fail("polarization vector is zero");
    D.px /= pn; D.py /= pn;
    D.field = F.hasAttr(p, "envelopeField") ? F.attrS(p, "envelopeField") : "normalized_vector_potential";
    const bool vecpot = D.field == "normalized_vector_potential";
    if (!vecpot && D.field != "electric_field")
        F.fail("envelopeField '" + D.field + "' (need electric_field or normalized_vector_potential)");
    const double uSI = F.attr1(p, "unitSI", 1.0);

    const std::vector<hsize_t> dims = F.dims(p);
    const std::vector<cdl> raw = F.dataC(p);
    const int it = axis("t");
    if (D.cartesian) {
        if (dims.size() != 3) F.fail("cartesian envelope: dataset must have 3 dimensions");
        const int iy = axis("y"), ix = axis("x");
        D.nt = int(dims[it]); D.ny = int(dims[iy]); D.nx = int(dims[ix]);
        D.t0 = (off[it] + pos[it] * sp[it]) * gu / Tu; D.dt = sp[it] * gu / Tu;
        D.y0 = (off[iy] + pos[iy] * sp[iy]) * gu / Lu; D.dy = sp[iy] * gu / Lu;
        D.x0 = (off[ix] + pos[ix] * sp[ix]) * gu / Lu; D.dx = sp[ix] * gu / Lu;
        const size_t st[3] = {size_t(dims[1] * dims[2]), size_t(dims[2]), 1};
        D.env.resize(size_t(D.nt) * D.ny * D.nx);
        for (int l = 0; l < D.nt; ++l)
            for (int y = 0; y < D.ny; ++y)
                for (int x = 0; x < D.nx; ++x)
                    D.env[(size_t(l) * D.ny + y) * D.nx + x] = raw[l * st[it] + y * st[iy] + x * st[ix]];
    } else {
        if (dims.size() != 3) F.fail("thetaMode envelope: dataset must have 3 dimensions (components, 2 axes)");
        const int ir = axis("r");
        D.ncomp = int(dims[0]);
        if (D.ncomp % 2 != 1) F.fail("thetaMode envelope: the number of components must be odd (2 K + 1)");
        D.nt = int(dims[1 + it]); D.nr = int(dims[1 + ir]);
        D.t0 = (off[it] + pos[it] * sp[it]) * gu / Tu; D.dt = sp[it] * gu / Tu;
        D.r0 = (off[ir] + pos[ir] * sp[ir]) * gu / Lu; D.dr = sp[ir] * gu / Lu;
        if (D.r0 < -1e-9 * D.dr) F.fail("thetaMode envelope: the radial axis starts below r = 0");
        D.r0 = std::max(0.0, D.r0);
        const size_t st[2] = {size_t(dims[2]), 1};
        D.env.resize(size_t(D.ncomp) * D.nt * D.nr);
        for (int c = 0; c < D.ncomp; ++c)
            for (int l = 0; l < D.nt; ++l)
                for (int j = 0; j < D.nr; ++j)
                    D.env[(size_t(c) * D.nt + l) * D.nr + j] = raw[c * dims[1] * dims[2] + l * st[it] + j * st[ir]];
    }
    if (D.nt < 4 || !(D.dt > 0)) F.fail("the time axis needs at least 4 points and a positive spacing");
    // units: E [V/m] -> code; normalized vector potential: E = i omegaL a - da/dt
    if (!vecpot) {
        for (auto& v : D.env) v *= uSI / E0;
    } else {
        const size_t inner = D.cartesian ? size_t(D.ny) * D.nx : size_t(D.nr);
        const int outer = D.cartesian ? 1 : D.ncomp;
        std::vector<cdl> col(D.nt);
        for (int c = 0; c < outer; ++c)
            for (size_t q = 0; q < inner; ++q) {
                const size_t base = size_t(c) * D.nt * inner + q;
                for (int l = 0; l < D.nt; ++l) col[l] = D.env[base + l * inner] * uSI;
                for (int l = 0; l < D.nt; ++l) {
                    const int la = std::max(0, l - 1), lb = std::min(D.nt - 1, l + 1);
                    D.env[base + l * inner] = I1 * D.omegaL * col[l] - (col[lb] - col[la]) / ((lb - la) * D.dt);
                }
            }
    }
    if (D.cartesian) {
        for (const auto& v : D.env) D.peak = std::max(D.peak, std::abs(v));
    } else {
        const int Kf = (D.ncomp - 1) / 2, nth = 4 * Kf + 4;
        for (int l = 0; l < D.nt; ++l)
            for (int j = 0; j < D.nr; ++j)
                for (int q = 0; q < nth; ++q) {
                    const double th = 2.0 * M_PI * q / nth;
                    cdl e = D.env[size_t(l) * D.nr + j];
                    for (int m = 1; m <= Kf; ++m)
                        e += D.env[(size_t(2 * m - 1) * D.nt + l) * D.nr + j] * std::cos(m * th) +
                             D.env[(size_t(2 * m) * D.nt + l) * D.nr + j] * std::sin(m * th);
                    D.peak = std::max(D.peak, std::abs(e));
                }
    }
    if (!(D.peak > 0)) F.fail("the envelope is zero everywhere");
    return D;
#endif
}

void write_laser_file(const std::string& path, const LaserFileData& d, double Lu, double Tu) {
#if !defined(GEMPIC_HDF5)
    (void)d; (void)Lu; (void)Tu;
    throw std::runtime_error(path + ": writing laser files needs the HDF5 build (make HDF5=1)");
#else
    auto ck = [&](long r, const char* what) { if (r < 0) throw std::runtime_error(path + ": HDF5 error: " + what); return r; };
    const hid_t f = hid_t(ck(H5Fcreate(path.c_str(), H5F_ACC_TRUNC, H5P_DEFAULT, H5P_DEFAULT), "create"));
    auto attrS = [&](hid_t o, const char* n, const std::string& v) {
        const hid_t t = H5Tcopy(H5T_C_S1); H5Tset_size(t, v.size() + 1); H5Tset_strpad(t, H5T_STR_NULLTERM);
        const hid_t s = H5Screate(H5S_SCALAR);
        const hid_t a = hid_t(ck(H5Acreate2(o, n, t, s, H5P_DEFAULT, H5P_DEFAULT), n));
        ck(H5Awrite(a, t, v.c_str()), n); H5Aclose(a); H5Sclose(s); H5Tclose(t);
    };
    auto attrSV = [&](hid_t o, const char* n, const std::vector<std::string>& v) {
        size_t len = 1;
        for (const auto& x : v) len = std::max(len, x.size() + 1);
        std::string buf(len * v.size(), '\0');
        for (size_t k = 0; k < v.size(); ++k) std::copy(v[k].begin(), v[k].end(), buf.begin() + long(k * len));
        const hid_t t = H5Tcopy(H5T_C_S1); H5Tset_size(t, len); H5Tset_strpad(t, H5T_STR_NULLTERM);
        const hsize_t nn = v.size();
        const hid_t s = H5Screate_simple(1, &nn, nullptr);
        const hid_t a = hid_t(ck(H5Acreate2(o, n, t, s, H5P_DEFAULT, H5P_DEFAULT), n));
        ck(H5Awrite(a, t, buf.data()), n); H5Aclose(a); H5Sclose(s); H5Tclose(t);
    };
    auto attrD = [&](hid_t o, const char* n, const std::vector<double>& v) {
        const hsize_t nn = v.size();
        const hid_t s = H5Screate_simple(1, &nn, nullptr);
        const hid_t a = hid_t(ck(H5Acreate2(o, n, H5T_NATIVE_DOUBLE, s, H5P_DEFAULT, H5P_DEFAULT), n));
        ck(H5Awrite(a, H5T_NATIVE_DOUBLE, v.data()), n); H5Aclose(a); H5Sclose(s);
    };
    const hid_t ct = H5Tcreate(H5T_COMPOUND, 2 * sizeof(double));
    H5Tinsert(ct, "r", 0, H5T_NATIVE_DOUBLE); H5Tinsert(ct, "i", sizeof(double), H5T_NATIVE_DOUBLE);
    const hid_t root = H5Gopen2(f, "/", H5P_DEFAULT);
    attrS(root, "openPMD", "1.1.0"); attrS(root, "openPMDextension", "LaserEnvelope");
    attrS(root, "basePath", "/data/%T/"); attrS(root, "meshesPath", "meshes/");
    attrS(root, "software", "GEM-PIC-3D");
    H5Gclose(root);
    const hid_t lcpl = H5Pcreate(H5P_LINK_CREATE);
    H5Pset_create_intermediate_group(lcpl, 1);
    const hid_t g = hid_t(ck(H5Gcreate2(f, "/data/0/meshes", lcpl, H5P_DEFAULT, H5P_DEFAULT), "groups"));
    H5Pclose(lcpl);
    const double E0 = kMe * kC * kC / (kE * Lu);
    std::vector<hsize_t> dims;
    if (d.cartesian) dims = {hsize_t(d.nt), hsize_t(d.ny), hsize_t(d.nx)};
    else dims = {hsize_t(d.ncomp), hsize_t(d.nt), hsize_t(d.nr)};
    const hid_t s = H5Screate_simple(3, dims.data(), nullptr);
    const hid_t ds = hid_t(ck(H5Dcreate2(g, "laserEnvelope", ct, s, H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT), "dataset"));
    ck(H5Dwrite(ds, ct, H5S_ALL, H5S_ALL, H5P_DEFAULT, d.env.data()), "write");
    attrS(ds, "geometry", d.cartesian ? "cartesian" : "thetaMode");
    attrS(ds, "dataOrder", "C");
    const bool vecpot = d.field == "normalized_vector_potential";   // env: a (dimensionless), else E in code units
    attrS(ds, "envelopeField", vecpot ? "normalized_vector_potential" : "electric_field");
    attrD(ds, "unitSI", {vecpot ? 1.0 : E0});
    attrD(ds, "gridUnitSI", {1.0});
    attrD(ds, "angularFrequency", {d.omegaL / Tu});
    {
        const hsize_t two = 2;
        const hid_t ps = H5Screate_simple(1, &two, nullptr);
        const hid_t a = hid_t(ck(H5Acreate2(ds, "polarization", ct, ps, H5P_DEFAULT, H5P_DEFAULT), "polarization"));
        const cdl pv[2] = {d.px, d.py};
        ck(H5Awrite(a, ct, pv), "polarization"); H5Aclose(a); H5Sclose(ps);
    }
    if (d.cartesian) {
        attrSV(ds, "axisLabels", {"t", "y", "x"});
        attrD(ds, "gridSpacing", {d.dt * Tu, d.dy * Lu, d.dx * Lu});
        attrD(ds, "gridGlobalOffset", {d.t0 * Tu, d.y0 * Lu, d.x0 * Lu});
        attrD(ds, "position", {0, 0, 0});
    } else {
        attrSV(ds, "axisLabels", {"t", "r"});
        attrD(ds, "gridSpacing", {d.dt * Tu, d.dr * Lu});
        attrD(ds, "gridGlobalOffset", {d.t0 * Tu, d.r0 * Lu});
        attrD(ds, "position", {0, 0});
    }
    H5Dclose(ds); H5Sclose(s); H5Gclose(g); H5Tclose(ct); H5Fclose(f);
#endif
}

// ---------------------------------------------------------------- field at the start plane
LaserFileField::LaserFileField(const LaserFileData& d, const Params& p) : omegaL(d.omegaL), px(d.px), py(d.py), P(p) {
    const bool prop = std::fabs(p.file_z) > 1e-12;
    const int pt = prop ? std::max(1, p.padding) : 1, pxy = prop ? std::max(1, p.padding_xy) : 1;
    // transverse grid of the envelope: the file's (cartesian) or one with the radial step (thetaMode)
    int fx, fy;
    double fx0, fy0, fdx, fdy;
    if (d.cartesian) { fx = d.nx; fy = d.ny; fx0 = d.x0; fy0 = d.y0; fdx = d.dx; fdy = d.dy; }
    else {
        const double rmax = d.r0 + (d.nr - 1) * d.dr;
        const int h = int(std::ceil(rmax / d.dr - 1e-9));
        fx = fy = 2 * h + 1; fdx = fdy = d.dr; fx0 = fy0 = -h * d.dr;
    }
    nx = fx * pxy; ny = fy * pxy; nt = d.nt * pt;
    dx = fdx; dy = fdy; dt = d.dt; t0 = d.t0;
    x0 = fx0 - (nx - fx) / 2 * dx; y0 = fy0 - (ny - fy) / 2 * dy;
    A.assign(size_t(nt) * ny * nx, cdl(0));
    const int ox = (nx - fx) / 2, oy = (ny - fy) / 2;
    const int Kf = d.cartesian ? 0 : (d.ncomp - 1) / 2;
    for (int l = 0; l < d.nt; ++l)
        for (int j = 0; j < fy; ++j)
            for (int i = 0; i < fx; ++i) {
                cdl v;
                if (d.cartesian) v = d.env[(size_t(l) * d.ny + j) * d.nx + i];
                else {
                    const double xx = fx0 + i * fdx, yy = fy0 + j * fdy, r = std::sqrt(xx * xx + yy * yy);
                    const double u = (r - d.r0) / d.dr;
                    if (u > d.nr - 1 + 1e-9) continue;
                    const int k = std::max(0, std::min(d.nr - 2, int(std::floor(u))));
                    const double w = std::max(0.0, std::min(1.0, u - k));
                    auto at = [&](int c) { const size_t b = (size_t(c) * d.nt + l) * d.nr; return (1 - w) * d.env[b + k] + w * d.env[b + k + 1]; };
                    const double th = std::atan2(yy, xx);
                    v = at(0);
                    for (int m = 1; m <= Kf; ++m) v += at(2 * m - 1) * std::cos(m * th) + at(2 * m) * std::sin(m * th);
                }
                A[(size_t(l) * ny + j + oy) * nx + i + ox] = v;
            }
    info.push_back(fmt("envelope grid %.0f x %.0f x %.0f (t, y, x)", nt, ny, nx) +
                   fmt(", dt = %.4g, dy = %.4g, dx = %.4g (code units)", dt, dy, dx));
    if (prop) {
        // exact propagation of every (omega, k_x, k_y) from file_z to z = 0 at fixed retarded time
        const double Dz = -p.file_z;
        FFT1 Ft(nt), Fy(ny), Fx(nx);
        auto pass = [&](bool inv) {
            #pragma omp parallel
            {
                std::vector<cplx> col(std::max(nt, ny));
                #pragma omp for
                for (long q = 0; q < long(nt) * ny; ++q) Fx.run(&A[size_t(q) * nx], inv);
                #pragma omp for
                for (long q = 0; q < long(nt) * nx; ++q) {
                    const long l = q / nx, i = q % nx;
                    for (int j = 0; j < ny; ++j) col[j] = A[(size_t(l) * ny + j) * nx + i];
                    Fy.run(col.data(), inv);
                    for (int j = 0; j < ny; ++j) A[(size_t(l) * ny + j) * nx + i] = col[j];
                }
                #pragma omp for
                for (long q = 0; q < long(ny) * nx; ++q) {
                    for (int l = 0; l < nt; ++l) col[l] = A[size_t(l) * ny * nx + q];
                    Ft.run(col.data(), inv);
                    for (int l = 0; l < nt; ++l) A[size_t(l) * ny * nx + q] = col[l];
                }
            }
        };
        pass(false);
        const double norm = 1.0 / (double(nt) * ny * nx);
        double etot = 0, edrop = 0;
        #pragma omp parallel for reduction(+:etot,edrop)
        for (long l = 0; l < nt; ++l) {
            const int ql = int(l < (nt + 1) / 2 ? l : l - nt);
            const double om = omegaL - 2 * M_PI * ql / (nt * dt);
            for (int j = 0; j < ny; ++j) {
                const int qj = j < (ny + 1) / 2 ? j : j - ny;
                const double ky = 2 * M_PI * qj / (ny * dy);
                for (int i = 0; i < nx; ++i) {
                    const int qi = i < (nx + 1) / 2 ? i : i - nx;
                    const double kx = 2 * M_PI * qi / (nx * dx), k2 = kx * kx + ky * ky;
                    cdl& a = A[(size_t(l) * ny + j) * nx + i];
                    const double e = std::norm(a);
                    etot += e;
                    if (om > 0 && om * om > k2) {
                        const double kz = std::sqrt(om * om - k2);
                        a *= std::polar(norm, (kz - om) * Dz);
                    } else { edrop += e; a = 0; }
                }
            }
        }
        pass(true);
        info.push_back(fmt("propagated exactly from z = %.6g to 0 (code units); %.3g of the spectral energy evanescent or at omega <= 0, dropped",
                           p.file_z, etot > 0 ? edrop / etot : 0.0));
    }
    double pk = 0, pe = 0;
    for (int l = 0; l < nt; ++l)
        for (int j = 0; j < ny; ++j)
            for (int i = 0; i < nx; ++i) {
                const double v = std::abs(A[(size_t(l) * ny + j) * nx + i]);
                pk = std::max(pk, v);
                if (l == 0 || l == nt - 1 || j == 0 || j == ny - 1 || i == 0 || i == nx - 1) pe = std::max(pe, v);
            }
    edge = pk > 0 ? pe / pk : 0;
}

cdl LaserFileField::env(double x, double y, double t) const {
    const double ux = (x - x0) / dx, uy = (y - y0) / dy, ut = (t - t0) / dt;
    if (!(ux >= 0 && ux <= nx - 1 && uy >= 0 && uy <= ny - 1 && ut >= 0 && ut <= nt - 1)) return 0;
    const int i = std::min(nx - 2, int(ux)), j = std::min(ny - 2, int(uy)), l = std::min(nt - 2, int(ut));
    const double wx = ux - i, wy = uy - j, wt = ut - l;
    cdl v = 0;
    for (int c = 0; c < 8; ++c) {
        const int a = c & 1, b = c >> 1 & 1, e = c >> 2;
        const double w = (a ? wx : 1 - wx) * (b ? wy : 1 - wy) * (e ? wt : 1 - wt);
        if (w != 0) v += w * A[(size_t(l + e) * ny + j + b) * nx + i + a];
    }
    return v;
}

void LaserFileField::field(double x, double y, double zeta, double& ex, double& ey) const {
    const double t = tau(zeta);
    const cdl E = P.scale * env(x, y, t) * std::polar(1.0, -omegaL * t);
    ex = (px * E).real(); ey = (py * E).real();
}
