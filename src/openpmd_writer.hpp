// HDF5 helpers of the openPMD writer (src/openpmd.cpp): handles, a dataset/attribute writer, interpolation
// stencils. Included only by openpmd.cpp, inside #ifdef GEMPIC_HDF5.
#pragma once
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
