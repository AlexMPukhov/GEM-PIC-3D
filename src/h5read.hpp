// ============================================================================
//  h5read.hpp -- small read-only HDF5 helper for the importers
//  (from GEM-PIC-RZ; here: laser_file.cpp, laser envelopes)
//
//  Only compiled with GEMPIC_HDF5. All errors throw std::runtime_error
//  with the file name in front of the message.
// ============================================================================
#pragma once
#if defined(GEMPIC_HDF5)
#include <hdf5.h>

#include <algorithm>
#include <complex>
#include <cstdlib>
#include <stdexcept>
#include <string>
#include <vector>

namespace gem {
namespace h5r {

struct H {  // owning handle
  hid_t id;
  herr_t (*close)(hid_t);
  H(hid_t i, herr_t (*c)(hid_t)) : id(i), close(c) {}
  H(const H&) = delete;
  H& operator=(const H&) = delete;
  ~H() { if (id >= 0) close(id); }
  operator hid_t() const { return id; }
};

class File {
 public:
  explicit File(const std::string& p) : path(p), f(H5Fopen(p.c_str(), H5F_ACC_RDONLY, H5P_DEFAULT), H5Fclose) {
    if (f.id < 0) fail("cannot open the file (not HDF5?)");
  }
  std::string path;
  H f;

  [[noreturn]] void fail(const std::string& m) const { throw std::runtime_error(path + ": " + m); }

  bool exists(const std::string& p) const {
    // check every level (H5Lexists fails on a missing parent)
    size_t q = 1;
    for (;;) {
      q = p.find('/', q);
      const std::string sub = p.substr(0, q);
      if (H5Lexists(f, sub.c_str(), H5P_DEFAULT) <= 0) return false;
      if (q == std::string::npos) return true;
      ++q;
    }
  }
  bool isGroup(const std::string& p) const {
    H5O_info_t info;
#if H5_VERSION_GE(1, 12, 0)
    if (H5Oget_info_by_name3(f, p.c_str(), &info, H5O_INFO_BASIC, H5P_DEFAULT) < 0) return false;
#else
    if (H5Oget_info_by_name(f, p.c_str(), &info, H5P_DEFAULT) < 0) return false;
#endif
    return info.type == H5O_TYPE_GROUP;
  }
  bool hasAttr(const std::string& obj, const char* name) const {
    return H5Aexists_by_name(f, obj.c_str(), name, H5P_DEFAULT) > 0;
  }
  // numeric attribute (any integer / float type, scalar or array)
  std::vector<double> attrD(const std::string& obj, const char* name) const {
    H a(H5Aopen_by_name(f, obj.c_str(), name, H5P_DEFAULT, H5P_DEFAULT), H5Aclose);
    if (a.id < 0) fail("missing attribute " + obj + "@" + name);
    H sp(H5Aget_space(a), H5Sclose);
    std::vector<double> v(size_t(std::max<hssize_t>(1, H5Sget_simple_extent_npoints(sp))));
    if (H5Aread(a, H5T_NATIVE_DOUBLE, v.data()) < 0) fail("cannot read attribute " + obj + "@" + name);
    return v;
  }
  double attr1(const std::string& obj, const char* name, double def) const {
    return hasAttr(obj, name) ? attrD(obj, name)[0] : def;
  }
  // numeric attribute that may be real or complex (compound {r, i})
  std::vector<std::complex<double>> attrC(const std::string& obj, const char* name) const {
    H a(H5Aopen_by_name(f, obj.c_str(), name, H5P_DEFAULT, H5P_DEFAULT), H5Aclose);
    if (a.id < 0) fail("missing attribute " + obj + "@" + name);
    H t(H5Aget_type(a), H5Tclose);
    H sp(H5Aget_space(a), H5Sclose);
    const size_t n = size_t(std::max<hssize_t>(1, H5Sget_simple_extent_npoints(sp)));
    std::vector<std::complex<double>> v(n);
    if (H5Tget_class(t) == H5T_COMPOUND) {
      H mt(complexType(), H5Tclose);
      if (H5Aread(a, mt, v.data()) < 0) fail("cannot read complex attribute " + obj + "@" + name);
    } else {
      std::vector<double> d(n);
      if (H5Aread(a, H5T_NATIVE_DOUBLE, d.data()) < 0) fail("cannot read attribute " + obj + "@" + name);
      for (size_t k = 0; k < n; ++k) v[k] = d[k];
    }
    return v;
  }
  std::string attrS(const std::string& obj, const char* name) const {
    const std::vector<std::string> v = attrSV(obj, name);
    return v.empty() ? std::string() : v[0];
  }
  // string attribute, scalar or array, fixed or variable length
  std::vector<std::string> attrSV(const std::string& obj, const char* name) const {
    std::vector<std::string> out;
    H a(H5Aopen_by_name(f, obj.c_str(), name, H5P_DEFAULT, H5P_DEFAULT), H5Aclose);
    if (a.id < 0) return out;
    H t(H5Aget_type(a), H5Tclose);
    if (H5Tget_class(t) != H5T_STRING) return out;
    H sp(H5Aget_space(a), H5Sclose);
    const size_t n = size_t(std::max<hssize_t>(1, H5Sget_simple_extent_npoints(sp)));
    if (H5Tis_variable_str(t) > 0) {
      std::vector<char*> s(n, nullptr);
      H mt(H5Tcopy(H5T_C_S1), H5Tclose);
      H5Tset_size(mt, H5T_VARIABLE);
      if (H5Aread(a, mt, s.data()) < 0) return out;
      for (size_t k = 0; k < n; ++k) {
        out.push_back(s[k] ? std::string(s[k]) : std::string());
        if (s[k]) H5free_memory(s[k]);
      }
      return out;
    }
    const size_t len = H5Tget_size(t);
    std::string buf(len * n, '\0');
    if (H5Aread(a, t, &buf[0]) < 0) return out;
    for (size_t k = 0; k < n; ++k) out.push_back(std::string(buf.data() + k * len, len).c_str());
    return out;
  }
  std::vector<hsize_t> dims(const std::string& p) const {
    H d(H5Dopen2(f, p.c_str(), H5P_DEFAULT), H5Dclose);
    if (d.id < 0) fail("no dataset " + p);
    H sp(H5Dget_space(d), H5Sclose);
    std::vector<hsize_t> v(size_t(std::max(0, H5Sget_simple_extent_ndims(sp))));
    H5Sget_simple_extent_dims(sp, v.data(), nullptr);
    return v;
  }
  std::vector<double> data(const std::string& p) const {
    H d(H5Dopen2(f, p.c_str(), H5P_DEFAULT), H5Dclose);
    if (d.id < 0) fail("no dataset " + p);
    H sp(H5Dget_space(d), H5Sclose);
    std::vector<double> v(size_t(H5Sget_simple_extent_npoints(sp)));
    if (!v.empty() && H5Dread(d, H5T_NATIVE_DOUBLE, H5S_ALL, H5S_ALL, H5P_DEFAULT, v.data()) < 0)
      fail("cannot read " + p);
    return v;
  }
  // real or complex (compound {r, i}) dataset as complex numbers
  std::vector<std::complex<double>> dataC(const std::string& p) const {
    H d(H5Dopen2(f, p.c_str(), H5P_DEFAULT), H5Dclose);
    if (d.id < 0) fail("no dataset " + p);
    H sp(H5Dget_space(d), H5Sclose);
    H t(H5Dget_type(d), H5Tclose);
    const size_t n = size_t(H5Sget_simple_extent_npoints(sp));
    std::vector<std::complex<double>> v(n);
    if (n == 0) return v;
    if (H5Tget_class(t) == H5T_COMPOUND) {
      H mt(complexType(), H5Tclose);
      if (H5Dread(d, mt, H5S_ALL, H5S_ALL, H5P_DEFAULT, v.data()) < 0)
        fail("cannot read " + p + " as complex numbers (compound {r, i})");
    } else {
      std::vector<double> x(n);
      if (H5Dread(d, H5T_NATIVE_DOUBLE, H5S_ALL, H5S_ALL, H5P_DEFAULT, x.data()) < 0) fail("cannot read " + p);
      for (size_t k = 0; k < n; ++k) v[k] = x[k];
    }
    return v;
  }
  std::vector<std::string> children(const std::string& p) const {
    std::vector<std::string> out;
    H g(H5Gopen2(f, p.c_str(), H5P_DEFAULT), H5Gclose);
    if (g.id < 0) return out;
    H5G_info_t gi;
    H5Gget_info(g, &gi);
    for (hsize_t k = 0; k < gi.nlinks; ++k) {
      char buf[512];
      if (H5Lget_name_by_idx(g, ".", H5_INDEX_NAME, H5_ITER_INC, k, buf, sizeof buf, H5P_DEFAULT) > 0)
        out.push_back(buf);
    }
    return out;
  }
  // openPMD: sorted iteration numbers under the base path, and the chosen one
  // (iteration < 0: the last); fails with the list if it is not there
  long iteration(long it, std::string& itRoot) const {
    std::string base = attrS("/", "basePath");
    if (base.empty()) base = "/data/%T/";
    itRoot = base.substr(0, base.find("%T"));
    std::vector<long> its;
    for (const auto& c : children(itRoot.empty() ? "/" : itRoot)) {
      char* end = nullptr;
      const long v = std::strtol(c.c_str(), &end, 10);
      if (end != c.c_str() && *end == '\0') its.push_back(v);
    }
    if (its.empty()) fail("no iteration found under " + itRoot);
    std::sort(its.begin(), its.end());
    if (it < 0) return its.back();
    if (!std::binary_search(its.begin(), its.end(), it)) {
      std::string l;
      for (long v : its) l += (l.empty() ? "" : ", ") + std::to_string(v);
      fail("iteration " + std::to_string(it) + " not in the file (iterations: " + l + ")");
    }
    return it;
  }

 private:
  static hid_t complexType() {
    hid_t t = H5Tcreate(H5T_COMPOUND, 2 * sizeof(double));
    H5Tinsert(t, "r", 0, H5T_NATIVE_DOUBLE);
    H5Tinsert(t, "i", sizeof(double), H5T_NATIVE_DOUBLE);
    return t;
  }
};

}  // namespace h5r
}  // namespace gem
#endif
