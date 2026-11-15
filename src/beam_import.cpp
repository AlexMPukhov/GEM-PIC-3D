// ============================================================================
//  beam_import.cpp -- read a particle beam from an openPMD file (from GEM-PIC-RZ)
// ============================================================================
#include "beam_import.hpp"

#include <cmath>
#include <stdexcept>

#if defined(GEMPIC_HDF5)
#include "h5read.hpp"

#include <algorithm>
#include <map>
#include <sstream>

namespace gem {

namespace {

constexpr double QE = 1.602176634e-19, ME = 9.1093837015e-31, CL = 299792458.0;

using h5r::H;

// openPMD record helpers on top of the shared reader
class File : public h5r::File {
 public:
  using h5r::File::File;
  // openPMD record component: dataset or constant (group with "value"), times unitSI
  std::vector<double> component(const std::string& p, size_t n, bool required = true) const {
    if (!exists(p)) {
      if (required) fail("missing record component " + p);
      return std::vector<double>(n, 0.0);
    }
    const double u = attr1(p, "unitSI", 1.0);
    std::vector<double> v;
    if (isGroup(p)) {
      v.assign(n, attrD(p, "value")[0]);
    } else {
      v = data(p);
      if (v.size() != n) fail(p + " has " + std::to_string(v.size()) + " entries, expected " + std::to_string(n));
    }
    for (auto& x : v) x *= u;
    return v;
  }
  // number of particles of a record (dataset size or constant "shape")
  size_t count(const std::string& p) const {
    if (isGroup(p)) return size_t(attrD(p, "shape")[0]);
    H d(H5Dopen2(f, p.c_str(), H5P_DEFAULT), H5Dclose);
    if (d.id < 0) fail("no dataset " + p);
    H sp(H5Dget_space(d), H5Sclose);
    return size_t(H5Sget_simple_extent_npoints(sp));
  }
};

std::string join(const std::vector<std::string>& v) {
  std::string s;
  for (const auto& x : v) s += (s.empty() ? "" : ", ") + x;
  return s.empty() ? "none" : s;
}

ImportedBeam readOpenPMD(const File& F, const std::string& species, long iteration) {
  ImportedBeam B;
  B.format = "openPMD " + F.attrS("/", "openPMD");
  std::string base = F.attrS("/", "basePath");
  if (base.empty()) base = "/data/%T/";
  const std::string itRoot = base.substr(0, base.find("%T"));
  std::vector<long> its;
  for (const auto& c : F.children(itRoot.empty() ? "/" : itRoot)) {
    char* end = nullptr;
    const long v = std::strtol(c.c_str(), &end, 10);
    if (end != c.c_str() && *end == '\0') its.push_back(v);
  }
  if (its.empty()) F.fail("no iteration found under " + itRoot);
  std::sort(its.begin(), its.end());
  if (iteration < 0) iteration = its.back();
  if (!std::binary_search(its.begin(), its.end(), iteration)) {
    std::string l;
    for (long v : its) l += (l.empty() ? "" : ", ") + std::to_string(v);
    F.fail("iteration " + std::to_string(iteration) + " not in the file (iterations: " + l + ")");
  }
  B.iteration = iteration;
  std::string pp = F.attrS("/", "particlesPath");
  if (pp.empty()) pp = "particles/";
  if (pp.back() != '/') pp += '/';
  const std::string ip = itRoot + std::to_string(iteration) + "/" + pp;
  const std::vector<std::string> sps = F.children(ip);
  std::string sp = species;
  if (sp.empty()) {
    // the only species with particles (writers such as GEM-PIC-RZ keep empty species in every iteration)
    std::vector<std::string> full;
    for (const auto& c : sps)
      if (!F.exists(ip + c + "/position/x") || F.count(ip + c + "/position/x") > 0) full.push_back(c);
    if (full.size() != 1) F.fail("give file_species: the file holds the species " + join(sps));
    sp = full[0];
  } else if (std::find(sps.begin(), sps.end(), sp) == sps.end()) {
    F.fail("species '" + sp + "' not found (species: " + join(sps) + ")");
  }
  B.species = sp;
  const std::string g = ip + sp + "/";
  const size_t n = F.count(g + "position/x");
  // charge and mass of one particle: electrons or positrons only
  const double q = F.exists(g + "charge") ? F.component(g + "charge", n)[0] : -QE;
  const double m = F.exists(g + "mass") ? F.component(g + "mass", n)[0] : ME;
  if (std::fabs(std::fabs(q) / QE - 1.0) > 1e-3 || std::fabs(m / ME - 1.0) > 1e-3)
    F.fail("species '" + sp + "' is not electrons or positrons (charge " + std::to_string(q) + " C, mass " +
           std::to_string(m) + " kg)");
  B.charge_sign = q < 0 ? -1.0 : 1.0;
  const std::vector<double> w = F.exists(g + "weighting") ? F.component(g + "weighting", n)
                                                          : std::vector<double>(n, 1.0);
  std::vector<double> pos[3], mom[3];
  const char* ax[3] = {"x", "y", "z"};
  for (int a = 0; a < 3; ++a) {
    pos[a] = F.component(g + "position/" + ax[a], n, a != 1 || F.exists(g + "position/y"));
    const std::vector<double> off = F.component(g + "positionOffset/" + ax[a], n, false);
    for (size_t k = 0; k < n; ++k) pos[a][k] += off[k];
    mom[a] = F.component(g + "momentum/" + ax[a], n, a != 1 || F.exists(g + "momentum/y"));
  }
  // momentum per real particle: macroWeighted = 1 records hold the sum over the macroparticle
  const bool mw = F.hasAttr(g + "momentum", "macroWeighted") && F.attrD(g + "momentum", "macroWeighted")[0] != 0.0;
  const double wp = F.attr1(g + "momentum", "weightingPower", 1.0);
  B.p.resize(n);
  for (size_t k = 0; k < n; ++k) {
    const double f = mw ? std::pow(w[k], -wp) : 1.0;
    B.p[k] = ImportedParticle{pos[0][k], pos[1][k], pos[2][k], mom[0][k] * f / (ME * CL), mom[1][k] * f / (ME * CL),
                              mom[2][k] * f / (ME * CL), w[k]};
  }
  return B;
}

}  // namespace

ImportedBeam importBeam(const std::string& path, const std::string& species, long iteration) {
  H5Eset_auto2(H5E_DEFAULT, nullptr, nullptr);
  File F(path);
  ImportedBeam B;
  if (F.hasAttr("/", "openPMD")) B = readOpenPMD(F, species, iteration);
  else F.fail("not an openPMD file");
  if (B.p.empty()) F.fail("species '" + B.species + "' holds no particles");
  return B;
}

}  // namespace gem

#else

namespace gem {
ImportedBeam importBeam(const std::string& path, const std::string&, long) {
  throw std::runtime_error(path + ": reading beams from files needs the HDF5 build (make HDF5=1)");
}
}  // namespace gem

#endif
