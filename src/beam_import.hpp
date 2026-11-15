// ============================================================================
//  beam_import.hpp -- read a particle beam from a file (deck: bunch.file)
//
//  Formats (detected from the file):
//    openPMD 1.x (HDF5, any code: GEM-PIC-RZ, FBPIC, HiPACE++, WarpX, ...):
//        /data/<iteration>/<particlesPath>/<species>/ position, positionOffset,
//        momentum, weighting, charge, mass (datasets or constant records)
//    (ported from GEM-PIC-RZ, src/beam_import.*; its native HDF5 format is not read here)
//  The particles are returned in SI-free form: positions in metres (lab z),
//  momenta u = p / (m_e c), weight = number of real particles, and the sign
//  of the charge. Only electrons and positrons are accepted.
// ============================================================================
#pragma once
#include <string>
#include <vector>

namespace gem {

struct ImportedParticle {
  double x, y, z;     // [m]; z: lab position
  double ux, uy, uz;  // p / (m_e c)
  double w;           // real particles per macroparticle
};

struct ImportedBeam {
  std::vector<ImportedParticle> p;
  double charge_sign = -1.0;  // -1 electrons, +1 positrons
  std::string format;         // "openPMD 1.1.0"
  std::string species;        // species read
  long iteration = -1;        // openPMD iteration read
};

// species: empty = the only species of the file (error if there are several);
// iteration < 0: the only (or the last) iteration of the file.
// Throws std::runtime_error with a message naming the file.
ImportedBeam importBeam(const std::string& path, const std::string& species, long iteration);

}  // namespace gem
