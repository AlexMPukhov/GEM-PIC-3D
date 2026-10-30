// ============================================================================
//  atomic_data.hpp -- elements known to the code: atomic number, standard
//  atomic mass and the ionization energies of all charge states
//  (NIST Atomic Spectra Database, see atomic_data.cpp)
// ============================================================================
#pragma once
#include <string>
#include <vector>

namespace gem {

struct Element {
  const char* symbol;           // "H", "He", ... (capitalized)
  int Znuc;                     // atomic number
  double mass_u;                // standard atomic mass [u]
  std::vector<double> ip_eV;    // ip_eV[k]: ionization energy of charge state k -> k+1 [eV], size Znuc
};

// all elements of the table, in order of the atomic number
const std::vector<Element>& elements();

// look up an element by its chemical symbol (case-insensitive); nullptr if unknown
const Element* findElement(const std::string& symbol);

// provenance of the ionization energies (for the run summary)
const char* atomicDataSource();

}  // namespace gem
