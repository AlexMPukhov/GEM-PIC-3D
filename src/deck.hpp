// Structured ASCII input deck: blocks `name { key = value }`, comments with '#'; a repeated `laser` block
// becomes laser2, laser3, ... (also for overrides: laser2.a0=...), as do repeated `ion`, `bunch`, `undulator`,
// `quadrupole`, `plasma_lens` and `solenoid` blocks;
// values with physical units, command-line overrides `block.key=value`.
#pragma once
#include <map>
#include "expr.hpp"
#include <set>
#include <string>
#include <vector>

enum class Kind { None, Length, Time, Charge, Angle, Density, Mass, Energy, Field, Gradient };

class Deck {
public:
    static Deck read(const std::string& file);
    void set(const std::string& block, const std::string& key, const std::string& value, int line = 0);
    void override_from(const std::string& assignment);        // "block.key=value"
    bool has_block(const std::string& b) const { return blocks_.count(b) > 0; }
    bool has(const std::string& b, const std::string& k) const;
    std::string str(const std::string& b, const std::string& k, const std::string& def) const;
    std::string str(const std::string& b, const std::string& k) const;          // required
    double num(const std::string& b, const std::string& k, Kind kind) const;    // required, SI units
    double num(const std::string& b, const std::string& k, Kind kind, double def_si) const;
    long integer(const std::string& b, const std::string& k, long def) const;
    // list of numbers, comma or blank separated, with one unit at the end: "0, 40, 400 um" (SI units)
    std::vector<double> list(const std::string& b, const std::string& k, Kind kind) const;
    bool flag(const std::string& b, const std::string& k, bool def) const;
    void check_unused() const;                                                  // unknown keys are errors
    std::string text() const;                                                   // the deck after overrides
    std::string file = "(deck)";
    // expressions (as GEM-PIC-RZ): code units (set once lambda0 is known), the constants of the block
    // `constants` (code units, in the order of the deck); a single value that is not "number [unit]" is an
    // expression of the constants, evaluated in code units and returned in SI
    void set_code_units(double Lu, double Tu, double nc) const;
    void read_constants() const;
    Expr::UnitFn unit_fn() const;                                               // unit -> factor to code units
    Expr expr(const std::string& b, const std::string& k, const std::vector<std::string>& vars) const;
    std::vector<std::string> keys(const std::string& b) const;                  // in the order of the deck
private:
    std::string where(const std::string& b, const std::string& k) const;
    std::map<std::string, std::map<std::string, std::string>> blocks_;
    std::map<std::string, std::map<std::string, int>> lines_, order_;
    mutable std::set<std::string> used_;
    mutable double Lu_ = 0, Tu_ = 0, nc_ = 0;
    mutable std::map<std::string, double> consts_;
    double expr_si(const std::string& b, const std::string& k, Kind kind) const;
};

double parse_with_unit(const std::string& text, Kind kind, const std::string& where);
