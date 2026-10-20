// Structured ASCII input deck: blocks `name { key = value }`, comments with '#',
// values with physical units, command-line overrides `block.key=value`.
#pragma once
#include <map>
#include <set>
#include <string>
#include <vector>

enum class Kind { None, Length, Time, Charge, Angle };

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
    std::string file = "(deck)";
private:
    std::string where(const std::string& b, const std::string& k) const;
    std::map<std::string, std::map<std::string, std::string>> blocks_;
    std::map<std::string, std::map<std::string, int>> lines_;
    mutable std::set<std::string> used_;
};

double parse_with_unit(const std::string& text, Kind kind, const std::string& where);
