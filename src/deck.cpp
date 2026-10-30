#include "deck.hpp"
#include <cmath>
#include <fstream>
#include <sstream>
#include <stdexcept>
#include <cstdlib>
#include <algorithm>

static std::string trim(const std::string& s) {
    size_t a = s.find_first_not_of(" \t\r"), b = s.find_last_not_of(" \t\r");
    return a == std::string::npos ? "" : s.substr(a, b - a + 1);
}

Deck Deck::read(const std::string& file) {
    std::ifstream in(file);
    if (!in) throw std::runtime_error("cannot open deck " + file);
    Deck d; d.file = file;
    std::string line, block; int ln = 0;
    auto err = [&](const std::string& m) { return std::runtime_error(file + ":" + std::to_string(ln) + ": " + m); };
    while (std::getline(in, line)) {
        ++ln;
        {   // comment: '#' outside quotes
            bool q = false;
            for (size_t c = 0; c < line.size(); ++c) {
                if (line[c] == '"') q = !q;
                else if (line[c] == '#' && !q) { line = line.substr(0, c); break; }
            }
        }
        line = trim(line);
        if (line.empty()) continue;
        if (block.empty()) {
            auto br = line.find('{');
            if (br == std::string::npos) throw err("expected 'block {'");
            block = trim(line.substr(0, br));
            if (d.blocks_.count(block)) {
                static const std::set<std::string> rep = {"laser", "ion", "bunch", "undulator", "quadrupole", "plasma_lens", "solenoid", "chicane"};
                if (!rep.count(block)) throw err("block '" + block + "' given twice");
                int k = 2;
                while (d.blocks_.count(block + std::to_string(k))) ++k;
                block += std::to_string(k);
            }
            d.blocks_[block];
            line = trim(line.substr(br + 1));
            if (line.empty()) continue;
        }
        bool close = false;
        if (line.back() == '}') { close = true; line = trim(line.substr(0, line.size() - 1)); }
        if (!line.empty()) {
            auto eq = line.find('=');
            if (eq == std::string::npos) throw err("expected 'key = value'");
            d.set(block, trim(line.substr(0, eq)), trim(line.substr(eq + 1)), ln);
        }
        if (close) block.clear();
    }
    if (!block.empty()) throw std::runtime_error(file + ": block '" + block + "' is not closed");
    return d;
}

void Deck::set(const std::string& b, const std::string& k, const std::string& v, int line) {
    blocks_[b][k] = v; lines_[b][k] = line;
    if (!order_[b].count(k)) order_[b][k] = line > 0 ? line : 1000000000 + int(order_[b].size());   // overrides keep the place
}

void Deck::override_from(const std::string& a) {
    // block.key=value; the block name may contain dots (diag.light.every=5: block diag.light, key every)
    const auto eq = a.find('=');
    const auto dot = eq == std::string::npos ? std::string::npos : a.rfind('.', eq);
    if (dot == std::string::npos || eq == std::string::npos)
        throw std::runtime_error("override '" + a + "' must read block.key=value");
    set(trim(a.substr(0, dot)), trim(a.substr(dot + 1, eq - dot - 1)), trim(a.substr(eq + 1)), -1);
}

bool Deck::has(const std::string& b, const std::string& k) const {
    auto it = blocks_.find(b);
    return it != blocks_.end() && it->second.count(k);
}

std::string Deck::where(const std::string& b, const std::string& k) const {
    auto it = lines_.find(b);
    int ln = (it != lines_.end() && it->second.count(k)) ? it->second.at(k) : 0;
    return file + (ln > 0 ? ":" + std::to_string(ln) : std::string(" (command line)")) + ": " + b + "." + k;
}

std::string Deck::str(const std::string& b, const std::string& k, const std::string& def) const {
    if (!has(b, k)) return def;
    used_.insert(b + "." + k);
    const std::string& v = blocks_.at(b).at(k);
    return v.size() >= 2 && v.front() == '"' && v.back() == '"' ? v.substr(1, v.size() - 2) : v;    // quotes
}
std::string Deck::str(const std::string& b, const std::string& k) const {
    if (!has(b, k)) throw std::runtime_error(file + ": missing key " + b + "." + k);
    return str(b, k, "");
}
double Deck::num(const std::string& b, const std::string& k, Kind kind) const {
    const std::string t = str(b, k);
    try { return parse_with_unit(t, kind, where(b, k)); }
    catch (const std::runtime_error&) {
        // "number [unit]" with a unit of the wrong kind stays an error; anything else may be an expression
        std::istringstream is(t);
        double v; std::string u, rest;
        double f;
        if (Lu_ <= 0 || ((is >> v) && !(is >> u >> rest) && (u.empty() || unit_fn()(u, f)))) throw;
        return expr_si(b, k, kind);
    }
}
double Deck::num(const std::string& b, const std::string& k, Kind kind, double def) const {
    return has(b, k) ? num(b, k, kind) : def;
}
long Deck::integer(const std::string& b, const std::string& k, long def) const {
    if (!has(b, k)) return def;
    double v = num(b, k, Kind::None);
    if (v != std::floor(v)) throw std::runtime_error(where(b, k) + ": integer expected");
    return long(v);
}
std::vector<double> Deck::list(const std::string& b, const std::string& k, Kind kind) const {
    std::string t = str(b, k);
    for (char& ch : t) if (ch == ',') ch = ' ';
    std::istringstream is(t);
    std::vector<std::string> tok;
    for (std::string w; is >> w;) tok.push_back(w);
    std::string unit;
    if (!tok.empty()) {
        char* end = nullptr;
        std::strtod(tok.back().c_str(), &end);
        if (end == tok.back().c_str()) { unit = tok.back(); tok.pop_back(); }
    }
    if (tok.empty()) throw std::runtime_error(where(b, k) + ": list of numbers expected");
    std::vector<double> v;
    for (auto& w : tok) v.push_back(parse_with_unit(unit.empty() ? w : w + " " + unit, kind, where(b, k)));
    return v;
}
bool Deck::flag(const std::string& b, const std::string& k, bool def) const {
    if (!has(b, k)) return def;
    std::string v = str(b, k, "");
    if (v == "true" || v == "on" || v == "yes" || v == "1") return true;
    if (v == "false" || v == "off" || v == "no" || v == "0") return false;
    throw std::runtime_error(where(b, k) + ": true or false expected");
}
void Deck::check_unused() const {
    for (auto& [b, kv] : blocks_)
        for (auto& [k, v] : kv)
            if (!used_.count(b + "." + k)) throw std::runtime_error(where(b, k) + ": unknown key");
}

double parse_with_unit(const std::string& text, Kind kind, const std::string& where) {
    std::istringstream is(text);
    double v; std::string unit;
    if (!(is >> v)) throw std::runtime_error(where + ": number expected, got '" + text + "'");
    is >> unit;
    static const std::map<std::string, std::pair<Kind, double>> units = {
        {"nm", {Kind::Length, 1e-9}}, {"um", {Kind::Length, 1e-6}}, {"mm", {Kind::Length, 1e-3}},
        {"cm", {Kind::Length, 1e-2}}, {"m", {Kind::Length, 1.0}},
        {"fs", {Kind::Time, 1e-15}}, {"ps", {Kind::Time, 1e-12}}, {"ns", {Kind::Time, 1e-9}},
        {"pC", {Kind::Charge, 1e-12}}, {"nC", {Kind::Charge, 1e-9}},
        {"cm^-3", {Kind::Density, 1e6}}, {"m^-3", {Kind::Density, 1.0}},
        {"rad", {Kind::Angle, 1.0}}, {"mrad", {Kind::Angle, 1e-3}}, {"deg", {Kind::Angle, M_PI / 180}},
        {"u", {Kind::Mass, 1.66053906660e-27}}, {"m_p", {Kind::Mass, 1.67262192369e-27}},
        {"m_e", {Kind::Mass, 9.1093837015e-31}}, {"kg", {Kind::Mass, 1.0}},
        {"eV", {Kind::Energy, 1.0}}, {"keV", {Kind::Energy, 1e3}}, {"MeV", {Kind::Energy, 1e6}},
        {"GeV", {Kind::Energy, 1e9}}, {"T", {Kind::Field, 1.0}}, {"mT", {Kind::Field, 1e-3}},
        {"T/m", {Kind::Gradient, 1.0}}};                                              // energies in eV
    if (unit.empty()) {
        if (kind == Kind::None || kind == Kind::Angle) return v;   // a bare angle is in radians
        throw std::runtime_error(where + ": a unit is required");
    }
    if (kind == Kind::None) throw std::runtime_error(where + ": dimensionless value expected");
    auto it = units.find(unit);
    if (it == units.end()) throw std::runtime_error(where + ": unknown unit '" + unit + "'");
    if (it->second.first != kind) throw std::runtime_error(where + ": unit '" + unit + "' is of the wrong kind");
    return v * it->second.second;
}

std::string Deck::text() const {
    std::string t;
    for (const auto& b : blocks_) {
        t += b.first + " {\n";
        for (const auto& kv : b.second) t += "  " + kv.first + " = " + kv.second + "\n";
        t += "}\n";
    }
    return t;
}

// ---------------------------------------------------------------- expressions
namespace {
// code unit of each kind in SI (lengths 1/k0, times 1/omega0, densities n_c, charge e, energy m c^2 [eV], mass m_e,
// magnetic field m omega0 / e, gradient m omega0 / (e k0^-1))
double code_si(Kind kind, double Lu, double Tu, double nc) {
    const double me = 9.1093837015e-31, e = 1.602176634e-19;
    switch (kind) {
        case Kind::Length: return Lu;
        case Kind::Time: return Tu;
        case Kind::Density: return nc;
        case Kind::Charge: return e;
        case Kind::Energy: return 510998.95;
        case Kind::Mass: return me;
        case Kind::Field: return me / (Tu * e);
        case Kind::Gradient: return me / (Tu * e) / Lu;
        default: return 1.0;
    }
}
}  // namespace

void Deck::set_code_units(double Lu, double Tu, double nc) const { Lu_ = Lu; Tu_ = Tu; nc_ = nc; }

Expr::UnitFn Deck::unit_fn() const {
    const double Lu = Lu_, Tu = Tu_, nc = nc_;
    return [Lu, Tu, nc](const std::string& u, double& f) {
        if (u == "k0^-1") { f = 1; return true; }
        if (u == "n_c") { f = 1; return true; }
        for (Kind kind : {Kind::Length, Kind::Time, Kind::Charge, Kind::Density, Kind::Angle, Kind::Mass, Kind::Energy,
                          Kind::Field, Kind::Gradient}) {
            try { f = parse_with_unit("1 " + u, kind, "") / code_si(kind, Lu, Tu, nc); return true; }
            catch (const std::runtime_error&) {}
        }
        return false;
    };
}

std::vector<std::string> Deck::keys(const std::string& b) const {
    std::vector<std::pair<int, std::string>> v;
    auto it = order_.find(b);
    if (it != order_.end()) for (auto& [k, o] : it->second) v.push_back({o, k});
    std::sort(v.begin(), v.end());
    std::vector<std::string> r;
    for (auto& p : v) r.push_back(p.second);
    return r;
}

// the block `constants`: each a value with unit or an expression of the constants above it (code units)
void Deck::read_constants() const {
    consts_.clear();
    for (const auto& k : keys("constants")) {
        std::string name = k;
        for (char& ch : name) ch = char(std::tolower(static_cast<unsigned char>(ch)));
        consts_[name] = expr("constants", k, {})(nullptr);
    }
}

Expr Deck::expr(const std::string& b, const std::string& k, const std::vector<std::string>& vars) const {
    try { return Expr(str(b, k), vars, consts_, unit_fn()); }
    catch (const std::invalid_argument& e) { throw std::runtime_error(where(b, k) + ": " + e.what()); }
}

double Deck::expr_si(const std::string& b, const std::string& k, Kind kind) const {
    return expr(b, k, {})(nullptr) * code_si(kind, Lu_, Tu_, nc_);
}
