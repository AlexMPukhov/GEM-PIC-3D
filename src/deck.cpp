#include "deck.hpp"
#include <cmath>
#include <fstream>
#include <sstream>
#include <stdexcept>
#include <cstdlib>

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
        if (auto c = line.find('#'); c != std::string::npos) line = line.substr(0, c);
        line = trim(line);
        if (line.empty()) continue;
        if (block.empty()) {
            auto br = line.find('{');
            if (br == std::string::npos) throw err("expected 'block {'");
            block = trim(line.substr(0, br));
            if (d.blocks_.count(block)) {
                if (block != "laser") throw err("block '" + block + "' given twice");
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
    return blocks_.at(b).at(k);
}
std::string Deck::str(const std::string& b, const std::string& k) const {
    if (!has(b, k)) throw std::runtime_error(file + ": missing key " + b + "." + k);
    return str(b, k, "");
}
double Deck::num(const std::string& b, const std::string& k, Kind kind) const {
    return parse_with_unit(str(b, k), kind, where(b, k));
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
        {"rad", {Kind::Angle, 1.0}}, {"mrad", {Kind::Angle, 1e-3}}, {"deg", {Kind::Angle, M_PI / 180}}};
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
