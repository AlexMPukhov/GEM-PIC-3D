// Mathematical expressions in the input deck (ported from GEM-PIC-RZ, src/expr.*). An expression is compiled
// once into a small stack program and evaluated on the host for given values of its variables, e.g.
//     density_function = "exp(-(r/w)^2) * (1 + 0.1*cos(z/(50 um)))"
// Syntax
//   numbers     2, 1.5e-3, and numbers with a unit: 10 um, 3e18 cm^-3, 5 fs, 0.2 k0^-1, 50% (code units)
//   variables   given by the context (x, y, z, r, ...), code units
//   constants   pi, and the entries of the deck block `constants`
//   operators   + - * / ^ (also **, right-associative, -x^2 = -(x^2)), comparisons < <= > >= == != (1 or 0),
//               && || !, cond ? a : b
//   functions   sin cos tan asin acos atan atan2 sinh cosh tanh exp log (= ln) log10 sqrt abs floor ceil min max
//               pow sign step (1 for x >= 0), if(cond, a, b), gauss(x, s) = exp(-x^2/(2 s^2)), rect(x, w) = 1
//               for |x| <= w/2
// Errors are thrown as std::invalid_argument with the position in the text.
#pragma once
#include <functional>
#include <initializer_list>
#include <map>
#include <string>
#include <vector>

class Expr {
public:
    // unit name -> factor to code units (false: not a unit)
    using UnitFn = std::function<bool(const std::string&, double&)>;
    Expr() = default;
    Expr(const std::string& text, const std::vector<std::string>& vars, const std::map<std::string, double>& consts,
         const UnitFn& units);
    double operator()(const double* v) const;
    double operator()(std::initializer_list<double> v) const { return (*this)(v.begin()); }
    bool empty() const { return code.empty(); }
    const std::string& text() const { return src; }

    enum Op : int { PUSH, VAR, NEG, NOT, ADD, SUB, MUL, DIV, POW, LT, LE, GT, GE, EQ, NE, AND, OR, SEL, F1, F2 };
    struct Ins { Op op; int arg; double val; };

private:
    std::string src;
    std::vector<Ins> code;
    int maxStack = 0;
};
