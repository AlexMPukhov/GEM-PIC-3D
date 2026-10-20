// Free-hand grids for zeta and y.
//
// StepMapping (as in GEM-PIC-RZ, src/mapping.*): the local step h(x) is given at reference points, linear
// in x between them and constant outside. The grid index xi(x) = int_0^x dx'/h(x') counts cells, so node i
// lies at X(i) and the cell size near x is h(x). On a segment with h = h_k + b (x - x_k) both directions are
// analytic: xi = xi_k + ln(h(x)/h_k)/b, X = x_k + h_k (exp(b (xi - xi_k)) - 1)/b. |b| is the relative change
// of the step from one cell to the next.
//
// Axis: the nodes of a grid on the device and the location of a coordinate in it.
#pragma once
#include <string>
#include <vector>
#include "portable.hpp"

class StepMapping {
public:
    // reference positions x (>= 0, strictly increasing) and steps h (> 0); returns "" or an error message
    std::string set(const std::vector<double>& x, const std::vector<double>& h);
    double coord(double xi) const;     // X(xi), X(0) = 0
    double index(double x) const;      // xi(x), the inverse of coord
    double step(double x) const;       // h(x)
    double max_change() const { return max_slope_; }   // max |dh/dx| = largest cell-to-cell change
    bool constant() const;             // one step everywhere
private:
    std::vector<double> xk_, hk_, xik_, bk_;
    double max_slope_ = 0;
};

// Nodes X[0..n-1] (strictly increasing). locate() returns the cell i with X[i] <= x < X[i+1] and the
// fraction f in it; beyond the ends the end cells are continued. Uniform grids take the direct path.
struct Axis {
    const double* X = nullptr;
    int n = 0, uniform = 0;
    double x0 = 0, ih = 1;
    HD void locate(double x, int& i, double& f) const {
        if (uniform) { const double t = (x - x0) * ih; i = int(::floor(t)); f = t - i; return; }
        if (x < X[0]) { const double t = (x - X[0]) / (X[1] - X[0]); i = int(::floor(t)); f = t - i; return; }
        if (x >= X[n - 1]) {
            const double t = (x - X[n - 1]) / (X[n - 1] - X[n - 2]);
            const int k = int(::floor(t));
            i = n - 1 + k; f = t - k; return;
        }
        int lo = 0, hi = n - 1;            // X[lo] <= x < X[hi]
        while (hi - lo > 1) { const int m = (lo + hi) >> 1; if (X[m] <= x) lo = m; else hi = m; }
        i = lo; f = (x - X[lo]) / (X[lo + 1] - X[lo]);
    }
};

// Symmetric tridiagonal eigenproblem by the implicit QL method: d (n) diagonal, e (n) with e[i] coupling
// i and i+1 (e[n-1] unused). On return d holds the eigenvalues and z (n x n, z[k*n + m]) the orthonormal
// eigenvectors in its columns m. Returns false if it did not converge.
bool tridiagonal_eigen(std::vector<double>& d, std::vector<double> e, std::vector<double>& z);
