#include "grid.hpp"
#include <algorithm>
#include <cmath>

namespace {
// ln(1 + y)/y and (exp(y) - 1)/y, accurate for small y
double log1p_over(double y) { return std::fabs(y) < 1e-8 ? 1.0 - 0.5 * y : std::log1p(y) / y; }
double expm1_over(double y) { return std::fabs(y) < 1e-8 ? 1.0 + 0.5 * y : std::expm1(y) / y; }
}  // namespace

std::string StepMapping::set(const std::vector<double>& x, const std::vector<double>& h) {
    if (x.empty() || x.size() != h.size()) return "needs as many steps as reference points (at least one)";
    for (size_t k = 0; k < x.size(); ++k) {
        if (!(x[k] >= 0)) return "reference points must not be negative";
        if (k > 0 && !(x[k] > x[k - 1])) return "reference points must be strictly increasing";
        if (!(h[k] > 0)) return "grid steps must be positive";
    }
    xk_.clear(); hk_.clear();
    if (x[0] > 0) { xk_.push_back(0.0); hk_.push_back(h[0]); }   // constant up to the first point
    xk_.insert(xk_.end(), x.begin(), x.end());
    hk_.insert(hk_.end(), h.begin(), h.end());
    const size_t K = xk_.size();
    bk_.assign(K, 0.0); xik_.assign(K, 0.0); max_slope_ = 0;
    for (size_t k = 0; k + 1 < K; ++k) {
        const double L = xk_[k + 1] - xk_[k];
        bk_[k] = (hk_[k + 1] - hk_[k]) / L;
        max_slope_ = std::max(max_slope_, std::fabs(bk_[k]));
        xik_[k + 1] = xik_[k] + (L / hk_[k]) * log1p_over(bk_[k] * L / hk_[k]);
    }
    bk_[K - 1] = 0;
    return "";
}
double StepMapping::step(double x) const {
    if (x <= xk_.front()) return hk_.front();
    if (x >= xk_.back()) return hk_.back();
    const size_t k = size_t(std::upper_bound(xk_.begin(), xk_.end(), x) - xk_.begin()) - 1;
    return hk_[k] + bk_[k] * (x - xk_[k]);
}
double StepMapping::index(double x) const {
    if (x <= 0) return x / hk_.front();
    const size_t k = std::min(size_t(std::upper_bound(xk_.begin(), xk_.end(), x) - xk_.begin()) - 1, xk_.size() - 1);
    const double d = x - xk_[k];
    return xik_[k] + (d / hk_[k]) * log1p_over(bk_[k] * d / hk_[k]);
}
double StepMapping::coord(double xi) const {
    if (xi <= 0) return xi * hk_.front();
    const size_t k = std::min(size_t(std::upper_bound(xik_.begin(), xik_.end(), xi) - xik_.begin()) - 1, xik_.size() - 1);
    const double t = xi - xik_[k];
    return xk_[k] + hk_[k] * t * expm1_over(bk_[k] * t);
}
bool StepMapping::constant() const {
    for (double h : hk_) if (h != hk_.front()) return false;
    return true;
}

bool tridiagonal_eigen(std::vector<double>& d, std::vector<double> e, std::vector<double>& z) {
    const int n = int(d.size());
    z.assign(size_t(n) * n, 0.0);
    for (int i = 0; i < n; ++i) z[size_t(i) * n + i] = 1;
    if (n == 0) return true;
    e.resize(n); e[n - 1] = 0;
    const double eps = 2.2e-16;
    for (int l = 0; l < n; ++l) {
        int iter = 0, m;
        do {
            for (m = l; m < n - 1; ++m) {
                const double dd = std::fabs(d[m]) + std::fabs(d[m + 1]);
                if (std::fabs(e[m]) <= eps * dd) break;
            }
            if (m == l) break;
            if (++iter > 60) return false;
            double g = (d[l + 1] - d[l]) / (2 * e[l]);
            double r = std::hypot(g, 1.0);
            g = d[m] - d[l] + e[l] / (g + std::copysign(r, g));       // shift: eigenvalue of the 2x2 corner
            double s = 1, c = 1, p = 0;
            int i;
            bool deflated = false;
            for (i = m - 1; i >= l; --i) {
                const double f = s * e[i], b = c * e[i];
                r = std::hypot(f, g);
                e[i + 1] = r;
                if (r == 0) { d[i + 1] -= p; e[m] = 0; deflated = true; break; }
                s = f / r; c = g / r;
                g = d[i + 1] - p;
                r = (d[i] - g) * s + 2 * c * b;
                p = s * r;
                d[i + 1] = g + p;
                g = c * r - b;
                for (int k = 0; k < n; ++k) {                         // accumulate the rotation
                    const double t = z[size_t(k) * n + i + 1];
                    z[size_t(k) * n + i + 1] = s * z[size_t(k) * n + i] + c * t;
                    z[size_t(k) * n + i] = c * z[size_t(k) * n + i] - s * t;
                }
            }
            if (deflated) continue;
            d[l] -= p; e[l] = g; e[m] = 0;
        } while (m != l);
    }
    return true;
}
