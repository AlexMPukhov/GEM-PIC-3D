// Self-contained radix-2 FFT in one and two dimensions (no external libraries).
// Layout of 2D arrays: a[ix*ny + iy]. Sizes must be powers of two.
#pragma once
#include <cmath>
#include <complex>
#include <stdexcept>
#include <string>
#include <vector>

using cplx = std::complex<double>;

class FFT1 {
public:
    explicit FFT1(int n = 1) { init(n); }
    void init(int n) {
        n_ = n; int lg = 0;
        while ((1 << lg) < n) ++lg;
        if ((1 << lg) != n) throw std::runtime_error("FFT size " + std::to_string(n) + " is not a power of two");
        rev_.resize(n);
        for (int i = 0; i < n; ++i) { int r = 0; for (int b = 0; b < lg; ++b) if (i >> b & 1) r |= 1 << (lg - 1 - b); rev_[i] = r; }
        w_.resize(std::max(1, n / 2));
        for (int k = 0; k < n / 2; ++k) w_[k] = std::polar(1.0, -2.0 * M_PI * k / n);
    }
    // Unnormalized in-place transform of n contiguous values; forward has the sign -1.
    void run(cplx* a, bool inverse) const {
        for (int i = 0; i < n_; ++i) { int r = rev_[i]; if (i < r) std::swap(a[i], a[r]); }
        for (int len = 2; len <= n_; len <<= 1) {
            const int half = len / 2, step = n_ / len;
            for (int i = 0; i < n_; i += len)
                for (int j = 0; j < half; ++j) {
                    const cplx w = inverse ? std::conj(w_[j * step]) : w_[j * step];
                    const cplx u = a[i + j], v = a[i + j + half] * w;
                    a[i + j] = u + v; a[i + j + half] = u - v;
                }
        }
    }
private:
    int n_ = 1; std::vector<int> rev_; std::vector<cplx> w_;
};

class FFT2 {
public:
    FFT2(int nx, int ny) : nx_(nx), ny_(ny), fx_(nx), fy_(ny), col_(nx), buf_(size_t(nx) * ny) {}
    void forward(cplx* a) { transform(a, false); }
    void inverse(cplx* a) {           // normalized: inverse(forward(a)) = a
        transform(a, true);
        const double s = 1.0 / (double(nx_) * ny_);
        for (int i = 0; i < nx_ * ny_; ++i) a[i] *= s;
    }
    // Two real fields a, b -> their spectra A, B with one complex transform (b, B may be null).
    void forward_pair(const double* a, const double* b, cplx* A, cplx* B) {
        const int n = nx_ * ny_;
        for (int i = 0; i < n; ++i) buf_[i] = cplx(a[i], b ? b[i] : 0.0);
        forward(buf_.data());
        for (int ix = 0; ix < nx_; ++ix)
            for (int iy = 0; iy < ny_; ++iy) {
                const int i = ix * ny_ + iy, m = ((nx_ - ix) % nx_) * ny_ + (ny_ - iy) % ny_;
                const cplx z = buf_[i], zm = std::conj(buf_[m]);
                A[i] = 0.5 * (z + zm);
                if (B) B[i] = cplx(0, -0.5) * (z - zm);
            }
    }
    // Two Hermitian spectra A, B -> real fields a, b (B, b may be null).
    void inverse_pair(const cplx* A, const cplx* B, double* a, double* b) {
        const int n = nx_ * ny_;
        for (int i = 0; i < n; ++i) buf_[i] = A[i] + cplx(0, 1) * (B ? B[i] : cplx(0));
        inverse(buf_.data());
        for (int i = 0; i < n; ++i) { a[i] = buf_[i].real(); if (b) b[i] = buf_[i].imag(); }
    }
private:
    void transform(cplx* a, bool inv) {
        for (int ix = 0; ix < nx_; ++ix) fy_.run(a + ix * ny_, inv);
        for (int iy = 0; iy < ny_; ++iy) {
            for (int ix = 0; ix < nx_; ++ix) col_[ix] = a[ix * ny_ + iy];
            fx_.run(col_.data(), inv);
            for (int ix = 0; ix < nx_; ++ix) a[ix * ny_ + iy] = col_[ix];
        }
    }
    int nx_, ny_; FFT1 fx_, fy_; std::vector<cplx> col_, buf_;
};
