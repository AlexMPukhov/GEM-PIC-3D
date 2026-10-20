// Self-contained FFT in one and two dimensions, any size (no external libraries).
// Layout of 2D arrays in FFT2: a[ix*ny + iy].
#pragma once
#include <cmath>
#include <algorithm>
#include <complex>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

using cplx = std::complex<double>;

// FFT1: transforms of any length n >= 1.
//  - n a power of two: in-place iterative radix-2 (as before; bit for bit the results of the earlier versions).
//  - n = product of small primes (<= 31): self-sorting mixed-radix Stockham, radix 4 first, then 2, 3, 5, 7, ...
//    with a direct butterfly for each radix; O(n sum p).
//  - any other n (a prime factor > 31): Bluestein's chirp-z algorithm on a radix-2 transform of length >= 2n-1.
// Sizes with small factors (2^a 3^b 5^c 7^d) are the fast ones. run() is const and thread safe: the scratch
// memory is thread-local.
class FFT1 {
public:
    explicit FFT1(int n = 1) { init(n); }
    void init(int n) {
        if (n < 1) throw std::runtime_error("FFT size " + std::to_string(n) + " must be positive");
        n_ = n; pow2_ = (n & (n - 1)) == 0; radices_.clear(); tw_.clear(); bl_.reset();
        if (pow2_) {
            int lg = 0;
            while ((1 << lg) < n) ++lg;
            rev_.resize(n);
            for (int i = 0; i < n; ++i) { int r = 0; for (int b = 0; b < lg; ++b) if (i >> b & 1) r |= 1 << (lg - 1 - b); rev_[i] = r; }
            w_.resize(std::max(1, n / 2));
            for (int k = 0; k < n / 2; ++k) w_[k] = std::polar(1.0, -2.0 * M_PI * k / n);
            return;
        }
        int m = n;
        while (m % 4 == 0) { radices_.push_back(4); m /= 4; }
        while (m % 2 == 0) { radices_.push_back(2); m /= 2; }
        for (int p = 3; p <= 31 && m > 1; p += 2)
            while (m % p == 0) { radices_.push_back(p); m /= p; }
        if (m > 1) {                                   // Bluestein
            radices_.clear();
            bl_.reset(new Bluestein(n));
            return;
        }
        // twiddles per stage: stage with radix p after sub-transforms of length L: exp(-2 pi i q k / (L p))
        int L = 1;
        for (int p : radices_) {
            std::vector<cplx> t(size_t(L) * p);
            for (int k = 0; k < L; ++k)
                for (int q = 0; q < p; ++q) t[size_t(k) * p + q] = std::polar(1.0, -2.0 * M_PI * double(q) * k / (double(L) * p));
            tw_.push_back(std::move(t));
            L *= p;
        }
        for (int p : radices_)
            if (int(roots_.size()) <= p) roots_.resize(p + 1);
        for (int p : radices_) {
            roots_[p].resize(p);
            for (int r = 0; r < p; ++r) roots_[p][r] = std::polar(1.0, -2.0 * M_PI * r / p);
        }
    }
    int size() const { return n_; }
    // Unnormalized in-place transform of n contiguous values; forward has the sign -1.
    void run(cplx* a, bool inverse) const {
        if (n_ == 1) return;
        if (pow2_) { radix2(a, inverse); return; }
        // the inverse as the conjugate of the forward transform of the conjugate
        if (inverse) for (int i = 0; i < n_; ++i) a[i] = std::conj(a[i]);
        if (bl_) bl_->forward(a);
        else stockham(a);
        if (inverse) for (int i = 0; i < n_; ++i) a[i] = std::conj(a[i]);
    }
private:
    void radix2(cplx* a, bool inverse) const {
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
    // Stockham autosort, decimation in time: stage s combines p sub-transforms of length L (stride n/(L p) in
    // the input order) into transforms of length L p.
    void stockham(cplx* a) const {
        static thread_local std::vector<cplx> buf;
        if (int(buf.size()) < n_) buf.resize(n_);
        cplx* x = a; cplx* y = buf.data();
        cplx v[32], u[32];
        int L = 1;
        for (size_t s = 0; s < radices_.size(); ++s) {
            const int p = radices_[s], l = n_ / (L * p);
            const cplx* t = tw_[s].data();
            const cplx* rt = roots_[p].data();
            for (int j = 0; j < l; ++j)
                for (int k = 0; k < L; ++k) {
                    for (int q = 0; q < p; ++q) v[q] = x[(size_t(q) * l + j) * L + k] * t[size_t(k) * p + q];
                    if (p == 2) { u[0] = v[0] + v[1]; u[1] = v[0] - v[1]; }
                    else if (p == 4) {
                        const cplx s02 = v[0] + v[2], d02 = v[0] - v[2], s13 = v[1] + v[3], d13 = v[1] - v[3];
                        const cplx md13(d13.imag(), -d13.real());          // -i (v1 - v3)
                        u[0] = s02 + s13; u[1] = d02 + md13; u[2] = s02 - s13; u[3] = d02 - md13;
                    } else
                        for (int r = 0; r < p; ++r) {
                            cplx acc = v[0];
                            for (int q = 1; q < p; ++q) acc += v[q] * rt[(size_t(r) * q) % p];
                            u[r] = acc;
                        }
                    for (int r = 0; r < p; ++r) y[(size_t(j) * p * L) + size_t(r) * L + k] = u[r];
                }
            std::swap(x, y);
            L *= p;
        }
        if (x != a) std::copy(x, x + n_, a);
    }
    // Bluestein: X_k = c_k^* sum_j (x_j c_j^*) c_{k-j}, c_j = exp(i pi j^2 / n), as a cyclic convolution of
    // length M (a power of two >= 2n - 1).
    struct Bluestein {
        int n, M;
        std::vector<int> rev; std::vector<cplx> w, chirp, B;
        explicit Bluestein(int n_) : n(n_) {
            M = 1; while (M < 2 * n - 1) M <<= 1;
            chirp.resize(n);
            for (long j = 0; j < n; ++j) chirp[j] = std::polar(1.0, M_PI * double((j * j) % (2L * n)) / n);
            int lg = 0; while ((1 << lg) < M) ++lg;
            rev.resize(M);
            for (int i = 0; i < M; ++i) { int r = 0; for (int b = 0; b < lg; ++b) if (i >> b & 1) r |= 1 << (lg - 1 - b); rev[i] = r; }
            w.resize(M / 2);
            for (int k = 0; k < M / 2; ++k) w[k] = std::polar(1.0, -2.0 * M_PI * k / M);
            B.assign(M, cplx(0));
            B[0] = chirp[0];
            for (int j = 1; j < n; ++j) B[j] = B[M - j] = chirp[j];
            fft(B.data(), false);
        }
        void fft(cplx* a, bool inv) const {
            for (int i = 0; i < M; ++i) { int r = rev[i]; if (i < r) std::swap(a[i], a[r]); }
            for (int len = 2; len <= M; len <<= 1) {
                const int half = len / 2, step = M / len;
                for (int i = 0; i < M; i += len)
                    for (int j = 0; j < half; ++j) {
                        const cplx ww = inv ? std::conj(w[j * step]) : w[j * step];
                        const cplx u = a[i + j], v = a[i + j + half] * ww;
                        a[i + j] = u + v; a[i + j + half] = u - v;
                    }
            }
        }
        void forward(cplx* a) const {
            static thread_local std::vector<cplx> A;
            A.assign(M, cplx(0));
            for (int j = 0; j < n; ++j) A[j] = a[j] * std::conj(chirp[j]);
            fft(A.data(), false);
            for (int i = 0; i < M; ++i) A[i] *= B[i];
            fft(A.data(), true);
            for (int k = 0; k < n; ++k) a[k] = std::conj(chirp[k]) * A[k] / double(M);
        }
    };
    int n_ = 1; bool pow2_ = true;
    std::vector<int> rev_; std::vector<cplx> w_;
    std::vector<int> radices_;
    std::vector<std::vector<cplx>> tw_, roots_;
    std::shared_ptr<const Bluestein> bl_;
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
