// Batched FFT of `count` consecutive planes, in place. A plane holds ny rows of nx contiguous points
// (index j*nx + ix). Fourier y (two_d): 2D transforms of the planes; free-hand y: 1D transforms along x of
// every row. Forward has the sign -1; the inverse is normalized (inverse(forward(a)) = a).
// CPU: the own radix-2 FFT, OpenMP over rows and columns. GPU: cuFFT.
#pragma once
#include <map>
#include <memory>
#include <vector>
#include "fft.hpp"
#include "portable.hpp"
#if GEMPIC_GPU
#include <cufft.h>
#else
#include <complex>
#endif

class FftMany {
public:
    FftMany(int nx, int ny, bool two_d) : nx_(nx), ny_(ny), two_d_(two_d), fx_(nx), fy_(two_d ? ny : 1) {}
    ~FftMany() {
#if GEMPIC_GPU
        for (auto& kv : plans_) cufftDestroy(kv.second);
#endif
    }
    FftMany(const FftMany&) = delete;
    FftMany& operator=(const FftMany&) = delete;

    void run(cpx* data, int count, bool inverse) {
        if (count <= 0) return;
        const long rows = long(count) * ny_, nk = long(nx_) * ny_;
        const double s = 1.0 / double(two_d_ ? nk : nx_);
#if GEMPIC_GPU
        const long key = two_d_ ? -long(count) : rows;
        auto it = plans_.find(key);
        if (it == plans_.end()) {
            cufftHandle p;
            cufftResult r;
            if (two_d_) {
                int n[2] = {ny_, nx_};
                r = cufftPlanMany(&p, 2, n, nullptr, 1, int(nk), nullptr, 1, int(nk), CUFFT_Z2Z, count);
            } else {
                int n[1] = {nx_};
                r = cufftPlanMany(&p, 1, n, nullptr, 1, nx_, nullptr, 1, nx_, CUFFT_Z2Z, int(rows));
            }
            if (r != CUFFT_SUCCESS) throw std::runtime_error("cufftPlanMany failed");
            it = plans_.emplace(key, p).first;
        }
        auto* z = reinterpret_cast<cufftDoubleComplex*>(data);
        if (cufftExecZ2Z(it->second, z, z, inverse ? CUFFT_INVERSE : CUFFT_FORWARD) != CUFFT_SUCCESS)
            throw std::runtime_error("cufftExecZ2Z failed");
        if (inverse) {
            const long n = rows * nx_;
            par_for(n, [=] HD(long i) { data[i] = s * data[i]; });
        }
#else
        auto* z = reinterpret_cast<std::complex<double>*>(data);
        const FFT1& fx = fx_; const FFT1& fy = fy_;
        const int nx = nx_, ny = ny_;
#pragma omp parallel for schedule(static)
        for (long r = 0; r < rows; ++r) fx.run(z + r * nx, inverse);
        if (two_d_) {
            const long cols = long(count) * nx;
#pragma omp parallel
            {
                std::vector<std::complex<double>> col(ny);
#pragma omp for schedule(static)
                for (long c = 0; c < cols; ++c) {
                    std::complex<double>* p = z + (c / nx) * nk + c % nx;
                    for (int j = 0; j < ny; ++j) col[j] = p[long(j) * nx];
                    fy.run(col.data(), inverse);
                    for (int j = 0; j < ny; ++j) p[long(j) * nx] = col[j];
                }
            }
        }
        if (inverse) {
#pragma omp parallel for schedule(static)
            for (long i = 0; i < rows * nx; ++i) z[i] *= s;
        }
#endif
    }
private:
    int nx_, ny_;
    bool two_d_;
    FFT1 fx_, fy_;    // stateless during run(): shared by the threads
#if GEMPIC_GPU
    std::map<long, cufftHandle> plans_;
#endif
};
