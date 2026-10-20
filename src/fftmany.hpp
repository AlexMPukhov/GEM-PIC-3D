// Batched 2D FFT of `count` consecutive nx*ny planes, in place. Forward has the sign -1; the inverse is
// normalized (inverse(forward(a)) = a). CPU: the own radix-2 FFT, OpenMP over planes. GPU: cuFFT.
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
#ifdef _OPENMP
#include <omp.h>
#endif
#endif

class FftMany {
public:
    FftMany(int nx, int ny) : nx_(nx), ny_(ny) {
#if !GEMPIC_GPU
        int nt = 1;
#ifdef _OPENMP
        nt = omp_get_max_threads();
#endif
        for (int t = 0; t < nt; ++t) per_thread_.emplace_back(new FFT2(nx, ny));
#else
        FFT1 check_x(nx), check_y(ny);   // same size rule as the CPU build (powers of two)
#endif
    }
    ~FftMany() {
#if GEMPIC_GPU
        for (auto& kv : plans_) cufftDestroy(kv.second);
#endif
    }
    FftMany(const FftMany&) = delete;
    FftMany& operator=(const FftMany&) = delete;

    void run(cpx* data, int count, bool inverse) {
        if (count <= 0) return;
        const long nk = long(nx_) * ny_;
#if GEMPIC_GPU
        auto it = plans_.find(count);
        if (it == plans_.end()) {
            cufftHandle p;
            int n[2] = {nx_, ny_};
            if (cufftPlanMany(&p, 2, n, nullptr, 1, int(nk), nullptr, 1, int(nk), CUFFT_Z2Z, count) != CUFFT_SUCCESS)
                throw std::runtime_error("cufftPlanMany failed");
            it = plans_.emplace(count, p).first;
        }
        auto* z = reinterpret_cast<cufftDoubleComplex*>(data);
        if (cufftExecZ2Z(it->second, z, z, inverse ? CUFFT_INVERSE : CUFFT_FORWARD) != CUFFT_SUCCESS)
            throw std::runtime_error("cufftExecZ2Z failed");
        if (inverse) {
            const double s = 1.0 / double(nk);
            const long n = long(count) * nk;
            par_for(n, [=] HD(long i) { data[i] = s * data[i]; });
        }
#else
        auto* z = reinterpret_cast<std::complex<double>*>(data);
#pragma omp parallel for schedule(dynamic)
        for (int c = 0; c < count; ++c) {
            int t = 0;
#ifdef _OPENMP
            t = omp_get_thread_num();
#endif
            if (inverse) per_thread_[t]->inverse(z + long(c) * nk);
            else per_thread_[t]->forward(z + long(c) * nk);
        }
#endif
    }
private:
    int nx_, ny_;
#if GEMPIC_GPU
    std::map<int, cufftHandle> plans_;
#else
    std::vector<std::unique_ptr<FFT2>> per_thread_;
#endif
};
