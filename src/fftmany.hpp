// Batched FFT of `count` consecutive planes, in place. A plane holds ny rows of nx contiguous points
// (index j*nx + ix). Fourier y (two_d): 2D transforms of the planes; free-hand y: 1D transforms along x of
// every row. Forward has the sign -1; the inverse is normalized (inverse(forward(a)) = a).
// CPU: the own FFT (any size, src/fft.hpp), OpenMP over rows and columns, or FFTW 3 with `make FFTW=1`
// (-DGEMPIC_FFTW, threads via fftw3_omp). GPU: cuFFT.
#pragma once
#include <map>
#include <memory>
#include <vector>
#include "fft.hpp"
#include "portable.hpp"
#if GEMPIC_GPU && defined(GEMPIC_EMU)
#include "emu/cufft.h"
#elif GEMPIC_GPU
#include <cufft.h>
#else
#include <complex>
#ifdef GEMPIC_FFTW
#include <fftw3.h>
#ifdef _OPENMP
#include <omp.h>
#endif
#endif
#endif

class FftMany {
public:
    FftMany(int nx, int ny, bool two_d) : nx_(nx), ny_(ny), two_d_(two_d), fx_(nx), fy_(two_d ? ny : 1) {}
    ~FftMany() {
#if GEMPIC_GPU
        for (auto& kv : plans_) cufftDestroy(kv.second);
#elif defined(GEMPIC_FFTW)
        for (auto& kv : fw_) { fftw_destroy_plan(kv.second.first); fftw_destroy_plan(kv.second.second); }
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
#elif defined(GEMPIC_FFTW)
        auto it = fw_.find(count);
        if (it == fw_.end()) {
            static bool threads = false;
            if (!threads) {
#ifdef _OPENMP
                fftw_init_threads();
                fftw_plan_with_nthreads(omp_get_max_threads());
#endif
                threads = true;
            }
            // planned on a scratch array (FFTW_ESTIMATE leaves it alone), executed on any unaligned array
            std::vector<fftw_complex> tmp(size_t(rows) * nx_);
            int n2[2] = {ny_, nx_}, n1[1] = {nx_};
            const unsigned fl = FFTW_ESTIMATE | FFTW_UNALIGNED;
            auto plan = [&](int sign) {
                return two_d_ ? fftw_plan_many_dft(2, n2, count, tmp.data(), nullptr, 1, int(nk), tmp.data(), nullptr, 1, int(nk), sign, fl)
                              : fftw_plan_many_dft(1, n1, int(rows), tmp.data(), nullptr, 1, nx_, tmp.data(), nullptr, 1, nx_, sign, fl);
            };
            it = fw_.emplace(count, std::make_pair(plan(FFTW_FORWARD), plan(FFTW_BACKWARD))).first;
            if (!it->second.first || !it->second.second) throw std::runtime_error("FFTW planning failed");
        }
        auto* z = reinterpret_cast<fftw_complex*>(data);
        fftw_execute_dft(inverse ? it->second.second : it->second.first, z, z);
        if (inverse) {
#pragma omp parallel for schedule(static)
            for (long i = 0; i < rows * nx_; ++i) { z[i][0] *= s; z[i][1] *= s; }
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
#elif defined(GEMPIC_FFTW)
    std::map<int, std::pair<fftw_plan, fftw_plan>> fw_;     // per batch count: forward, inverse
#endif
};
