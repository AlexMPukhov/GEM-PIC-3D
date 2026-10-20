// GPU emulator (make EMU=1): cuFFT on the CPU with the own FFT of src/fft.hpp, for the plans GEM-PIC-3D makes
// (cufftPlanMany of rank 1 or 2, contiguous rows, Z2Z). Semantics as in cuFFT: CUFFT_FORWARD = -1,
// CUFFT_INVERSE = +1, both unnormalized; the data are device memory.
#pragma once
#include <map>
#include <vector>
#include "cuda_runtime.h"
#include "../fft.hpp"

typedef int cufftHandle;
typedef int cufftResult;
typedef int cufftType;
struct cufftDoubleComplex { double x, y; };
enum { CUFFT_SUCCESS = 0, CUFFT_INVALID_VALUE = 4, CUFFT_NOT_SUPPORTED = 16 };
enum { CUFFT_Z2Z = 0x69 };
#define CUFFT_FORWARD (-1)
#define CUFFT_INVERSE 1

namespace gemu {
struct FftPlan { int rank = 0, n0 = 0, n1 = 1, batch = 0; long dist = 0; };
inline std::map<int, FftPlan>& plans() { static std::map<int, FftPlan> m; return m; }
}  // namespace gemu

inline cufftResult cufftPlanMany(cufftHandle* h, int rank, int* n, int* inembed, int istride, int idist, int* onembed,
                                 int ostride, int odist, cufftType type, int batch) {
    if (type != CUFFT_Z2Z || (rank != 1 && rank != 2) || inembed || onembed || istride != 1 || ostride != 1 || idist != odist)
        return CUFFT_NOT_SUPPORTED;
    gemu::FftPlan p;
    p.rank = rank; p.n0 = n[0]; p.n1 = rank == 2 ? n[1] : 1; p.batch = batch; p.dist = idist;
    if (p.dist < long(p.n0) * p.n1) return CUFFT_INVALID_VALUE;
    const int id = int(gemu::plans().size()) + 1;
    gemu::plans()[id] = p;
    *h = id;
    return CUFFT_SUCCESS;
}
inline cufftResult cufftDestroy(cufftHandle h) { gemu::plans().erase(h); return CUFFT_SUCCESS; }
// n = {n0, n1} row-major: n1 is the contiguous dimension
inline cufftResult cufftExecZ2Z(cufftHandle h, cufftDoubleComplex* in, cufftDoubleComplex* out, int dir) {
    auto it = gemu::plans().find(h);
    if (it == gemu::plans().end() || in != out) return CUFFT_INVALID_VALUE;
    const gemu::FftPlan p = it->second;
    gemu::Open o;
    auto* z = reinterpret_cast<std::complex<double>*>(in);
    const bool inv = dir == CUFFT_INVERSE;
    if (p.rank == 1) {
        FFT1 f(p.n0);
        for (int b = 0; b < p.batch; ++b) f.run(z + b * p.dist, inv);
    } else {
        FFT1 f0(p.n0), f1(p.n1);
        std::vector<std::complex<double>> col(p.n0);
        for (int b = 0; b < p.batch; ++b) {
            std::complex<double>* a = z + b * p.dist;
            for (int r = 0; r < p.n0; ++r) f1.run(a + long(r) * p.n1, inv);
            for (int c = 0; c < p.n1; ++c) {
                for (int r = 0; r < p.n0; ++r) col[r] = a[long(r) * p.n1 + c];
                f0.run(col.data(), inv);
                for (int r = 0; r < p.n0; ++r) a[long(r) * p.n1 + c] = col[r];
            }
        }
    }
    return CUFFT_SUCCESS;
}
