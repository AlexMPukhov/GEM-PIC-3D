// Portability layer: every kernel is written once as a lambda and runs either as an OpenMP loop
// (CPU build) or as a CUDA kernel (nvcc build, --extended-lambda). Device memory is Buf<T>.
#pragma once
#include <cmath>
#include <cstddef>
#include <cstring>
#include <stdexcept>
#include <string>
#include <vector>

#if defined(__CUDACC__)
#include <cuda_runtime.h>
#define HD __host__ __device__
#define GEMPIC_GPU 1
#else
#define HD
#define GEMPIC_GPU 0
#endif

// ---------------------------------------------------------------- complex numbers (layout of cufftDoubleComplex)
struct cpx { double re, im; };
HD inline cpx C(double r, double i = 0) { return cpx{r, i}; }
HD inline cpx operator+(cpx a, cpx b) { return cpx{a.re + b.re, a.im + b.im}; }
HD inline cpx operator-(cpx a, cpx b) { return cpx{a.re - b.re, a.im - b.im}; }
HD inline cpx operator-(cpx a) { return cpx{-a.re, -a.im}; }
HD inline cpx operator*(cpx a, cpx b) { return cpx{a.re * b.re - a.im * b.im, a.re * b.im + a.im * b.re}; }
HD inline cpx operator*(double s, cpx a) { return cpx{s * a.re, s * a.im}; }
HD inline cpx operator*(cpx a, double s) { return cpx{s * a.re, s * a.im}; }
HD inline cpx operator/(cpx a, double s) { return cpx{a.re / s, a.im / s}; }
HD inline cpx operator/(cpx a, cpx b) {
    const double d = b.re * b.re + b.im * b.im;
    return cpx{(a.re * b.re + a.im * b.im) / d, (a.im * b.re - a.re * b.im) / d};
}
HD inline cpx conj(cpx a) { return cpx{a.re, -a.im}; }
HD inline cpx ix(double k) { return cpx{0, k}; }          // i k
HD inline double abs2(cpx a) { return a.re * a.re + a.im * a.im; }

// ---------------------------------------------------------------- errors
inline void gpu_check(int code, const char* what) {
#if GEMPIC_GPU
    if (code != 0) throw std::runtime_error(std::string("CUDA error in ") + what + ": " + cudaGetErrorString(cudaError_t(code)));
#else
    (void)code; (void)what;
#endif
}

// ---------------------------------------------------------------- kernels
#if GEMPIC_GPU
template <class F> __global__ void gempic_kernel(long n, F f) {
    const long i = long(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i < n) f(i);
}
template <class F> inline void par_for(long n, F f) {
    if (n <= 0) return;
    const int t = 256;
    gempic_kernel<<<unsigned((n + t - 1) / t), t>>>(n, f);
    gpu_check(cudaGetLastError(), "kernel launch");
}
inline void device_sync() { gpu_check(cudaDeviceSynchronize(), "synchronize"); }
#else
template <class F> inline void par_for(long n, F f) {
#pragma omp parallel for schedule(static)
    for (long i = 0; i < n; ++i) f(i);
}
inline void device_sync() {}
#endif

HD inline void atomic_add(double* p, double v) {
#if defined(__CUDA_ARCH__)
    atomicAdd(p, v);
#elif defined(_OPENMP)
#pragma omp atomic
    *p += v;
#else
    *p += v;
#endif
}

// ---------------------------------------------------------------- device buffer
template <class T> class Buf {
public:
    Buf() = default;
    explicit Buf(size_t n) { resize(n); }
    ~Buf() { release(); }
    Buf(const Buf&) = delete;
    Buf& operator=(const Buf&) = delete;
    Buf(Buf&& o) noexcept : p_(o.p_), n_(o.n_) { o.p_ = nullptr; o.n_ = 0; }
    Buf& operator=(Buf&& o) noexcept { if (this != &o) { release(); p_ = o.p_; n_ = o.n_; o.p_ = nullptr; o.n_ = 0; } return *this; }
    void resize(size_t n) {
        release(); n_ = n;
        if (n_ == 0) return;
#if GEMPIC_GPU
        gpu_check(cudaMalloc(&p_, n_ * sizeof(T)), "cudaMalloc");
#else
        p_ = static_cast<T*>(::operator new(n_ * sizeof(T)));
#endif
        zero();
    }
    void zero() {
        if (!n_) return;
#if GEMPIC_GPU
        gpu_check(cudaMemset(p_, 0, n_ * sizeof(T)), "cudaMemset");
#else
        std::memset(static_cast<void*>(p_), 0, n_ * sizeof(T));
#endif
    }
    T* data() { return p_; }
    const T* data() const { return p_; }
    size_t size() const { return n_; }
    // copies between this buffer (offset, count) and host memory
    void to_host(T* h, size_t off, size_t cnt) const {
        if (!cnt) return;
#if GEMPIC_GPU
        gpu_check(cudaMemcpy(h, p_ + off, cnt * sizeof(T), cudaMemcpyDeviceToHost), "cudaMemcpy D2H");
#else
        std::memcpy(static_cast<void*>(h), p_ + off, cnt * sizeof(T));
#endif
    }
    void from_host(const T* h, size_t off, size_t cnt) {
        if (!cnt) return;
#if GEMPIC_GPU
        gpu_check(cudaMemcpy(p_ + off, h, cnt * sizeof(T), cudaMemcpyHostToDevice), "cudaMemcpy H2D");
#else
        std::memcpy(static_cast<void*>(p_ + off), h, cnt * sizeof(T));
#endif
    }
    std::vector<T> host() const { std::vector<T> v(n_); to_host(v.data(), 0, n_); return v; }
    void upload(const std::vector<T>& v) { if (v.size() != n_) resize(v.size()); from_host(v.data(), 0, v.size()); }
private:
    void release() {
        if (!p_) return;
#if GEMPIC_GPU
        cudaFree(p_);
#else
        ::operator delete(p_);
#endif
        p_ = nullptr; n_ = 0;
    }
    T* p_ = nullptr;
    size_t n_ = 0;
};
