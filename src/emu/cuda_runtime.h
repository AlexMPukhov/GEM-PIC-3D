// GPU emulator (make EMU=1): a CPU stand-in for the parts of the CUDA runtime that GEM-PIC-3D uses, so that the
// GPU code paths of the sources (GEMPIC_GPU = 1: device buffers, kernel launches, cuFFT, the per-mode march,
// device-side atomics) run and can be tested on a machine without a GPU.
//
// Device memory is kept apart from host memory as on a real GPU: every cudaMalloc is its own mmap region,
// readable and writable only inside a kernel launch, cudaMemcpy, cudaMemset or a cuFFT call, and PROT_NONE
// otherwise. Host code that dereferences a device pointer (a bug that crashes or reads garbage on a GPU) stops
// with a message naming the address and a backtrace. Kernel threads run in a scrambled order (index
// i = k * P mod n with a large prime P), so code that relies on the order of the threads gives different
// results. Not covered: device code that reads host memory (captured host pointers) and compile-time
// restrictions of nvcc (check those with a real nvcc build).
#pragma once
#include <cstddef>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <csignal>
#include <map>
#include <mutex>
#include <execinfo.h>
#include <sys/mman.h>
#include <unistd.h>

typedef int cudaError_t;
enum { cudaSuccess = 0, cudaErrorMemoryAllocation = 2, cudaErrorInvalidValue = 1 };
enum cudaMemcpyKind { cudaMemcpyHostToHost = 0, cudaMemcpyHostToDevice = 1, cudaMemcpyDeviceToHost = 2, cudaMemcpyDeviceToDevice = 3 };

namespace gemu {
struct State {
    std::map<char*, size_t> regions;   // start -> mapped bytes
    int depth = 0;                     // > 0: device memory accessible
    long launches = 0;
    std::mutex mu;
};
inline State& st() { static State s; return s; }
inline void protect_all(int prot) {
    for (auto& r : st().regions)
        if (mprotect(r.first, r.second, prot) != 0) { std::perror("gemu: mprotect"); std::abort(); }
}
inline void on_segv(int, siginfo_t* si, void*) {
    char* a = static_cast<char*>(si->si_addr);
    bool dev = false;
    for (auto& r : st().regions) if (a >= r.first && a < r.first + r.second) dev = true;
    char msg[160];
    const int len = std::snprintf(msg, sizeof msg, dev ? "\nGPU emulator: host code accessed device memory at %p\n"
                                                       : "\nGPU emulator: segmentation fault at %p\n", (void*)a);
    if (write(2, msg, size_t(len)) < 0) {}
    void* bt[48];
    backtrace_symbols_fd(bt, backtrace(bt, 48), 2);
    _exit(3);
}
inline void install() {
    static bool done = false;
    if (done) return;
    done = true;
    struct sigaction sa;
    std::memset(&sa, 0, sizeof sa);
    sa.sa_sigaction = on_segv; sa.sa_flags = SA_SIGINFO;
    sigaction(SIGSEGV, &sa, nullptr);
    sigaction(SIGBUS, &sa, nullptr);
}
// device memory is open while at least one device operation is running
inline void enter() { install(); if (st().depth++ == 0) protect_all(PROT_READ | PROT_WRITE); }
inline void leave() { if (--st().depth == 0) protect_all(PROT_NONE); }
struct Open { Open() { enter(); } ~Open() { leave(); } };
}  // namespace gemu

inline cudaError_t cudaMalloc(void** p, size_t bytes) {
    gemu::install();
    const size_t pg = size_t(sysconf(_SC_PAGESIZE)), len = ((bytes ? bytes : 1) + pg - 1) / pg * pg;
    void* m = mmap(nullptr, len, gemu::st().depth > 0 ? PROT_READ | PROT_WRITE : PROT_NONE, MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
    if (m == MAP_FAILED) return cudaErrorMemoryAllocation;
    gemu::st().regions[static_cast<char*>(m)] = len;
    *p = m;
    return cudaSuccess;
}
template <class T> inline cudaError_t cudaMalloc(T** p, size_t bytes) { return cudaMalloc(reinterpret_cast<void**>(p), bytes); }
inline cudaError_t cudaFree(void* p) {
    if (!p) return cudaSuccess;
    auto it = gemu::st().regions.find(static_cast<char*>(p));
    if (it == gemu::st().regions.end()) return cudaErrorInvalidValue;
    munmap(it->first, it->second);
    gemu::st().regions.erase(it);
    return cudaSuccess;
}
inline cudaError_t cudaMemcpy(void* dst, const void* src, size_t bytes, cudaMemcpyKind) {
    gemu::Open o; std::memcpy(dst, src, bytes); return cudaSuccess;
}
inline cudaError_t cudaMemset(void* p, int v, size_t bytes) { gemu::Open o; std::memset(p, v, bytes); return cudaSuccess; }
inline cudaError_t cudaDeviceSynchronize() { return cudaSuccess; }
inline cudaError_t cudaGetLastError() { return cudaSuccess; }
inline const char* cudaGetErrorString(cudaError_t e) { return e == cudaSuccess ? "no error" : "emulated CUDA error"; }

namespace gemu {
// kernel launch of n threads (blocks of 256 as in portable.hpp), threads in a scrambled order
template <class F> inline void launch(long n, F f) {
    Open o;
    ++st().launches;
    const long P = 1000003;                         // prime: k -> k*P mod n is a permutation unless n % P == 0
    const bool perm = n % P != 0;
#pragma omp parallel for schedule(static)
    for (long k = 0; k < n; ++k) f(perm ? long((__int128)k * P % n) : k);
}
}  // namespace gemu
