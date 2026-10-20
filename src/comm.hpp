// Thin MPI layer (domain decomposition along zeta). Without -DGEMPIC_MPI it is a single-rank stub.
#pragma once
#include <cstddef>
#include <deque>
#include <stdexcept>
#include <vector>
#ifdef GEMPIC_MPI
#include <mpi.h>
#endif

class Comm {
public:
    int rank = 0, size = 1;
    void init(int* argc, char*** argv);
    void finalize();
    // point to point: sends are non-blocking (the buffer is kept until wait_sends), receives blocking
    template <class T> void isend(int dest, int tag, const T* p, size_t count) {
        isend_bytes(dest, tag, p, count * sizeof(T));
    }
    template <class T> void recv(int src, int tag, T* p, size_t count) { recv_bytes(src, tag, p, count * sizeof(T)); }
    template <class T> std::vector<T> recv_vec(int src, int tag) {       // message of unknown length
        const size_t bytes = probe_bytes(src, tag);
        std::vector<T> v(bytes / sizeof(T));
        recv_bytes(src, tag, v.data(), bytes);
        return v;
    }
    void wait_sends();
    // collectives (called in the same order on all ranks)
    void allreduce_sum(double* v, int n);
    double allreduce_max(double v);
    int iallreduce_sum(double* v, int n);      // non-blocking, in place; returns a handle for wait()
    void wait(int handle);
    void barrier();
    std::vector<char> gather_bytes(const void* p, size_t bytes);   // concatenated on rank 0
    void bcast_int(long* v, int n);
private:
    void isend_bytes(int dest, int tag, const void* p, size_t bytes);
    void recv_bytes(int src, int tag, void* p, size_t bytes);
    size_t probe_bytes(int src, int tag);
#ifdef GEMPIC_MPI
    std::deque<std::vector<char>> sbuf_;
    std::vector<MPI_Request> sreq_;
    std::deque<MPI_Request> coll_;
    int coll_base_ = 0;
#endif
};
