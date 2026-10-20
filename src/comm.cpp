#include "comm.hpp"
#include <cstring>
#include <map>

#ifdef GEMPIC_MPI
namespace { std::map<int, MPI_Request> g_coll; int g_next = 0; }

void Comm::init(int* argc, char*** argv) {
    int flag = 0;
    MPI_Initialized(&flag);
    if (!flag) MPI_Init(argc, argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
}
void Comm::finalize() { wait_sends(); MPI_Finalize(); }
void Comm::isend_bytes(int dest, int tag, const void* p, size_t bytes) {
    sbuf_.emplace_back(static_cast<const char*>(p), static_cast<const char*>(p) + bytes);
    sreq_.emplace_back();
    MPI_Isend(sbuf_.back().data(), int(bytes), MPI_BYTE, dest, tag, MPI_COMM_WORLD, &sreq_.back());
}
void Comm::recv_bytes(int src, int tag, void* p, size_t bytes) {
    MPI_Status st;
    MPI_Recv(p, int(bytes), MPI_BYTE, src, tag, MPI_COMM_WORLD, &st);
    int got = 0;
    MPI_Get_count(&st, MPI_BYTE, &got);
    if (size_t(got) != bytes) throw std::runtime_error("MPI message of unexpected size");
}
size_t Comm::probe_bytes(int src, int tag) {
    MPI_Status st;
    MPI_Probe(src, tag, MPI_COMM_WORLD, &st);
    int got = 0;
    MPI_Get_count(&st, MPI_BYTE, &got);
    return size_t(got);
}
void Comm::wait_sends() {
    if (!sreq_.empty()) MPI_Waitall(int(sreq_.size()), sreq_.data(), MPI_STATUSES_IGNORE);
    sreq_.clear(); sbuf_.clear();
}
void Comm::allreduce_sum(double* v, int n) { MPI_Allreduce(MPI_IN_PLACE, v, n, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD); }
double Comm::allreduce_max(double v) { MPI_Allreduce(MPI_IN_PLACE, &v, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD); return v; }
int Comm::iallreduce_sum(double* v, int n) {
    const int h = g_next++;
    MPI_Iallreduce(MPI_IN_PLACE, v, n, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD, &g_coll[h]);
    return h;
}
void Comm::wait(int h) {
    auto it = g_coll.find(h);
    if (it == g_coll.end()) return;
    MPI_Wait(&it->second, MPI_STATUS_IGNORE);
    g_coll.erase(it);
}
void Comm::barrier() { MPI_Barrier(MPI_COMM_WORLD); }
std::vector<char> Comm::gather_bytes(const void* p, size_t bytes) {
    int b = int(bytes);
    std::vector<int> counts(size), displs(size);
    MPI_Gather(&b, 1, MPI_INT, counts.data(), 1, MPI_INT, 0, MPI_COMM_WORLD);
    std::vector<char> out;
    if (rank == 0) {
        long tot = 0;
        for (int r = 0; r < size; ++r) { displs[r] = int(tot); tot += counts[r]; }
        out.resize(size_t(tot));
    }
    MPI_Gatherv(p, b, MPI_BYTE, out.data(), counts.data(), displs.data(), MPI_BYTE, 0, MPI_COMM_WORLD);
    return out;
}
void Comm::bcast_int(long* v, int n) { MPI_Bcast(v, n, MPI_LONG, 0, MPI_COMM_WORLD); }

#else   // single rank

void Comm::init(int*, char***) { rank = 0; size = 1; }
void Comm::finalize() {}
void Comm::isend_bytes(int, int, const void*, size_t) { throw std::runtime_error("send without MPI"); }
void Comm::recv_bytes(int, int, void*, size_t) { throw std::runtime_error("receive without MPI"); }
size_t Comm::probe_bytes(int, int) { throw std::runtime_error("probe without MPI"); }
void Comm::wait_sends() {}
void Comm::allreduce_sum(double*, int) {}
double Comm::allreduce_max(double v) { return v; }
int Comm::iallreduce_sum(double*, int) { return 0; }
void Comm::wait(int) {}
void Comm::barrier() {}
std::vector<char> Comm::gather_bytes(const void* p, size_t bytes) {
    return std::vector<char>(static_cast<const char*>(p), static_cast<const char*>(p) + bytes);
}
void Comm::bcast_int(long*, int) {}
#endif
