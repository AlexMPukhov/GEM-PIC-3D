// Checkpoints: the complete state after a step, one binary file per rank (<output.dir>/checkpoint_NNNNNN_rRRRR.bin),
// in double precision as in memory; a restart (main: --restart latest | NNNNNN) needs the same deck grid and the
// same number of ranks. Stored: level, step, window offsets, counters; the forward fields R, E_z, B_z, L of the
// own slices, the real fields with the ghost slices, the half-cell currents of level s0, the streaming current of
// the last march (prediction), the beam list, the particles that join the next march, and the completed centroid
// sums of the transverse window.
#include "sim_internal.hpp"

namespace {
const char MAGIC[8] = {'G', '3', 'D', 'C', 'K', 'P', 'T', '1'};
struct Header {
    char magic[8];
    int ranks, rank, N, a, b, ns, nk, nx, ny, plasma;
    long istep, lost, trapped, restreamed;
    double s, X0, Y0, beta_b, gm2_b;
    int stop, ncent;
};
template <class T> void put(FILE* f, const std::vector<T>& v) {
    const unsigned long long n = v.size();
    if (std::fwrite(&n, sizeof n, 1, f) != 1 || (n && std::fwrite(v.data(), sizeof(T), n, f) != n))
        throw std::runtime_error("checkpoint: write error");
}
template <class T> std::vector<T> get(FILE* f) {
    unsigned long long n = 0;
    if (std::fread(&n, sizeof n, 1, f) != 1) throw std::runtime_error("checkpoint: file too short");
    std::vector<T> v(n);
    if (n && std::fread(v.data(), sizeof(T), n, f) != n) throw std::runtime_error("checkpoint: file too short");
    return v;
}
}  // namespace

std::string Simulation::checkpoint_file(long step, int rank) const {
    char fn[96];
    std::snprintf(fn, sizeof fn, "/checkpoint_%06ld_r%04d.bin", step, rank);
    return cfg.dir + fn;
}

// collective (the pending centroid reductions are completed)
void Simulation::write_checkpoint() {
    for (auto& c : cent_) { comm.wait(c.first); c.first = -1; }
    Header h{};
    std::memcpy(h.magic, MAGIC, 8);
    h.ranks = comm.size; h.rank = comm.rank; h.N = N; h.a = a; h.b = b; h.ns = ns; h.nk = nk; h.nx = cfg.nx; h.ny = cfg.ny;
    h.plasma = cfg.plasma_on; h.istep = istep; h.lost = lost_total; h.trapped = trapped_total; h.restreamed = restreamed_total;
    h.s = s; h.X0 = X0; h.Y0 = Y0; h.beta_b = beta_b; h.gm2_b = gm2_b; h.stop = stop_; h.ncent = int(cent_.size());
    const std::string fn = checkpoint_file(istep, comm.rank), tmp = fn + ".tmp";
    FILE* f = std::fopen(tmp.c_str(), "wb");
    if (!f) throw std::runtime_error("cannot write " + tmp);
    std::fwrite(&h, sizeof h, 1, f);
    for (Buf<cpx>* p : {&Rx, &Ry, &Ez, &Bz, &Lx, &Ly, &j0x, &j0y, &js0x, &js0y, &js0z}) put(f, p->host());
    put(f, F.host());
    put(f, host_particles());
    put(f, joining_);
    for (const auto& c : cent_) put(f, c.second);
    std::fclose(f);
    std::rename(tmp.c_str(), fn.c_str());
    comm.barrier();
}

// instead of init(): the state of the checkpoint after step `step`
void Simulation::read_checkpoint(long step) {
    const std::string fn = checkpoint_file(step, comm.rank);
    FILE* f = std::fopen(fn.c_str(), "rb");
    if (!f) throw std::runtime_error("cannot read " + fn);
    Header h{};
    if (std::fread(&h, sizeof h, 1, f) != 1 || std::memcmp(h.magic, MAGIC, 8))
        throw std::runtime_error(fn + ": not a GEM-PIC-3D checkpoint");
    if (h.ranks != comm.size) throw std::runtime_error(fn + ": written by " + std::to_string(h.ranks) + " ranks; restart with as many");
    if (h.N != N || h.a != a || h.b != b || h.nx != cfg.nx || h.ny != cfg.ny || h.plasma != int(cfg.plasma_on))
        throw std::runtime_error(fn + ": the grid (nx, ny, zeta slices) or the plasma differ from the deck");
    istep = h.istep; lost_total = h.lost; trapped_total = h.trapped; restreamed_total = h.restreamed;
    s = h.s; X0 = h.X0; Y0 = h.Y0; beta_b = h.beta_b; gm2_b = h.gm2_b; stop_ = h.stop;
    for (Buf<cpx>* p : {&Rx, &Ry, &Ez, &Bz, &Lx, &Ly, &j0x, &j0y, &js0x, &js0y, &js0z}) {
        const std::vector<cpx> v = get<cpx>(f);
        if (v.size() != p->size()) throw std::runtime_error(fn + ": array sizes differ from the deck");
        if (!v.empty()) p->from_host(v.data(), 0, v.size());
    }
    const std::vector<float> fv = get<float>(f);
    if (fv.size() != F.size()) throw std::runtime_error(fn + ": array sizes differ from the deck");
    F.from_host(fv.data(), 0, fv.size());
    set_particles(get<Particle>(f));
    joining_ = get<Particle>(f);
    cent_.clear();
    for (int k = 0; k < h.ncent; ++k) cent_.emplace_back(-1, get<double>(f));
    std::fclose(f);
}

// the newest step for which every rank has a checkpoint file (-1: none); collective
long Simulation::latest_checkpoint() {
    long best = -1;
    if (comm.rank == 0) {
        namespace fs = std::filesystem;
        if (fs::is_directory(cfg.dir))
            for (const auto& e : fs::directory_iterator(cfg.dir)) {
                const std::string n = e.path().filename().string();
                long st; int r;
                if (std::sscanf(n.c_str(), "checkpoint_%ld_r%d.bin", &st, &r) != 2 || r != 0 || st <= best) continue;
                bool all = true;
                for (int q = 1; q < comm.size && all; ++q) all = fs::exists(checkpoint_file(st, q));
                if (all && n == fs::path(checkpoint_file(st, 0)).filename().string()) best = st;
            }
    }
    comm.bcast_int(&best, 1);
    return best;
}
