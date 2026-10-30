#include <chrono>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <sstream>
#include <stdexcept>
#include "sim.hpp"

// rank 0: a table of the run (diag.txt, ionization.txt); on a restart after step `step` the lines of later steps
// are cut and the table continues
static FILE* open_table(const std::string& file, long step) {
    if (step < 0) return std::fopen(file.c_str(), "w");
    std::ifstream in(file);
    std::string keep, line;
    while (std::getline(in, line)) {
        long st;
        std::istringstream is(line);
        if (line.empty() || line[0] == '#' || ((is >> st) && st <= step)) keep += line + "\n";
    }
    FILE* f = std::fopen(file.c_str(), "w");
    if (f) std::fputs(keep.c_str(), f);
    return f;
}

int main(int argc, char** argv) {
    Comm comm;
    comm.init(&argc, &argv);
    int rc = 0;
    try {
        if (argc < 2) throw std::runtime_error("usage: gempic3d deck [--check] [--restart latest|STEP] [block.key=value ...]");
        Deck d = Deck::read(argv[1]);
        bool check = false;
        long restart = -2;                                 // -2: none, -1: the latest checkpoint
        for (int i = 2; i < argc; ++i) {
            if (!std::strcmp(argv[i], "--check")) check = true;
            else if (!std::strcmp(argv[i], "--restart")) {
                if (i + 1 >= argc) throw std::runtime_error("--restart needs latest or a step");
                restart = !std::strcmp(argv[++i], "latest") ? -1 : std::stol(argv[i]);
            } else d.override_from(argv[i]);
        }
        Config c = Config::from_deck(d);
        if (comm.rank == 0) {
            c.summary(stdout, comm.size);
            const double pts = double(c.nx) * c.ny * c.nzeta / comm.size;
            std::printf("  memory per rank: about %.2f GB (fields) + %.2f GB (particles)\n", pts * 256 / 1e9,
                        c.nmacro * 72.0 / comm.size / 1e9);
        }
        if (!check) {
            if (comm.rank == 0) std::filesystem::create_directories(c.dir);
            comm.barrier();
            Simulation sim(c, comm);
            long rs = -1;                                  // restart step
            if (restart != -2) {
                rs = restart == -1 ? sim.latest_checkpoint() : restart;
                if (rs < 0) throw std::runtime_error("--restart latest: no complete checkpoint in " + c.dir);
                sim.read_checkpoint(rs);
                if (comm.rank == 0) std::printf("restart after step %ld (z = %.6g mm)\n", rs, sim.s * c.Lu * 1e3);
            } else sim.init();
            FILE* f = comm.rank == 0 ? open_table(c.dir + "/diag.txt", rs) : nullptr;
            if (comm.rank == 0 && !f) throw std::runtime_error("cannot write " + c.dir + "/diag.txt");
            if (rs < 0) sim.diagnostics(f, true);
            FILE* fi = comm.rank == 0 && c.ionizing() ? open_table(c.dir + "/ionization.txt", rs) : nullptr;
            if (rs < 0) sim.write_ionization(fi, true);
            char fn[64];
            // diag.txt and GUI frames every output.every steps (0: start and end only); output groups at their own periods
            auto frame = [&]() {
                if (c.frames) { std::snprintf(fn, sizeof fn, "/frame_%05ld.bin", sim.istep); sim.write_frame(c.dir + fn); }
            };
            if (rs < 0) {
                frame();
                sim.write_outputs(true);
                if (c.dump_particles) sim.dump_particles(c.dir + "/particles_start.txt");
            }
            auto t0 = std::chrono::steady_clock::now();
            long steps = 0;
            while (!sim.finished()) {
                sim.step();
                ++steps;
                sim.write_ionization(fi, false);
                if (c.every > 0 && sim.istep % c.every == 0) { sim.diagnostics(f, false); frame(); }
                sim.write_outputs(false);
                if (c.checkpoint_every > 0 && sim.istep % c.checkpoint_every == 0) sim.write_checkpoint();
                if (c.walltime_hours > 0) {                // stop 1.5 mean step times before the limit (rank 0 decides)
                    long stop = 0;
                    if (comm.rank == 0) {
                        const double el = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
                        stop = el + 1.5 * el / steps > 3600 * c.walltime_hours;
                    }
                    comm.bcast_int(&stop, 1);
                    if (stop) {
                        if (!(c.checkpoint_every > 0 && sim.istep % c.checkpoint_every == 0)) sim.write_checkpoint();
                        if (comm.rank == 0) std::printf("wall-time limit: checkpoint after step %ld, stop (exit code 3)\n", sim.istep);
                        rc = 3;
                        break;
                    }
                }
            }
            if (c.every <= 0 || sim.istep % c.every) { sim.diagnostics(f, false); frame(); }   // the last step
            if (c.dump_particles) sim.dump_particles(c.dir + "/particles_end.txt");
            if (comm.rank == 0) {
                std::fclose(f);
                if (fi) std::fclose(fi);
                const double sec = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
                std::printf("done: %ld steps, %.1f s\n", sim.istep, sec);
            }
        }
    } catch (const std::exception& e) {
        std::fprintf(stderr, "error (rank %d): %s\n", comm.rank, e.what());
        rc = 2;
    }
    comm.finalize();
    return rc;
}
