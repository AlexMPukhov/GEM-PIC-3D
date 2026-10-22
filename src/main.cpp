#include <chrono>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <stdexcept>
#include "sim.hpp"

int main(int argc, char** argv) {
    Comm comm;
    comm.init(&argc, &argv);
    int rc = 0;
    try {
        if (argc < 2) throw std::runtime_error("usage: gempic3d deck [--check] [block.key=value ...]");
        Deck d = Deck::read(argv[1]);
        bool check = false;
        for (int i = 2; i < argc; ++i) {
            if (!std::strcmp(argv[i], "--check")) check = true;
            else d.override_from(argv[i]);
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
            sim.init();
            FILE* f = comm.rank == 0 ? std::fopen((c.dir + "/diag.txt").c_str(), "w") : nullptr;
            if (comm.rank == 0 && !f) throw std::runtime_error("cannot write " + c.dir + "/diag.txt");
            sim.diagnostics(f, true);
            char fn[64];
            // GUI frames every output.every steps; output groups (output, diag.<name>) at their own periods
            auto frame = [&]() {
                if (c.frames) { std::snprintf(fn, sizeof fn, "/frame_%05ld.bin", sim.istep); sim.write_frame(c.dir + fn); }
            };
            frame();
            sim.write_outputs(true);
            if (c.dump_particles) sim.dump_particles(c.dir + "/particles_start.txt");
            auto t0 = std::chrono::steady_clock::now();
            while (!sim.finished()) {
                sim.step();
                if (sim.istep % c.every == 0) { sim.diagnostics(f, false); frame(); }
                sim.write_outputs(false);
            }
            if (sim.istep % c.every) { sim.diagnostics(f, false); frame(); }
            if (c.dump_particles) sim.dump_particles(c.dir + "/particles_end.txt");
            if (comm.rank == 0) {
                std::fclose(f);
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
