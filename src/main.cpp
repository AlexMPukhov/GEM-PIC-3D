#include <chrono>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <stdexcept>
#include "sim.hpp"

int main(int argc, char** argv) {
    try {
        if (argc < 2) { std::fprintf(stderr, "usage: gempic3d deck [--check] [block.key=value ...]\n"); return 1; }
        Deck d = Deck::read(argv[1]);
        bool check = false;
        for (int i = 2; i < argc; ++i) {
            if (!std::strcmp(argv[i], "--check")) check = true;
            else d.override_from(argv[i]);
        }
        Config c = Config::from_deck(d);
        c.summary(stdout);
        const double pts = double(c.nx) * c.ny * c.nzeta;
        std::printf("  memory: about %.2f GB (fields) + %.2f GB (particles)\n", pts * 148 / 1e9, c.nmacro * 56 / 1e9);
        if (check) return 0;
        std::filesystem::create_directories(c.dir);
        Simulation sim(c);
        sim.init();
        FILE* f = std::fopen((c.dir + "/diag.txt").c_str(), "w");
        if (!f) throw std::runtime_error("cannot write " + c.dir + "/diag.txt");
        sim.diagnostics(f, true);
        sim.diagnostics(stdout, true);
        if (c.dump_particles) sim.dump_particles(c.dir + "/particles_start.txt");
        auto t0 = std::chrono::steady_clock::now();
        while (!sim.finished()) {
            sim.step();
            if (sim.istep % c.every == 0) { sim.diagnostics(f, false); sim.diagnostics(stdout, false); }
        }
        if (sim.istep % c.every) { sim.diagnostics(f, false); sim.diagnostics(stdout, false); }
        std::fclose(f);
        if (c.dump_particles) sim.dump_particles(c.dir + "/particles_end.txt");
        const double sec = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
        std::printf("done: %ld steps, %.1f s, %ld particles lost\n", sim.istep, sec, sim.lost);
    } catch (const std::exception& e) {
        std::fprintf(stderr, "error: %s\n", e.what());
        return 2;
    }
}
