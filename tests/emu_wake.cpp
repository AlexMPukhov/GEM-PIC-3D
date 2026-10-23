// GPU emulator check of the 3D beam wake (tests/test_plasma.cpp, beam_wake_axis) on a reduced grid
// (32 x 32 modes, dzeta = 1, 10 steps; the full check needs > 30 min in the emulator). Build it once without and
// once with -DGEMPIC_EMU and compare the two output files: E_z on the axis must agree bit for bit.
//   S="src/deck.cpp src/grid.cpp src/sim_*.cpp src/comm.cpp src/openpmd.cpp tests/test_plasma.cpp"
//   g++ -O3 -std=c++17 -fopenmp -Isrc -Itests -o cmp_cpu $S tests/emu_wake.cpp && ./cmp_cpu wake_cpu.txt
//   g++ -O3 -std=c++17 -fopenmp -DGEMPIC_EMU -Isrc -Itests -o cmp_emu $S tests/emu_wake.cpp && ./cmp_emu wake_emu.txt
//   cmp wake_cpu.txt wake_emu.txt
#include "test_util.hpp"
int failures = 0; Comm comm;
// beam_wake_axis (tests/test_plasma.cpp) on a smaller grid: the same code paths, 1/24 of the work
static std::vector<double> wake(bool freehand) {
    Config c = Config::base_units(10e-6);
    const double sig = 5, sz = 5, Lw = 150, h = 1.0;
    c.nx = 32; c.Lx = 120;
    if (freehand) c.y_table({0}, {3.75}, 60); else c.y_uniform(32, 120);
    c.zeta_uniform(int(Lw / h) + 1, Lw);
    c.sponge = c.sponge_y = 0; c.move_x = c.move_y = false; c.k_filter = false;
    c.plasma_on = true; c.n0 = 0.01; c.ppc_x = 1; c.ppc_y = 1;
    const double nb = 0.01 * c.n0, Ne = nb * std::pow(2 * M_PI, 1.5) * sig * sig * sz;
    c.gamma = 1000; c.sigx = c.sigy = sig; c.sigz = sz; c.nmacro = 50000; c.zeta_center = 120;
    c.charge = Ne * c.nc * std::pow(c.Lu, 3) * 1.602176634e-19;
    c.dz = 10; c.dz_push = 10; c.freeze_bunch = true;
    Simulation sim(c, comm);
    sim.init();
    for (int k = 0; k < 10; ++k) sim.step();
    const std::vector<float> F = sim.F.host();
    const int ic = c.nx / 2, jc = c.ny / 2;
    std::vector<double> ez(c.nzeta);
    for (int n = 0; n < c.nzeta; ++n) ez[n] = field(F, sim, 2, n, size_t(jc) * c.nx + ic);
    return ez;
}
int main(int argc, char** argv) {
    comm.init(&argc, &argv);
    FILE* f = std::fopen(argv[1], "w");
    for (int fh = 0; fh < 2; ++fh) for (double v : wake(fh)) std::fprintf(f, "%d %.9e\n", fh, v);
    std::fclose(f);
    comm.finalize();
    return 0;
}
