// Tests of GEM-PIC-3D against independent references (tests/test_*.cpp, common part in tests/test_util.hpp).
#include "test_util.hpp"

int failures = 0;
Comm comm;

int main(int argc, char** argv) {
    comm.init(&argc, &argv);
    std::printf("GEM-PIC-3D tests (%s build)\n", GEMPIC_GPU ? "CUDA" : "CPU");
    test_fft(16, 8);
    test_fft(12, 10);
    test_grids();
    test_dispersion("default");
    test_dispersion("zeta");
    test_dispersion("y");
    test_dispersion("sizes");
    test_diffraction("Fourier", 0);
    test_diffraction("free-hand 1 -> 4", test_diffraction("constant step", 0));
    test_plane_wave("s");
    test_plane_wave("p");
    test_gauss("uniform");
    test_gauss("y");
    test_gauss("zeta");
    test_gauss("120 x 120 modes");
    test_equilibrium(false);
    test_equilibrium(true);
    test_sponge(false);
    test_sponge(true);
    test_constant_tables(true);
    test_constant_tables(false);
    test_plasma_wake_1d(true);
    test_plasma_wake_1d(false);
    test_plasma_wake_beam();
    test_smoothing_divergence();
    test_streaming_kinematics();
    std::printf("%s: %d failure(s)\n", failures ? "FAILED" : "ALL PASSED", failures);
    comm.finalize();
    return failures ? 1 : 0;
}
