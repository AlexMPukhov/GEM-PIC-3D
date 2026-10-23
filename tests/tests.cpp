// Tests of GEM-PIC-3D against independent references (tests/test_*.cpp, common part in tests/test_util.hpp).
#include "test_util.hpp"

int failures = 0;
Comm comm;

int main(int argc, char** argv) {
    comm.init(&argc, &argv);
    std::printf("GEM-PIC-3D tests (%s build)\n", GEMPIC_GPU ? "CUDA" : "CPU");
    test_fft_pair();
    test_fft_sizes();
    test_fft_batched(16, 8);
    test_fft_batched(12, 10);
    test_grids();
    for (const char* kind : {"default", "zeta", "y", "sizes"}) test_dispersion(kind);
    test_diffraction("Fourier", 0);
    test_diffraction("free-hand 1 -> 4", test_diffraction("constant step", 0));
    for (const char* pol : {"s", "p"}) test_plane_wave(pol);
    for (const char* kind : {"uniform", "y", "zeta", "120 x 120 modes"}) test_gauss(kind);
    for (bool b : {false, true}) test_equilibrium(b);
    for (bool b : {false, true}) test_sponge(b);
    for (bool b : {true, false}) test_constant_tables(b);
    for (bool b : {true, false}) test_plasma_wake_1d(b);
    test_plasma_wake_beam();
    test_smoothing_divergence();
    test_streaming_kinematics();
    std::printf("%s: %d failure(s)\n", failures ? "FAILED" : "ALL PASSED", failures);
    comm.finalize();
    return failures ? 1 : 0;
}
