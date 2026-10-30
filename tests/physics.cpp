// Physics checks of GEM-PIC-3D that need larger grids than make test (make physics; a few minutes on a GPU or a
// 24-core node): ion collapse in a dense bunch against a test-ion model, hydrogen and deuterium.
#include "test_util.hpp"

int failures = 0;
Comm comm;

int main(int argc, char** argv) {
    comm.init(&argc, &argv);
    std::printf("GEM-PIC-3D physics checks (%s build)\n", GEMPIC_GPU ? "CUDA" : "CPU");
    test_ion_collapse();
    std::printf("%s: %d failure(s)\n", failures ? "FAILED" : "ALL PASSED", failures);
    comm.finalize();
    return failures ? 1 : 0;
}
