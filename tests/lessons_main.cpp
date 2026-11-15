// Only the lesson checks (tests/test_lessons.cpp): for quick runs while developing.
#include "test_util.hpp"
int failures = 0;
Comm comm;
int main(int argc, char** argv) {
    comm.init(&argc, &argv);
    const bool lw = argc > 1 && std::string(argv[1]) == "long";
    if (!lw) { test_static_bunch(); for (double dz : {50.0, 500.0}) test_beam_drift(dz); }
    else if (argc > 2) test_long_wake(true, std::atoi(argv[2]));
    else { test_long_wake(true, 2); test_long_wake(true, 4); test_long_wake(false, 2); }
    std::printf("%s: %d failure(s)\n", failures ? "FAILED" : "ALL PASSED", failures);
    comm.finalize();
    return failures ? 1 : 0;
}
