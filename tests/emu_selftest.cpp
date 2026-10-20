// Self-test of the GPU emulator (make EMU=1 emu-selftest): device memory must be reachable in kernels and
// copies, and a host dereference of a device pointer must stop the program (exit code 3) with a message.
#include <cstdio>
#include <cstring>
#include <sys/wait.h>
#include <unistd.h>
#include "portable.hpp"

int main() {
    Buf<double> b(1000);
    double* d = b.data();
    par_for(1000, [=] HD(long i) { d[i] = 2.0 * i; });
    std::vector<double> h = b.host();
    bool ok = h[999] == 1998.0 && h[0] == 0.0;
    std::printf("emu: kernel writes, copy back                      %s\n", ok ? "PASS" : "FAIL");
    // order of the threads: scrambled
    Buf<long> ord(1); Buf<long> first(1000);
    long* o = ord.data(); long* f = first.data();
    par_for(1000, [=] HD(long i) { f[i] = o[0]++; });
    std::vector<long> fv = first.host();
    const bool scr = fv[0] != 0 || fv[1] != 1;
    std::printf("emu: threads run out of order                      %s\n", scr ? "PASS" : "FAIL");
    ok = ok && scr;
    const pid_t pid = fork();
    if (pid == 0) {                     // child: the forbidden access
        std::fclose(stderr);
        volatile double x = d[5];
        (void)x;
        _exit(0);
    }
    int status = 0;
    waitpid(pid, &status, 0);
    const bool caught = WIFEXITED(status) && WEXITSTATUS(status) == 3;
    std::printf("emu: host access to device memory is caught        %s\n", caught ? "PASS" : "FAIL");
    ok = ok && caught;
    std::printf("%s\n", ok ? "EMU SELFTEST PASSED" : "EMU SELFTEST FAILED");
    return ok ? 0 : 1;
}
