// Tests: checkpoint and restart.
#include "test_util.hpp"
#include <filesystem>
#include <unistd.h>

// 24. Checkpoint: a run with a laser wake in a plasma with an up-ramp, mobile H+, ionizing N5+ and a bunch, moving
//     transverse window: 2 steps, checkpoint, restart in a new Simulation, 2 more steps, against 4 steps without
//     interruption: the fields and the beam list agree (to the rounding of the current sums of the threads).
static Config ckpt_config(const std::string& dir) {
    Config c = Config::base_units(0.8e-6);
    c.nx = 16; c.Lx = 120; c.y_uniform(16, 120); c.zeta_uniform(161, 160);
    c.sponge = c.sponge_y = 10; c.move_x = c.move_y = true; c.window_lag = 1;
    c.plasma_on = true; c.n0 = 0.004; c.ppc_x = 1; c.ppc_y = 1; c.pz_points = {0, 200}; c.pz_values = {0.5, 1};
    Config::GridLaser g; g.a0 = 2; g.kL = 1; g.waist = 25; g.tau = 30; g.zeta_c = 130;
    c.glasers.push_back(g);
    Config::IonSpecies h; h.atoms = 0.9; c.ions.push_back(h);
    Config::IonSpecies n; n.name = "N5+"; n.mass = 14.007 * 1822.888486; n.Z0 = 5; n.Zmax = 7; n.atoms = 0.01;
    n.ionization = true; n.mobile = false;
    for (double e : {14.53413, 29.60125, 47.4453, 77.4735, 97.8901, 552.06741, 667.0461377}) n.ip_eV.push_back(e);
    c.ions.push_back(n);
    Config::Bunch b; b.charge = 5e-12; b.gamma = 200; b.sigx = b.sigy = 3; b.sigz = 4; b.zeta_center = 60; b.nmacro = 4000;
    c.bunches = {b};
    c.dz = 40; c.dz_push = 40; c.dir = dir;
    return c;
}
void test_checkpoint() {
    const std::string dir = (std::filesystem::temp_directory_path() / ("gempic3d_ckpt_" + std::to_string(::getpid()))).string();
    std::filesystem::create_directories(dir);
    const Config c = ckpt_config(dir);
    Simulation A(c, comm);
    A.init();
    for (int k = 0; k < 4; ++k) A.step();
    Simulation B(c, comm);
    B.init();
    for (int k = 0; k < 2; ++k) B.step();
    B.write_checkpoint();
    Simulation C(c, comm);
    C.read_checkpoint(2);
    for (int k = 0; k < 2; ++k) C.step();
    const std::vector<float> fa = A.F.host(), fc = C.F.host();
    double d = 0, m = 0;
    for (size_t i = 0; i < fa.size(); ++i) { d = std::max(d, double(std::fabs(fa[i] - fc[i]))); m = std::max(m, double(std::fabs(fa[i]))); }
    std::vector<Particle> pa = A.host_particles(), pc = C.host_particles();
    double dp = pa.size() == pc.size() ? 0 : 1e300;
    for (size_t i = 0; i < pa.size() && i < pc.size(); ++i) dp = std::max(dp, std::fabs(pa[i].pz - pc[i].pz) / std::fabs(pa[i].pz));
    check(m > 0 && d <= 1e-5 * m, "checkpoint: restart after 2 of 4 steps = run without interruption (fields)", d / m, 0, 1e-5);
    check(dp < 1e-9 && A.X0 == C.X0 && A.Y0 == C.Y0, "checkpoint: ... beam list (p_z) and transverse window", dp, 0, 1e-9);
    std::filesystem::remove_all(dir);
}
