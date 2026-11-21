// Tests: plasma density profiles from files (plasma.density_file, src/DensityFile.*, format docs/density_file.md).
#include <fstream>
#include <functional>
#include <sstream>
#include "test_util.hpp"

static void put(const std::string& name, const std::string& text) {
    std::ofstream f(name);
    f << text;
}
static std::string g17(double x) { return strf("%.17g", x); }

// tests/mpi_plasma.deck with the inline table replaced by `plasma_lines` (keys of the plasma block)
static Config deck_config(const std::string& plasma_lines) {
    std::ifstream in("tests/mpi_plasma.deck");
    std::stringstream out;
    std::string l;
    while (std::getline(in, l)) {
        if (l.find("z_points") != std::string::npos || l.find("z_values") != std::string::npos) continue;
        out << l << "\n";
        if (l.find("density       = 8e18") != std::string::npos) out << plasma_lines;
    }
    put("df_test.deck", out.str());
    return Config::from_deck(Deck::read("df_test.deck"));
}
static std::string error_of(const std::string& plasma_lines) {
    try { deck_config(plasma_lines); } catch (const std::exception& e) { return e.what(); }
    return "";
}
// largest |a - b| / n0 of the density of two configurations over points of the box and of the run
static double density_diff(const Config& a, const Config& b) {
    double d = 0;
    for (int k = 0; k < 4000; ++k) {
        const double x = (std::fmod(k * 0.6180339887, 1.0) - 0.5) * 0.98 * a.Lx, y = (std::fmod(k * 0.7548776662, 1.0) - 0.5) * 0.98 * a.Lx,
                     z = std::fmod(k * 0.5698402910, 1.0) * a.z_end;
        d = std::max(d, std::fabs(a.plasma_density(x, y, z) - b.plasma_density(x, y, z)) / a.n0);
    }
    return d;
}

// 3 steps of a laser wake in a plasma whose density is given by `set` (fields, float)
static std::vector<float> wake_run(const std::function<void(Config&, const Deck&)>& set) {
    Config c = Config::base_units(10e-6);
    const double h = 2 * M_PI / 16;
    const int N = int(120 / h) + 1;
    c.nx = 16; c.Lx = 60; c.y_uniform(16, 60); c.zeta_uniform(N, h * (N - 1));
    c.sponge = c.sponge_y = 0; c.move_x = c.move_y = false;
    c.plasma_on = true; c.n0 = 0.01; c.ppc_x = 1; c.ppc_y = 1;
    c.nmacro = 0; c.dz = 25; c.dz_push = 25; c.zeta_center = 0.75 * c.Lwin;
    Config::GridLaser g; g.a0 = 0.5; g.kL = 1; g.waist = 12; g.tau = 25; g.zeta_c = 90;
    c.glasers.push_back(g);
    Deck d;
    d.set_code_units(c.Lu, c.Tu, c.nc);
    d.read_constants();
    set(c, d);
    Simulation sim(c, comm);
    sim.init();
    for (int k = 0; k < 3; ++k) sim.step();
    return sim.F.host();
}

// 33. Density profiles from files. (a) The density function of the code with the profile from a file against the
//     same profile given in the deck, on tests/mpi_plasma.deck: n(z) columns layout = the inline table (to rounding);
//     absolute density in cm^-3 over um; n(r, z) rotated about the axis and n(x, y, z) (grid layout, a trilinear
//     function) against density_function; n(z) with a transverse function in the header. (b) Three steps of a laser
//     wake with an asymmetric n(x, y, z) from a file against the same density_function (fields, float rounding).
//     (c) Rejected files and combinations, summary lines and warnings.
void test_density_file() {
    const Config ref = Config::from_deck(Deck::read("tests/mpi_plasma.deck"));   // table 0 -> 0.2, 20 um -> 1
    const double n0cm = 8e18;
    // (a) n(z)
    put("df_z.txt", "# ramp of tests/mpi_plasma.deck\n# geometry = z\n# columns = z[um] n\n0 0.2\n20 1\n100 1\n");
    const Config fz = deck_config("  density_file = df_z.txt\n");
    close("density file: n(z), columns layout (factor) = z_points / z_values", 1 + density_diff(ref, fz), 1, 1e-14);
    put("df_zabs.txt", "# geometry = z\n# density_unit = cm^-3\n# z = list 0 20 100 um\n" + g17(0.2 * n0cm) + " " + g17(n0cm) + "\n" + g17(n0cm) + "\n");
    close("density file: n(z), grid layout in cm^-3 = the inline table", 1 + density_diff(ref, deck_config("  density_file = df_zabs.txt\n")), 1, 1e-12);
    put("df_zshift.txt", "# geometry = z\n# columns = z[mm], n\n1.000, 0.2\n1.020, 1\n");
    close("density file: file_z and file_outside = hold",
          1 + density_diff(ref, deck_config("  density_file = df_zshift.txt\n  file_z = -1 mm\n  file_outside = hold\n")), 1, 1e-12);
    // n(r, z): ramp x (1 + 0.5 min(r / 10 um, 1)), linear between the grid points
    {
        std::string t = "# geometry = rz\n# columns = z[um] r[um] n\n";
        for (double z : {0.0, 20.0, 100.0})
            for (double r : {0.0, 10.0, 40.0}) t += g17(z) + " " + g17(r) + " " + g17((z < 20 ? 0.2 + 0.04 * z : 1.0) * (1 + 0.5 * std::min(r / 10, 1.0))) + "\n";
        put("df_rz.txt", t);
        const Config a = deck_config("  z_points = 0, 20 um\n  z_values = 0.2, 1\n  density_function = \"1 + 0.5 * min(r / (10 um), 1)\"\n");
        close("density file: n(r, z) rotated about the axis = table x density_function(r)",
              1 + density_diff(a, deck_config("  density_file = df_rz.txt\n")), 1, 1e-12);
    }
    // n(x, y, z): trilinear, not symmetric
    {
        auto f = [](double x, double y, double z) { return (0.2 + 0.008 * z) * (1 + 0.01 * x) * (1 - 0.02 * y); };
        std::string t = "# density_profile = 1\n# geometry = xyz\n# density_unit = 1\n# x = uniform -25 50 2 um\n# y = list -25 0 25 um\n"
                        "# z = uniform 0 100 2 um\n# order = z y x\n";
        for (double z : {0.0, 100.0})
            for (double y : {-25.0, 0.0, 25.0}) t += g17(f(-25, y, z)) + " " + g17(f(25, y, z)) + "\n";
        put("df_xyz.txt", t);
        const Config a = deck_config("  density_function = \"(0.2 + 0.008 * z / (1 um)) * (1 + 0.01 * x / (1 um)) * (1 - 0.02 * y / (1 um))\"\n");
        const Config b = deck_config("  density_file = df_xyz.txt\n");
        close("density file: n(x, y, z), grid layout = density_function(x, y, z)", 1 + density_diff(a, b), 1, 1e-12);
        check(b.plasma_density(26 / (b.Lu * 1e6), 0, 1) == 0 && b.plasma_density(0, 0, 1) > 0, "density file: zero outside the x range of the file",
              b.plasma_density(26 / (b.Lu * 1e6), 0, 1), 0, 0);
    }
    // n(z) with a transverse function in the header
    put("df_t.txt", "# geometry = z\n# columns = z[um] n\n# transverse = \"(1 + 0.02 * x / (1 um)) * exp(-(r / (15 um))^2)\"\n0 0.2\n20 1\n100 1\n");
    {
        const Config a = deck_config("  z_points = 0, 20 um\n  z_values = 0.2, 1\n  density_function = \"(1 + 0.02 * x / (1 um)) * exp(-(r / (15 um))^2)\"\n");
        close("density file: n(z) x transverse function of the header = table x density_function",
              1 + density_diff(a, deck_config("  density_file = df_t.txt\n")), 1, 1e-12);
    }
    // (b) a run: asymmetric n(x, y, z) from a file against the same function
    {
        const double um = 10.0 / (2 * M_PI);   // code length in um (lambda0 = 10 um)
        auto f = [](double x, double y) { return (1 + 0.01 * x) * (1 - 0.006 * y); };   // x, y in code units
        std::string t = "# geometry = xyz\n# density_unit = 1\n# x = list " + g17(-40 * um) + " " + g17(40 * um) + " um\n# y = list " +
                        g17(-40 * um) + " " + g17(40 * um) + " um\n# z = list 0 " + g17(1000 * um) + " um\n# order = z y x\n";
        for (int k = 0; k < 2; ++k)
            for (double y : {-40.0, 40.0}) t += g17(f(-40, y)) + " " + g17(f(40, y)) + "\n";
        put("df_run.txt", t);
        const std::vector<float> a = wake_run([](Config& c, const Deck& d) {
            c.density_fn = Expr("(1 + 0.01 * x / (1 k0^-1)) * (1 - 0.006 * y / (1 k0^-1))", {"x", "y", "z", "r"}, {}, d.unit_fn());
        });
        const std::vector<float> b = wake_run([](Config& c, const Deck&) {
            densfile::DensityFileUnits u;
            const double Lu = c.Lu;
            u.length = [Lu](double x, const std::string&) { return x * 1e-6 / Lu; };   // um
            u.density = [](double x, const std::string&) { return x; };
            c.dfile = std::make_shared<densfile::DensityFile>(densfile::readDensityFile("df_run.txt", "", -1, u));
        });
        double dmax = 0, m = 0;
        for (size_t i = 0; i < a.size(); ++i) { dmax = std::max(dmax, double(std::fabs(a[i] - b[i]))); m = std::max(m, double(std::fabs(a[i]))); }
        check(a.size() == b.size() && m > 0 && dmax <= 1e-5 * m, "density file: 3 steps of a laser wake, n(x, y, z) file = density_function (fields, float)",
              dmax / m, 0, 1e-5);
    }
    // (c) rejections, summary
    auto rejected = [](const char* name, const std::string& lines, const char* expect) {
        const std::string e = error_of(lines);
        const bool ok = e.find(expect) != std::string::npos;
        if (!ok) std::printf("    expected '%s', got: %s\n", expect, e.empty() ? "(accepted)" : e.c_str());
        check(ok, std::string("density file rejected: ") + name, ok, 1, 0);
    };
    put("df_xz.txt", "# geometry = xz\n# columns = x[um] z[um] n\n0 0 1\n1 0 1\n0 9 1\n1 9 1\n");
    rejected("geometry xz", "  density_file = df_xz.txt\n", "Cartesian 2D profile");
    rejected("z_points with the file", "  density_file = df_z.txt\n  z_points = 0, 20 um\n  z_values = 0.2, 1\n", "given by density_file");
    rejected("profile with n(r, z)", "  density_file = df_rz.txt\n  profile = uniform\n", "holds the transverse profile");
    rejected("file_z without a file", "  z_points = 0, 20 um\n  z_values = 0.2, 1\n  file_z = 1 um\n", "only used with plasma.density_file");
    rejected("file outside the run", "  density_file = df_z.txt\n  file_z = 1 mm\n", "outside the run");
    put("df_bad.txt", "# geometry = z\n# columns = z[um] n\n0 0\n5 -1\n");
    rejected("negative density", "  density_file = df_bad.txt\n", "line 4: negative density");
    put("df_bad.txt", "# columns = z[um] n\n0 0\n5 1\n");
    rejected("no geometry", "  density_file = df_bad.txt\n", "no line '# geometry = ...'");
    put("df_bad.txt", "# geometry = z\n# columns = z[fs] n\n0 0\n5 1\n");
    rejected("wrong unit", "  density_file = df_bad.txt\n", "df_bad.txt");
    auto has = [](const std::vector<std::string>& L, const char* s) {
        for (const auto& l : L) if (l.find(s) != std::string::npos) return true;
        return false;
    };
    const bool sum = has(fz.dfile_info, "n(z) on 3 points, factor on 'density'") && has(fz.dfile_info, "z = 0 ... 0.1 mm (run: 0 ... 0.06 mm)") &&
                     has(fz.dfile_info, "peak 8e+18 cm^-3") && fz.dfile_warn.empty();
    if (!sum) for (const auto& l : fz.dfile_info) std::printf("    | %s\n", l.c_str());
    check(sum, "density file: summary (points, ranges against the run, peak), no warning", sum, 1, 0);
    put("df_short.txt", "# geometry = z\n# columns = z[um] n\n0 0\n40 1\n");
    const Config sh = deck_config("  density_file = df_short.txt\n");
    check(has(sh.dfile_warn, "ends at z = 0.04 mm with 100 % of its peak"), "density file: warning, the file ends inside the run", sh.dfile_warn.size(), 1, 0);
    put("df_steep.txt", "# geometry = z\n# columns = z[um] n\n0 0\n10 0\n12 1\n100 1\n");
    const Config st = deck_config("  density_file = df_steep.txt\n");
    check(has(st.dfile_warn, "less than 4 steps"), "density file: warning, gradient shorter than 4 steps", st.dfile_warn.size(), 1, 0);
    for (const char* f : {"df_z.txt", "df_zabs.txt", "df_zshift.txt", "df_rz.txt", "df_xyz.txt", "df_t.txt", "df_run.txt", "df_xz.txt", "df_bad.txt",
                          "df_short.txt", "df_steep.txt", "df_test.deck"})
        std::remove(f);
}
