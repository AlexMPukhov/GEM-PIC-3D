#!/usr/bin/env python3
"""Checks of bunches from files (bunch.file), bunches.txt and the slice table (a build with make HDF5=1).

  usage: python3 tests/beam_import_check.py [build-h5/gempic3d]      exit code 77 (skipped) without h5py / numpy

Run 1: a Gaussian bunch with emittance, energy spread and an offset, and a second (positron) bunch, in vacuum;
       openPMD particles at the start.
  1. bunches.txt of run 1 (step 0) = the moments computed here from the openPMD particles (charge, energy,
     energy spread, sizes, emittances, length, centroid), per bunch;
  2. slices_000000.txt = the same moments per slice, computed here.
Run 2: the electron bunch of run 1 read with bunch.file, placed at another zeta_center and shifted by x0;
  3. the summary of the import and bunches.txt of run 2 = the moments of the file (shifted);
  4. every particle of run 2 is a particle of the file (openPMD of run 2 against run 1).
Run 3: select (an energy window), charge (rescaled) and macroparticles (resampled, equal weights);
  5. charge as requested; moments of the selection within the sampling error; equal weights.
"""
import os
import shutil
import subprocess
import sys
import tempfile

try:
    import h5py
    import numpy as np
except ImportError:
    print("SKIP: h5py or numpy not found")
    sys.exit(77)

here = os.path.dirname(os.path.abspath(__file__))
exe = os.path.abspath(sys.argv[1] if len(sys.argv) > 1 else os.path.join(here, "..", "build-h5", "gempic3d"))
fails = 0
ME_C, QE, MEV = 9.1093837015e-31 * 2.99792458e8, 1.602176634e-19, 0.51099895


def check(ok, name, detail=""):
    global fails
    print(f"{name:<78} {'PASS' if ok else 'FAIL'}  {detail}")
    if not ok:
        fails += 1


def run(deck, *over):
    r = subprocess.run([exe, deck, *over], capture_output=True, text=True)
    if r.returncode != 0:
        print(r.stdout[-3000:], r.stderr[-3000:])
        raise SystemExit(f"FAIL: {exe} exited with {r.returncode}")
    return r.stdout


def species(path, name):
    """x, y, zeta [um], ux, uy, uz, w of one species of an openPMD file of the code"""
    with h5py.File(path, "r") as f:
        it = f["data"][sorted(f["data"].keys(), key=int)[-1]]
        g = it["particles"][name]
        pos = [g["position"][k][()] * g["position"][k].attrs["unitSI"] * 1e6 for k in ("x", "y", "z")]
        mom = [g["momentum"][k][()] * g["momentum"][k].attrs["unitSI"] / ME_C for k in ("x", "y", "z")]
        return np.array(pos + mom + [g["weighting"][()] * 1.0])


def moments(P):
    """N, Q [pC], E [MeV], dE/E, sig_x, sig_y, eps_nx, eps_ny, <zeta>, sig_z, <x>, <y> (lengths in um)"""
    x, y, z, ux, uy, uz, w = P
    W = w.sum()
    av = lambda a: (w * a).sum() / W
    g = np.sqrt(1 + ux**2 + uy**2 + uz**2)
    var = lambda a, b: av((a - av(a)) * (b - av(b)))
    ex = np.sqrt(max(0.0, var(x, x) * var(ux, ux) - var(x, ux)**2))
    ey = np.sqrt(max(0.0, var(y, y) * var(uy, uy) - var(y, uy)**2))
    return np.array([len(w), W * QE * 1e12, (av(g) - 1) * MEV, np.sqrt(var(g, g)) / av(g), np.sqrt(var(x, x)),
                     np.sqrt(var(y, y)), ex, ey, av(z), np.sqrt(var(z, z)), av(x), av(y)])


def table(path):
    """bunches.txt: {(step, name): values}"""
    out = {}
    for line in open(path):
        if line.startswith("#"):
            continue
        t = line.split()
        out[(int(t[0]), t[2])] = np.array([float(v) for v in t[3:]])
    return out


def close(a, b, rel, absol):
    """largest of |a - b| / (rel |b| + absol): <= 1 passes"""
    return float(np.max(np.abs(a - b) / (rel * np.abs(b) + absol)))


# printed digits of the table: N, Q, E, dE/E, 4 lengths with 5 decimals, <zeta>, sig_z with 4, <x>, <y> with 5
ABS = np.array([0.5, 1e-6, 1e-6, 1e-9, 1e-5, 1e-5, 1e-5, 1e-5, 1e-4, 1e-4, 1e-5, 1e-5])
REL = np.array([0, 1e-9, 1e-9, 1e-3, 1e-7, 1e-7, 1e-6, 1e-6, 1e-9, 1e-7, 1e-7, 1e-7])

tmp = tempfile.mkdtemp(prefix="g3d_import_")
try:
    deck1 = os.path.join(tmp, "run1.deck")
    open(deck1, "w").write("""
grid { lambda0 = 1 um; nx = 32; ny = 32; Lx = 200 um; Ly = 200 um; nzeta = 81; window_length = 80 um
       dz = 50 um; z_end = 50 um; move_x = false; move_y = false }
bunch { name = drive; charge = 20 pC; energy = 250 MeV; energy_spread = 0.01; sigma_x = 5 um; sigma_y = 3 um
        sigma_z = 4 um; emittance_x = 1.5 um; emittance_y = 0.4 um; zeta_center = 50 um; x0 = 2 um; y0 = -1 um
        macroparticles = 20000 }
bunch { name = pos; species = positron; charge = 5 pC; gamma = 100; sigma_x = 4 um; sigma_z = 3 um
        zeta_center = 25 um; macroparticles = 4000 }
output {
  every = 1
  beam_every = 1
  format = openpmd
  fields = none
  particles = false
  slice_length = 4 um
  hdf5_single = false
}
""")
    d1 = os.path.join(tmp, "r1")
    run(deck1, f"output.dir={d1}")
    f1 = os.path.join(d1, "openpmd", "data_000000.h5")
    T1 = table(os.path.join(d1, "bunches.txt"))
    worst = 0.0
    for name in ("drive", "pos"):
        worst = max(worst, close(T1[(0, name)], moments(species(f1, name)), REL, ABS))
    check(worst <= 1.0, "bunches.txt = moments of the openPMD particles (two bunches, step 0)", f"{worst:.2g} of the printed digits")
    check(T1[(0, "trapped")][0] == 0 and T1[(0, "trapped")][1] == 0, "bunches.txt: an empty species has a line of zeros")
    # slices
    P = species(f1, "drive")
    worst, n = 0.0, 0
    for line in open(os.path.join(d1, "slices_000000.txt")):
        if line.startswith("#") or line.split()[0] != "drive":
            continue
        t = line.split()
        zc, vals = float(t[1]), np.array([float(v) for v in t[2:]])
        m = np.floor(P[2] / 4.0) == np.floor(zc / 4.0)
        if m.sum() < 50:
            continue
        worst = max(worst, close(vals, moments(P[:, m]), REL * 10, ABS * 2))
        n += 1
    check(n >= 5 and worst <= 1.0, "slices_000000.txt = moments of the particles of each slice", f"{n} slices, {worst:.2g}")

    # run 2: import as it is, at another place
    deck2 = os.path.join(tmp, "run2.deck")
    open(deck2, "w").write(f"""
grid {{ lambda0 = 1 um; nx = 32; ny = 32; Lx = 200 um; Ly = 200 um; nzeta = 121; window_length = 120 um
       dz = 50 um; z_end = 50 um; move_x = false; move_y = false }}
bunch {{ name = witness; file = {f1}; file_species = drive; zeta_center = 70 um; x0 = 10 um }}
output {{ every = 1; beam_every = 1; format = openpmd; fields = none; particles = false; hdf5_single = false }}
""")
    d2 = os.path.join(tmp, "r2")
    out = run(deck2, f"output.dir={d2}")
    M1 = moments(P)
    ref = M1.copy()
    ref[8] = 70.0
    ref[10] += 10.0
    T2 = table(os.path.join(d2, "bunches.txt"))
    w2 = close(T2[(0, "witness")], ref, REL, ABS)
    check(w2 <= 1.0, "import: bunches.txt of the new run = moments of the file, placed at zeta_center, x0", f"{w2:.2g}")
    check("from " in out and "species drive" in out and f"{len(P[0])} of {len(P[0])} particles" in out,
          "import: the summary names the file, the species and the number of particles")
    P2 = species(os.path.join(d2, "openpmd", "data_000000.h5"), "witness")
    a = P[:, np.argsort(P[5])]
    b = P2[:, np.argsort(P2[5])]
    shift = np.array([10.0, 0, 70.0 - M1[8], 0, 0, 0, 0])[:, None]
    scale = np.array([5, 5, 5, 1, 1, 500, a[6].max()])[:, None]
    d = np.abs(b - a - shift) / scale
    check(a.shape == b.shape and d.max() < 2e-6, "import: every particle of the file, shifted (single-precision files)", f"{d.max():.2g}")

    # run 3: selection, charge, resampling
    gsel = 1 + 250 / MEV
    d3 = os.path.join(tmp, "r3")
    out = run(deck2, f"output.dir={d3}", f"bunch.select=gamma > {gsel:.6f}", "bunch.charge=7 pC", "bunch.macroparticles=50000",
              "bunch.x0=0 um")
    g = np.sqrt(1 + P[3]**2 + P[4]**2 + P[5]**2)
    S = P[:, g > gsel]
    ms = moments(S)
    T3 = table(os.path.join(d3, "bunches.txt"))[(0, "witness")]
    P3 = species(os.path.join(d3, "openpmd", "data_000000.h5"), "witness")
    check(abs(T3[1] - 7.0) < 1e-5 and T3[0] == 50000, "import: charge rescaled, resampled to the requested number", f"Q = {T3[1]:.6f} pC")
    check(np.ptp(P3[6]) < 1e-6 * P3[6].mean(), "import: resampled particles have equal weights")
    # resampling of n particles with (nearly) equal weights to N: every moment within a few n^-1/2 of its spread
    rel = np.abs(T3[[2, 4, 5, 6, 7, 9]] / ms[[2, 4, 5, 6, 7, 9]] - 1)
    check(rel.max() < 0.02 and abs(T3[10] - ms[10]) < 0.1, "import: moments of the selection after resampling",
          f"energy {rel[0]:.1e}, sizes {rel[1:3].max():.1e}, emittances {rel[3:5].max():.1e}, length {rel[5]:.1e}")
    check(f"{S.shape[1]} of {len(P[0])} particles" in out and "select" in out, "import: the summary gives the selected number")
finally:
    shutil.rmtree(tmp, ignore_errors=True)

print("ALL PASSED" if not fails else f"FAILED: {fails}")
sys.exit(1 if fails else 0)
