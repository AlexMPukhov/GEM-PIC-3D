#!/usr/bin/env python3
"""Checks of the output groups and the openPMD output of GEM-PIC-3D (a build with make HDF5=1).

  usage: python3 tests/openpmd_check.py [build-h5/gempic3d]      exit code 77 (skipped) without h5py / numpy

Run 1, tests/mpi.deck (free-hand zeta, laser, dense bunch, Fawley noise) with the main output in both formats
and a diagnostic group `light` (E_z only, openPMD, axis lineouts, no beams, its own period):
  1. the files of both groups appear at their own periods, and only those;
  2. openPMD 1.1.0 attributes, the input deck (with the overrides) in /data/<T>/input_deck;
  3. the group `light` holds E_z and nothing else;
  4. the particles in openPMD equal the native beam file (number, positions, momenta, weights);
  5. the native records on the code's own nodes exist for the free-hand zeta grid.
Run 2, a uniform grid (Fourier y, uniform zeta, window fixed): the openPMD fields are the code's own nodes,
  6. E and B in the plane y = 0 equal the native field file exactly;
  7. E_z on the axis equals the axis lineout.
"""
import json
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


def check(ok, name, detail=""):
    global fails
    print(f"{name:<72} {'PASS' if ok else 'FAIL'}  {detail}")
    if not ok:
        fails += 1


def run(deck, out, *over):
    r = subprocess.run([exe, deck, f"output.dir={out}", *over], capture_output=True, text=True)
    if r.returncode != 0:
        print(r.stdout[-2000:], r.stderr[-2000:])
        raise SystemExit(f"FAIL: {exe} exited with {r.returncode}")


def read_frame(path):
    with open(path, "rb") as f:
        head = json.loads(f.readline())
        data = np.frombuffer(f.read(), dtype=np.float32)
    N, nx, ny = head["nzeta"], head["nx"], head["ny"]
    planes, o = {}, 0
    for c in head["comps"]:
        planes[c] = (data[o:o + N * nx].reshape(N, nx), data[o + N * nx:o + N * (nx + ny)].reshape(N, ny))
        o += N * (nx + ny)
    return head, planes


tmp = tempfile.mkdtemp(prefix="gempic3d_openpmd_")
try:
    # ---------------------------------------------------------------- run 1
    o1 = os.path.join(tmp, "run1")
    run(os.path.join(here, "mpi.deck"), o1, "grid.z_end=0.6mm", "output.every=6", "output.beam_every=6",
        "output.format=both", "output.openpmd_grid=both", "output.particles=false", "diag.names=light",
        "diag.light.every=4", "diag.light.fields=ez", "diag.light.beams=none", "diag.light.axis=true",
        "diag.light.format=openpmd")
    main_pmd = sorted(os.listdir(os.path.join(o1, "openpmd")))
    light_pmd = sorted(os.listdir(os.path.join(o1, "light", "openpmd")))
    light_axis = sorted(f for f in os.listdir(os.path.join(o1, "light")) if f.startswith("axis_"))
    native = sorted(f for f in os.listdir(o1) if f.startswith("fields_"))
    check(main_pmd == ["data_000000.h5", "data_000006.h5", "data_000012.h5"], "groups: main openPMD files every 6 steps", main_pmd)
    check(light_pmd == ["data_000000.h5", "data_000004.h5", "data_000008.h5", "data_000012.h5"] and len(light_axis) == 4,
          "groups: light openPMD and axis files every 4 steps", light_pmd)
    check(native == ["fields_000000.bin", "fields_000006.bin", "fields_000012.bin"]
          and not any(f.startswith("fields_") or f.startswith("beam_") for f in os.listdir(os.path.join(o1, "light"))),
          "groups: native files of the main output only (light: format openpmd)", native)
    with h5py.File(os.path.join(o1, "openpmd", "data_000006.h5"), "r") as f:
        a = f.attrs
        dec = lambda v: v.decode() if isinstance(v, bytes) else str(v)
        ok = dec(a["openPMD"]) == "1.1.0" and dec(a["basePath"]) == "/data/%T/" and dec(a["iterationEncoding"]) == "fileBased"
        check(ok, "openPMD: version 1.1.0, basePath /data/%T/, fileBased")
        it = f["/data/6"]
        deck = it["input_deck"][()]
        deck = deck.decode() if isinstance(deck, bytes) else str(deck)
        check("diag.light {" in deck and "z_end = 0.6mm" in deck, "openPMD: input deck with the overrides stored")
        m = it["meshes"]
        shapes = {k: m["E"][k].shape for k in ("x", "y", "z")}
        check(len(set(shapes.values())) == 1 and "B" in m and "native_E_z" in m, "openPMD: E, B (uniform) and native records",
              str(shapes["z"]))
        nat = m["native_E_z"]
        gc = [dec(v) for v in nat.attrs["gridCoordinates"]]
        zc = f[gc[0]][()]
        check(nat.shape[0] == len(zc) and np.all(np.diff(zc) > 0) and np.ptp(np.diff(zc)) > 0.1 * np.diff(zc).min(),
              "openPMD: native E_z on the free-hand zeta slices (gridCoordinates)", f"{len(zc)} slices")
        pb = it["particles"]["bunch"]
        P = np.stack([pb["position"][k][()] * pb["position"][k].attrs["unitSI"] for k in ("x", "y", "z")] +
                     [pb["momentum"][k][()] * pb["momentum"][k].attrs["unitSI"] / (9.1093837015e-31 * 2.99792458e8)
                      for k in ("x", "y", "z")] + [pb["weighting"][()]], axis=1)
    with h5py.File(os.path.join(o1, "light", "openpmd", "data_000004.h5"), "r") as f:
        it = f["/data/4"]
        check(set(it["meshes"].keys()) <= {"E", "native_E_z"} and list(it["meshes"]["E"].keys()) == ["z"] and "particles" not in it,
              "groups: light holds E_z only, no particles")
    B = np.loadtxt(os.path.join(o1, "beam_bunch_000006.txt"))
    Bn = np.stack([B[:, 0] * 1e-6, B[:, 1] * 1e-6, B[:, 2] * 1e-6, B[:, 3], B[:, 4], B[:, 5], B[:, 6]], axis=1)
    if len(B) == len(P):
        # pair the particles: sorted by zeta, each with the nearest of its neighbours in the other list (zeta
        # agrees to the print precision of the native file only, so neighbours may swap)
        A, Z = P[np.argsort(P[:, 2])], Bn[np.argsort(Bn[:, 2])]
        scale = np.max(np.abs(Z), axis=0)
        d = 0.0
        for i in range(len(A)):
            j0, j1 = max(0, i - 4), min(len(Z), i + 5)
            e = np.max(np.abs(Z[j0:j1] - A[i]) / scale, axis=1)
            d = max(d, float(e.min()))
    else:
        d = np.inf
    check(d < 1e-6, "particles: openPMD bunch = native beam file", f"{len(P)} particles, max rel diff {d:.2g}")

    # ---------------------------------------------------------------- run 2: uniform grid
    deck2 = os.path.join(tmp, "uniform.deck")
    with open(deck2, "w") as f:
        f.write("""grid {
  lambda0 = 10 um
  nx = 32
  Lx = 240 um
  ny = 32
  Ly = 240 um
  nzeta = 97
  window_length = 96 um
  sponge = 30 um
  dz = 50 um
  z_end = 0.3 mm
  move_x = false
  move_y = false
}
laser {
  a0 = 5
  wavelength = 10 um
  angle = 0.1 rad
  tau_fwhm = 0.02 ps
  start_fwhm = 1.0
  stop_fwhm = 0
}
bunch {
  charge = 160 pC
  gamma = 50
  sigma_x = 10 um
  sigma_z = 10 um
  zeta_center = 60 um
  macroparticles = 4000
}
output {
  every = 3
  format = both
  axis = true
  particles = false
}
""")
    o2 = os.path.join(tmp, "run2")
    run(deck2, o2)
    head, planes = read_frame(os.path.join(o2, "fields_000006.bin"))
    ny, nx = head["ny"], head["nx"]
    with h5py.File(os.path.join(o2, "openpmd", "data_000006.h5"), "r") as f:
        m = f["/data/6/meshes"]
        diff = 0.0
        for vec in ("E", "B"):
            for k in ("x", "y", "z"):
                ds = m[vec][k]
                code = ds[()] * 1.0                      # [zeta][y][x], code units (unitSI aside)
                diff = max(diff, float(np.max(np.abs(code[:, ny // 2, :] - planes[vec.lower() + k][0]))))
        ez_axis = m["E"]["z"][()][:, ny // 2, nx // 2]
    check(diff == 0.0, "openPMD: E, B in the plane y = 0 = native field file (uniform grid)", f"max diff {diff:.2g}")
    ax = np.loadtxt(os.path.join(o2, "axis_000006.txt"))
    d = np.max(np.abs(ax[:, 1] - ez_axis)) / np.max(np.abs(ez_axis))
    check(d < 1e-6, "axis: E_z of axis_000006.txt = openPMD E_z on the axis", f"rel diff {d:.2g}")
finally:
    shutil.rmtree(tmp, ignore_errors=True)

print("ALL PASSED" if fails == 0 else f"FAILED: {fails} check(s)")
sys.exit(1 if fails else 0)
