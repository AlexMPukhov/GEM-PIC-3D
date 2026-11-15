#!/usr/bin/env python3
"""GEM-PIC-3D against GEM-PIC-RZ on an axisymmetric structure with an offset beam (channel_3d.deck, channel_rz.deck).

  usage: python3 compare.py <dir of the GEM-PIC-3D run> <dir of the GEM-PIC-RZ run> [step = 10] [tolerances: loss ez kick]

Compared at the same step (the same plane z = s):
  1. energy loss of the bunch from the start, from bunches.txt of both codes [MeV];
  2. E_z on the axis along the window, in the range of the bunch and of the first wake period behind it
     (zeta > zeta_bunch - 1.2 mm): largest difference over the largest |E_z|;
  3. the mean transverse momentum <u_x> gained by the particles in slices of the bunch (the dipole kick of the
     offset beam, from the particle files), weighted rms difference over the largest slice value.
Exit code 1 if a difference exceeds its tolerance (default 3 %, 6 %, 10 %).
"""
import sys
import numpy as np

d3, drz = sys.argv[1], sys.argv[2]
step = int(sys.argv[3]) if len(sys.argv) > 3 else 10
tol = [float(v) for v in sys.argv[4:7]] if len(sys.argv) > 6 else [0.03, 0.06, 0.10]


def energy(path, name):
    rows = [l.split() for l in open(path) if not l.startswith("#")]
    e = {int(r[0]): float(r[5]) for r in rows if r[2] == name}
    zc = {int(r[0]): float(r[11]) for r in rows if r[2] == name}
    return e, zc


e3, zc3 = energy(f"{d3}/bunches.txt", "bunch")
er, _ = energy(f"{drz}/bunches.txt", "bunch0")
l3, lr = e3[0] - e3[step], er[0] - er[step]
ok = []
ok.append(abs(l3 / lr - 1) <= tol[0])
print(f"energy loss after step {step}:  GEM-PIC-3D {l3:.4f} MeV   GEM-PIC-RZ {lr:.4f} MeV   difference {100 * (l3 / lr - 1):+.2f} %")

a3 = np.loadtxt(f"{d3}/axis_{step:06d}.txt")          # zeta_um Ez ...
ar = np.loadtxt(f"{drz}/axis_{step:05d}.txt")         # zeta_um Ex Ez ...
ez_r = np.interp(a3[:, 0], ar[:, 0], ar[:, 2])
m = a3[:, 0] > zc3[step] - 1200.0
d = np.abs(a3[m, 1] - ez_r[m]).max() / np.abs(ez_r[m]).max()
ok.append(d <= tol[1])
print(f"E_z on the axis (bunch and first period): largest difference {100 * d:.2f} % of the largest |E_z| = {np.abs(ez_r[m]).max():.4e}")
print(f"E_z on the axis (whole window):           largest difference "
      f"{100 * np.abs(a3[:, 1] - ez_r).max() / np.abs(ez_r).max():.2f} %")

p3 = np.loadtxt(f"{d3}/beam_bunch_{step:06d}.txt")    # x y zeta px py pz w
pr = np.loadtxt(f"{drz}/beam_{step:05d}.txt")         # x y zeta ux uy uz q species
pr = pr[pr[:, 7] == 0]
edges = np.linspace(zc3[step] - 400, zc3[step] + 400, 9)


def kicks(z, ux, w):
    k, n = [], []
    for lo, hi in zip(edges[:-1], edges[1:]):
        s = (z >= lo) & (z < hi)
        k.append((w[s] * ux[s]).sum() / w[s].sum())
        n.append(w[s].sum())
    return np.array(k), np.array(n) / np.sum(n)


k3, w3 = kicks(p3[:, 2], p3[:, 3], p3[:, 6])
kr, _ = kicks(pr[:, 2], pr[:, 3], np.abs(pr[:, 6]))
dk = np.sqrt((w3 * (k3 - kr) ** 2).sum()) / np.abs(kr).max()
ok.append(dk <= tol[2])
print("slice kicks <u_x> (8 slices of 100 um, tail to head):")
print("  GEM-PIC-3D " + " ".join(f"{v:9.5f}" for v in k3))
print("  GEM-PIC-RZ " + " ".join(f"{v:9.5f}" for v in kr))
print(f"  charge-weighted rms difference {100 * dk:.2f} % of the largest kick")
print("PASS" if all(ok) else "FAIL")
sys.exit(0 if all(ok) else 1)
