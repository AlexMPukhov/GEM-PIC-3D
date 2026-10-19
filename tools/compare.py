#!/usr/bin/env python3
"""Compare the mean momenta of a laser run and the reference run without the laser, plane by plane.

usage: tools/compare.py LASER_DIR REF_DIR
"""
import sys
import numpy as np

cols = "step z N x y zeta rx ry rzeta px py pz gamma pk rms_pz eta aenv Wf Wk".split()

def load(d):
    a = np.loadtxt(f"{d}/diag.txt")
    return {c: a[:, i] for i, c in enumerate(cols)}

L, R = load(sys.argv[1]), load(sys.argv[2])
n = min(len(L["z"]), len(R["z"]))
if not np.allclose(L["z"][:n], R["z"][:n]):
    sys.exit("the two runs do not share the output planes (use the same grid.dz and output.every)")
print("# z[mm]   a_env    <pz>_laser      <pz>_ref        d<pz>      d<pz>/<pz>   d<gamma>   d<p.khat>")
for i in range(n):
    d = L["pz"][i] - R["pz"][i]
    print(f"{L['z'][i]:8.2f} {L['aenv'][i]:7.4f} {L['pz'][i]:15.8f} {R['pz'][i]:15.8f} {d:11.3e} {d / R['pz'][i]:11.3e}"
          f" {L['gamma'][i] - R['gamma'][i]:10.3e} {L['pk'][i] - R['pk'][i]:10.3e}")
if L["aenv"][n - 1] > 1e-4 * max(L["aenv"].max(), 1e-30):
    print("warning: the pulse has not passed the bunch at the last plane (a_env not ~ 0); extend grid.z_end")
