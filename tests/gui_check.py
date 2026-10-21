"""GUI test (tools/gempic3d_gui.py), headless through its HTTP API:
  1. a run of tests/mpi.deck (laser, dense bunch, sponge, window shifts; 6 steps, frames every 2 steps)
     started from the GUI finishes with four frames and diag.txt;
  2. the cut along zeta that the GUI returns at x = 0 equals the frame file read directly;
  3. map, transverse-cut and bunch requests answer; a deck with an error is refused with the message;
     requests without the token are refused;
  4. with mpirun: a 2-rank run (frames assembled from both ranks) gives the same frames to rounding;
  5. a plasma run (examples/lwfa_plasma.deck, reduced) gives frames with n_e/n0.
usage: python3 tests/gui_check.py <gempic3d>     (exit 77: skipped without numpy)"""
import json
import os
import shutil
import socket
import subprocess
import sys
import tempfile
import time
import urllib.error
import urllib.parse
import urllib.request

try:
    import numpy as np
except ImportError:
    print("SKIP: needs numpy")
    sys.exit(77)

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "tools"))
import gempic3d_gui  # noqa: E402

exe = os.path.abspath(sys.argv[1])
fails = []


def check(cond, msg):
    print(("ok    " if cond else "FAIL  ") + msg, flush=True)
    if not cond:
        fails.append(msg)


def free_port():
    s = socket.socket()
    s.bind(("127.0.0.1", 0))
    p = s.getsockname()[1]
    s.close()
    return p


work = tempfile.mkdtemp(prefix="gempic3d_gui_")
deck = os.path.join(work, "run.deck")
shutil.copy(os.path.join(ROOT, "tests", "mpi.deck"), deck)
port = free_port()
srv = subprocess.Popen([sys.executable, os.path.join(ROOT, "tools", "gempic3d_gui.py"), deck, "--port", str(port),
                        "--no-browser", "--token", "T", "--exe", exe], stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
B = f"http://127.0.0.1:{port}"


def get(path, **q):
    q["token"] = "T"
    with urllib.request.urlopen(B + path + "?" + urllib.parse.urlencode(q), timeout=60) as r:
        return json.loads(r.read())


def post(path, body):
    req = urllib.request.Request(B + path + "?token=T", data=json.dumps(body).encode(), headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=120) as r:
            return json.loads(r.read())
    except urllib.error.HTTPError as e:
        return json.loads(e.read())


def wait_run(limit=900):
    t0 = time.time()
    while time.time() - t0 < limit:
        s = get("/api/status")
        if not s["running"] and s["rc"] is not None:
            return s
        time.sleep(1)
    return get("/api/status")


def frames_of(outdir):
    out = {}
    for fn in sorted(os.listdir(outdir)):
        if fn.startswith("frame_") and fn.endswith(".bin"):
            out[fn] = gempic3d_gui.read_frame(os.path.join(outdir, fn))
    return out


try:
    for _ in range(100):
        try:
            init = get("/api/init")
            break
        except Exception:
            time.sleep(0.1)
    text = open(deck).read()
    extra = "grid.z_end = 0.3 mm\nlaser.enabled = true"
    # 3. token and deck errors
    try:
        urllib.request.urlopen(B + "/api/status", timeout=10)
        check(False, "a request without the token is refused")
    except urllib.error.HTTPError as e:
        check(e.code == 403, "a request without the token is refused")
    r = post("/api/run", {"path": deck, "text": text.replace("nx = 32", "nx = 32\n  bogus = 1"), "threads": 1, "ranks": 1,
                          "every": 2, "extra": extra})
    check("error" in r and "unknown key" in r["error"], "a deck with an error is refused with the message")
    # 1. a run
    r = post("/api/run", {"path": deck, "text": text, "threads": 1, "ranks": 1, "every": 2, "extra": extra})
    check(r.get("ok"), "run started")
    s = wait_run()
    check(s["rc"] == 0, "run finished (exit code 0)")
    outdir = s["outdir"]
    fr = get("/api/frames")
    check(fr["frames"] == [0, 2, 4, 6], f"four frames (got {fr['frames']})")
    check(fr["beams"], "diag.txt is there")
    # 2. cut along zeta at x = 0 equals the frame file
    step = fr["frames"][-1]
    rec = gempic3d_gui.read_frame(os.path.join(outdir, f"frame_{step:05d}.bin"))
    xs, Fz = rec["comps"]["ez"]["y0"]
    k = int(np.argmin(np.abs(xs)))
    d = get("/api/lines", step=step, comps="ez,ex", x=float(xs[k]))
    check(np.allclose(d["series"]["ez"], Fz[:, k], rtol=0, atol=1e-12 * max(1e-30, np.abs(Fz).max())),
          "cut along zeta at the axis = frame file")
    m = get("/api/map", step=step, comp="ez", plane="x0", nr=50, nz=100)
    check(m["nx"] == 50 and m["nz"] <= 100 and m["vmax"] >= m["vmin"], "map of the plane x = 0")
    rl = get("/api/rline", step=step, comps="ez,by", zeta=60)
    check(len(rl["x"]) == 401 and "by" in rl["series"], "transverse cut")
    b = get("/api/beams")
    check("bunch" in b and len(b["bunch"]["z_mm"]) >= 4, "bunch table from diag.txt")
    one = frames_of(outdir)
    # 4. two ranks
    if shutil.which("mpirun"):
        r = post("/api/run", {"path": deck, "text": text, "threads": 1, "ranks": 2, "every": 2, "extra": extra})
        s = wait_run()
        check(s["rc"] == 0, "2-rank run finished")
        two = frames_of(outdir)
        def close(f):   # to rounding of the largest field of the frame
            sc = max(float(np.abs(one[f]["comps"][c][p][1]).max()) for c in one[f]["comps"] for p in ("y0", "x0"))
            return all(np.allclose(one[f]["comps"][c][p][1], two[f]["comps"][c][p][1], rtol=0, atol=1e-10 * sc)
                       for c in one[f]["comps"] for p in ("y0", "x0"))
        same = set(one) == set(two) and all(close(f) for f in one)
        check(same, "2 ranks: the same frames as 1 rank (to rounding: the frames are assembled from both ranks)")
    # 5. plasma
    pdeck = os.path.join(work, "lwfa.deck")
    shutil.copy(os.path.join(ROOT, "examples", "lwfa_plasma.deck"), pdeck)
    pext = "grid.nx = 16\ngrid.ny = 16\ngrid.nzeta = 301\ngrid.window_length = 30 um\nlaser.zeta_center = 22 um\ngrid.z_end = 12 um\nplasma.z_points = 0, 1 um"
    r = post("/api/run", {"path": pdeck, "text": open(pdeck).read(), "threads": 1, "ranks": 1, "every": 1, "extra": pext})
    s = wait_run()
    check(s["rc"] == 0, "plasma run finished")
    fr = get("/api/frames")
    check("ne" in fr["comps"] and len(fr["frames"]) == 4, "plasma frames with n_e/n0")
    m = get("/api/map", step=fr["frames"][-1], comp="ne", nr=20, nz=50)
    check(0.5 < m["rhi"] < 3 and m["vmin"] >= 0, f"n_e/n0 in a sensible range (up to {m['rhi']:.3g})")
finally:
    srv.terminate()
    srv.wait()
    shutil.rmtree(work, ignore_errors=True)
print("GUI: ALL PASSED" if not fails else f"GUI: {len(fails)} FAILED")
sys.exit(1 if fails else 0)
