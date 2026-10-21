#!/usr/bin/env python3
"""GEM-PIC-3D GUI: input-deck editor, run control and live plots in the web browser.

    python3 tools/gempic3d_gui.py [run.deck] [--port 8767] [--exe PATH] [--no-browser]

Opens http://127.0.0.1:8767/?token=... in the default browser. Only the Python standard library and numpy
are needed; the page has no external dependencies, so the GUI also works offline. On a remote machine,
start it with --no-browser and forward the port:  ssh -L 8767:127.0.0.1:8767 host, then open the URL.

Live plots: the GUI runs the deck with the command-line override output.frames=true (and output.every, if
"frames every" is given); the deck file is not changed. Every output step then writes frame_NNNNN.bin: E and
B (code units m c omega0/e) and, with plasma, n_e/n0 of the streaming electrons, in the planes y = 0 and
x = 0 through the axis. The GUI shows 2D maps of either plane, cuts along zeta and across, and the bunch
evolution from diag.txt. Finished runs can be browsed the same way. The deck is checked with
`gempic3d deck --check` before every run.

The server listens on 127.0.0.1 only and every request needs the random token of the URL.
"""
import argparse
import base64
import json
import os
import re
import secrets
import shutil
import signal
import subprocess
import sys
import threading
import time
import webbrowser
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import parse_qs, urlparse

import numpy as np


HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)

STEP_RE = re.compile(r"^\s*(\d+)\s+([-+0-9.eE]+)\s+\d+\s")       # a line of diag.txt on stdout: step, z [mm]
NSTEPS_RE = re.compile(r"\((\d+) steps to z_end\)")
OUTPUT_RE = re.compile(r"^\s*output: dir '([^']*)', every (\d+) steps")
FRAME_RE = re.compile(r"^frame_(\d+)\.bin$")
SIGNED = {"ex", "ey", "ez", "bx", "by", "bz"}          # diverging colour map; densities sequential
FIELDS = ["ex", "ey", "ez", "bx", "by", "bz", "ne"]
DIAG_COLS = ["step", "z_mm", "N", "x_um", "y_um", "zeta_um", "sig_x_um", "sig_y_um", "sig_z_um", "px", "py", "pz",
             "gamma", "p_khat", "rms_pz", "eta_rel", "a_env", "W_field_J", "W_kin_J", "N_trapped", "Q_trapped_pC"]


# ----------------------------------------------------------------------------- deck check
def check_deck(exe, deck_path, overrides=(), cwd=None):
    """run `gempic3d deck --check`: output dir (relative to cwd), output.every, number of steps and the
    messages (errors)"""
    p = subprocess.run([exe, deck_path, "--check"] + list(overrides), cwd=cwd or os.path.dirname(deck_path),
                       capture_output=True, text=True, timeout=120)
    info = {"ok": p.returncode == 0, "outdir": None, "every": None, "steps": 0,
            "messages": (p.stdout + p.stderr).splitlines()}
    for line in info["messages"]:
        m = OUTPUT_RE.match(line)
        if m:
            info["outdir"], info["every"] = m.group(1), int(m.group(2))
        m = NSTEPS_RE.search(line)
        if m:
            info["steps"] = int(m.group(1))
    return info


# ----------------------------------------------------------------------------- frames
def read_frame(fn):
    """frame_NNNNN.bin: header line (JSON), then per component the planes y = 0 [zeta][x] and x = 0 [zeta][y]"""
    with open(fn, "rb") as f:
        h = json.loads(f.readline())
        N, nx, ny = h["nzeta"], h["nx"], h["ny"]
        x = np.array(h["x_um"]); yn = np.array(h["y_um"]); yh = np.array(h["y_half_um"])
        zeta = np.array(h["zeta_um"])
        ox, oy = np.argsort(x), np.argsort(yn)       # Fourier grids are stored from the box edge: sort by position
        comps = {}
        for c, half in zip(h["comps"], h["half"]):
            a = np.frombuffer(f.read(4 * N * nx), np.float32).reshape(N, nx)[:, ox]
            b = np.frombuffer(f.read(4 * N * ny), np.float32).reshape(N, ny)
            yy = yh if half else yn
            o = np.argsort(yy)
            comps[c] = {"y0": (x[ox], a), "x0": (yy[o], b[:, o])}
    return {"z_mm": h["z_mm"], "zeta": zeta, "comps": comps}


class FrameCache:
    def __init__(self, n=4):
        self.n, self.d, self.lock = n, {}, threading.Lock()

    def get(self, fn):
        mt = os.path.getmtime(fn)
        with self.lock:
            e = self.d.get(fn)
            if e and e[0] == mt:
                return e[1]
        try:
            rec = read_frame(fn)
        except Exception:   # being written (text maps) or unreadable
            return None
        with self.lock:
            self.d[fn] = (mt, rec)
            while len(self.d) > self.n:
                self.d.pop(next(iter(self.d)))
        return rec


CACHE = FrameCache()


def plane(rec, name, pl, xs):
    """field `name` of the plane pl ('y0': x across, 'x0': y across) at the positions xs, [zeta][xs]"""
    nodes, F = rec["comps"][name][pl]
    xs = np.asarray(xs, float)
    j = np.clip(np.searchsorted(nodes, xs) - 1, 0, len(nodes) - 2)
    w = np.clip((xs - nodes[j]) / (nodes[j + 1] - nodes[j]), 0, 1)
    return rec["zeta"], F[:, j] * (1 - w) + F[:, j + 1] * w


# ----------------------------------------------------------------------------- run control
class Runner:
    def __init__(self, exe):
        self.exe = exe
        self.proc = None
        self.lock = threading.Lock()
        self.log = []
        self.step = -1
        self.z = 0.0
        self.rc = None
        self.workdir = os.getcwd()
        self.outdir = None
        self.steps = 0
        self.started = None
        self.ended = None

    def running(self):
        return self.proc is not None and self.proc.poll() is None

    def overrides(self, every, extra):
        ov = [e for e in extra if "=" in e]
        if every > 0:
            ov.append(f"output.every={every}")
        ov.append("output.frames=true")
        return ov

    def start(self, deck_path, text, threads, ranks, every, extra):
        if self.running():
            raise RuntimeError("a run is already in progress")
        deck_path = os.path.abspath(deck_path)
        with open(deck_path, "w") as f:
            f.write(text)
        wd = os.path.dirname(deck_path)
        base = [e for e in extra if "=" in e]
        info = check_deck(self.exe, deck_path, base, wd)
        if not info["ok"]:
            raise RuntimeError("the deck has errors:\n" + "\n".join(m for m in info["messages"] if m.strip())[-3000:])
        ov = self.overrides(every, extra)
        info = check_deck(self.exe, deck_path, ov, wd)
        if not info["ok"]:
            raise RuntimeError("\n".join(info["messages"])[-3000:])
        cmd = [self.exe, deck_path] + ov
        if ranks > 1:
            mpi = shutil.which("mpirun") or shutil.which("mpiexec")
            if mpi is None:
                raise RuntimeError("mpirun not found")
            flags = ["--allow-run-as-root"] if os.geteuid() == 0 else []
            try:   # Open MPI refuses more ranks than cores unless told
                if "Open MPI" in subprocess.run([mpi, "--version"], capture_output=True, text=True, timeout=20).stdout \
                        and ranks > (os.cpu_count() or 1):
                    flags.append("--oversubscribe")
            except (OSError, subprocess.SubprocessError):
                pass
            cmd = [mpi, "-np", str(ranks)] + flags + cmd
        env = dict(os.environ)
        env["OMP_NUM_THREADS"] = str(max(1, threads))
        env.setdefault("OMP_PROC_BIND", "false")
        outdir = os.path.join(wd, info["outdir"] or "out")
        # remove old frames of the same output directory (a new run starts at step 0)
        if os.path.isdir(outdir):
            for fn in os.listdir(outdir):
                if FRAME_RE.match(fn) or fn.endswith(".bin.tmp"):
                    os.remove(os.path.join(outdir, fn))
        with self.lock:
            self.log = ["$ " + " ".join(cmd), ""]
            self.step, self.z, self.rc = -1, 0.0, None
            self.workdir, self.outdir, self.steps = wd, outdir, info["steps"]
            self.started, self.ended = time.time(), None
        self.proc = subprocess.Popen(cmd, cwd=wd, env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                     text=True, bufsize=1, start_new_session=True)
        threading.Thread(target=self._reader, args=(self.proc,), daemon=True).start()

    def _reader(self, proc):
        for line in proc.stdout:
            line = line.rstrip("\n")
            m = STEP_RE.match(line)
            with self.lock:
                self.log.append(line)
                if len(self.log) > 4000:
                    del self.log[:1000]
                if m:
                    self.step, self.z = int(m.group(1)), float(m.group(2))
        proc.wait()
        with self.lock:
            self.rc = proc.returncode
            self.ended = time.time()
            self.log.append(f"[exit code {proc.returncode}]")

    def fresh(self, fn):
        """the file exists and, during a run, was written by this run"""
        if not os.path.exists(fn):
            return False
        return not (self.running() and self.started and os.path.getmtime(fn) < self.started)

    def stop(self):
        if self.running():
            try:
                os.killpg(self.proc.pid, signal.SIGTERM)
            except ProcessLookupError:
                pass

    def frames(self, outdir=None):
        """steps of the frames in the output directory"""
        d = outdir or self.outdir
        if d is None or not os.path.isdir(d):
            return []
        return sorted(int(m.group(1)) for m in (FRAME_RE.match(f) for f in os.listdir(d)) if m)

    def frame_file(self, step, outdir=None):
        return os.path.join(outdir or self.outdir, f"frame_{step:05d}.bin")

    def status(self, nlog=300):
        with self.lock:
            return {"running": self.running(), "rc": self.rc, "step": self.step, "z": self.z, "steps": self.steps,
                    "outdir": self.outdir, "workdir": self.workdir, "log": self.log[-nlog:],
                    "elapsed": ((self.ended or time.time()) - self.started) if self.started else 0.0}


def read_diag(fn):
    """diag.txt -> {"bunch": {column: list}} (one line per output step)"""
    out = {c: [] for c in DIAG_COLS if c != "step"}
    with open(fn) as f:
        for line in f:
            if line.startswith("#") or not line.strip():
                continue
            p = line.split()
            for c, v in zip(DIAG_COLS, p):
                if c != "step":
                    out[c].append(float(v))
    return {"bunch": out}


# ----------------------------------------------------------------------------- HTTP
class Handler(BaseHTTPRequestHandler):
    server_version = "GEM-PIC-3D-GUI"

    def log_message(self, *a):
        pass

    def _send(self, code, body, ctype="application/json"):
        if isinstance(body, (dict, list)):
            body = json.dumps(body).encode()
        elif isinstance(body, str):
            body = body.encode()
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def _auth(self, q):
        tok = self.headers.get("X-Token") or q.get("token", [""])[0]
        return secrets.compare_digest(tok, self.server.token)

    def do_GET(self):
        u = urlparse(self.path)
        q = parse_qs(u.query)
        if u.path == "/":
            if not self._auth(q):
                return self._send(403, "forbidden: open the URL with ?token=... printed by gempic3d_gui.py", "text/plain")
            with open(os.path.join(HERE, "gempic3d_gui.html"), "rb") as f:
                return self._send(200, f.read(), "text/html; charset=utf-8")
        if not self._auth(q):
            return self._send(403, {"error": "forbidden"})
        try:
            return self._send(200, self.api_get(u.path, q))
        except Exception as e:  # report to the page
            return self._send(400, {"error": str(e)})

    def do_POST(self):
        u = urlparse(self.path)
        if not self._auth(parse_qs(u.query)):
            return self._send(403, {"error": "forbidden"})
        n = int(self.headers.get("Content-Length", "0"))
        data = json.loads(self.rfile.read(n) or b"{}")
        try:
            return self._send(200, self.api_post(u.path, data))
        except Exception as e:
            return self._send(400, {"error": str(e)})

    # ------------------------------------------------------------------ GET API
    def api_get(self, path, q):
        R = self.server.runner
        a = lambda k, d=None: q.get(k, [d])[0]   # noqa: E731
        if path == "/api/init":
            p = self.server.deck
            text = open(p).read() if p and os.path.exists(p) else ""
            return {"deck": p or os.path.join(os.getcwd(), "run.deck"), "text": text, "examples": self.examples(),
                    "exe": R.exe, "cpus": os.cpu_count() or 1, "mpirun": bool(shutil.which("mpirun"))}
        if path == "/api/deck":
            p = os.path.abspath(a("path"))
            return {"path": p, "text": open(p).read()}
        if path == "/api/status":
            return R.status()
        od = a("outdir")
        if path == "/api/frames":
            steps = R.frames(od)
            if R.running() and steps:
                steps = [st for st in steps if R.fresh(R.frame_file(st, od))]
            comps, info = [], {}
            if steps:
                rec = CACHE.get(R.frame_file(steps[-1], od))
                if rec is not None:
                    comps = [c for c in FIELDS if c in rec["comps"]]
                    xs = rec["comps"][comps[0]]["y0"][0]
                    ys = rec["comps"][comps[0]]["x0"][0]
                    info = {"zeta": [float(rec["zeta"][0]), float(rec["zeta"][-1])],
                            "xr": [float(xs[0]), float(xs[-1])], "yr": [float(ys[0]), float(ys[-1])]}
            base = od or R.outdir
            beams = bool(base) and R.fresh(os.path.join(base, "diag.txt"))
            return {"dir": base, "frames": steps, "comps": comps, **info, "beams": beams}
        if path in ("/api/map", "/api/lines", "/api/rline"):
            rec = CACHE.get(R.frame_file(int(a("step")), od))
            if rec is None:
                raise RuntimeError("frame not available")
            comp = a("comp", "ez")
            pl = a("plane", "y0")
            if path == "/api/map":
                nodes = rec["comps"][comp][pl][0]
                lim = float(a("rmax", 0) or 0)
                lo, hi = (max(-lim, nodes[0]), min(lim, nodes[-1])) if lim > 0 else (nodes[0], nodes[-1])
                nx = int(a("nr", 240))
                xs = np.linspace(lo, hi, nx)
                zeta, g = plane(rec, comp, pl, xs)
                nz = int(a("nz", 900))
                if len(zeta) > nz:   # thin out in zeta by block maxima of |f| (keeps narrow features)
                    k = int(np.ceil(len(zeta) / nz))
                    n = len(zeta) // k
                    bl = g[: n * k].reshape(n, k, nx)
                    idx = np.abs(bl).argmax(axis=1)
                    g = np.take_along_axis(bl, idx[:, None, :], axis=1)[:, 0, :]
                    zs = zeta[: n * k].reshape(n, k).mean(axis=1)
                else:
                    zs = zeta
                fin = g[np.isfinite(g)]
                vmin, vmax = (float(fin.min()), float(fin.max())) if fin.size else (0.0, 1.0)
                if fin.size and comp in SIGNED:
                    a99 = float(np.percentile(np.abs(fin), 99.5))
                    rlo, rhi = -a99, a99
                elif fin.size:
                    rlo, rhi = float(np.percentile(fin, 0.5)), float(np.percentile(fin, 99.5))
                else:
                    rlo, rhi = vmin, vmax
                return {"z_mm": rec["z_mm"], "comp": comp, "signed": comp in SIGNED, "zeta0": float(zs[0]),
                        "zeta1": float(zs[-1]), "nz": int(len(zs)), "x0": float(xs[0]), "x1": float(xs[-1]), "nx": nx,
                        "vmin": vmin, "vmax": vmax, "rlo": rlo, "rhi": rhi,
                        "data": base64.b64encode(np.ascontiguousarray(g, dtype=np.float32).tobytes()).decode()}
            comps = [c for c in a("comps", "ez").split(",") if c in rec["comps"]]
            if path == "/api/lines":   # along zeta at the transverse position x (or y)
                x0 = float(a("x", 0))
                out, zeta = {}, rec["zeta"]
                for c in comps:
                    zeta, v = plane(rec, c, pl, [x0])
                    out[c] = v[:, 0].tolist()
                return {"z_mm": rec["z_mm"], "x": zeta.tolist(), "series": out, "xpos": x0}
            zc = float(a("zeta", 0))   # transverse cut at zeta
            nodes = rec["comps"][comps[0]][pl][0] if comps else np.array([-1.0, 1.0])
            lim = float(a("rmax", 0) or 0)
            lo, hi = (max(-lim, nodes[0]), min(lim, nodes[-1])) if lim > 0 else (nodes[0], nodes[-1])
            xs = np.linspace(lo, hi, 401)
            out, zk = {}, zc
            for c in comps:
                zeta, v = plane(rec, c, pl, xs)
                k = int(np.argmin(np.abs(zeta - zc)))
                zk = float(zeta[k])
                out[c] = v[k].tolist()
            return {"z_mm": rec["z_mm"], "x": xs.tolist(), "zeta": zk, "series": out}
        if path == "/api/beams":
            base = od or R.outdir
            fn = os.path.join(base or ".", "diag.txt")
            if not R.fresh(fn):
                return {}
            try:
                return read_diag(fn)
            except Exception:
                return {}
        raise RuntimeError("unknown request " + path)

    def examples(self):
        out = []
        for sub in ("examples", "tests"):
            d = os.path.join(ROOT, sub)
            if os.path.isdir(d):
                out += sorted(os.path.join(d, f) for f in os.listdir(d) if f.endswith(".deck"))
        return out

    # ------------------------------------------------------------------ POST API
    def api_post(self, path, d):
        R = self.server.runner
        if path == "/api/save":
            p = os.path.abspath(d["path"])
            with open(p, "w") as f:
                f.write(d["text"])
            return {"path": p}
        if path == "/api/check":
            p = os.path.abspath(d["path"])
            with open(p, "w") as f:
                f.write(d["text"])
            extra = [s.strip() for s in d.get("extra", "").split("\n") if "=" in s]
            info = check_deck(R.exe, p, extra)
            return {"ok": info["ok"], "messages": info["messages"]}
        if path == "/api/run":
            R.start(d["path"], d["text"], int(d.get("threads", 1)), int(d.get("ranks", 1)), int(d.get("every", 0) or 0),
                    [s.strip() for s in d.get("extra", "").split("\n") if s.strip()])
            return {"ok": True}
        if path == "/api/stop":
            R.stop()
            return {"ok": True}
        if path == "/api/open":   # browse the output of a finished run
            p = os.path.abspath(d["path"])
            if d.get("text"):
                with open(p, "w") as f:
                    f.write(d["text"])
            info = check_deck(R.exe, p)
            if info["outdir"] is None:
                raise RuntimeError("cannot find the output directory of this deck:\n" + "\n".join(info["messages"])[-2000:])
            R.outdir, R.workdir = os.path.join(os.path.dirname(p), info["outdir"]), os.path.dirname(p)
            return {"outdir": R.outdir}
        raise RuntimeError("unknown request " + path)


def find_exe(arg):
    for c in [arg, os.path.join(ROOT, "build", "gempic3d"), shutil.which("gempic3d")]:
        if c and os.path.isfile(c) and os.access(c, os.X_OK):
            return os.path.abspath(c)
    return arg or "gempic3d"


def main():
    ap = argparse.ArgumentParser(description="GEM-PIC-3D GUI (deck editor, run control, live plots in the browser)")
    ap.add_argument("deck", nargs="?", help="input deck to open")
    ap.add_argument("--port", type=int, default=8767)
    ap.add_argument("--exe", help="gempic3d executable (default: build/gempic3d of this repository, then PATH)")
    ap.add_argument("--no-browser", action="store_true", help="do not open a browser (e.g. on a remote machine)")
    ap.add_argument("--token", help=argparse.SUPPRESS)
    args = ap.parse_args()
    srv = ThreadingHTTPServer(("127.0.0.1", args.port), Handler)
    srv.daemon_threads = True
    srv.token = args.token or secrets.token_urlsafe(16)
    srv.runner = Runner(find_exe(args.exe))
    srv.deck = os.path.abspath(args.deck) if args.deck else None
    if srv.deck and os.path.exists(srv.deck):
        try:
            info = check_deck(srv.runner.exe, srv.deck)
            if info["outdir"]:
                srv.runner.outdir = os.path.join(os.path.dirname(srv.deck), info["outdir"])
                srv.runner.workdir = os.path.dirname(srv.deck)
        except (OSError, subprocess.SubprocessError):
            pass
    url = f"http://127.0.0.1:{args.port}/?token={srv.token}"
    print(f"GEM-PIC-3D GUI: {url}\n  executable: {srv.runner.exe}\n  Ctrl-C to quit", flush=True)
    if not args.no_browser:
        threading.Timer(0.5, lambda: webbrowser.open(url)).start()
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        srv.runner.stop()


if __name__ == "__main__":
    main()
