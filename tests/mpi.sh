#!/bin/sh
# MPI check: the decomposition along zeta must reproduce the single-rank run. tests/mpi.deck (Fourier y) and
# tests/mpi_freehand_y.deck have the laser, a dense bunch with Fawley noise, the sponge, the moving transverse
# window and a free-hand zeta grid; particles cross the block edges; tests/mpi_plasma.deck (at the end) the
# streaming plasma with ions and ionization. Usage: sh tests/mpi.sh [binary]
# (default build-mpi/gempic3d); extra mpirun flags in MPIRUN_FLAGS.
set -e
BIN=${1:-build-mpi/gempic3d}
FLAGS=${MPIRUN_FLAGS:---oversubscribe}
fail=0
for DECK in $(dirname "$0")/mpi.deck $(dirname "$0")/mpi_freehand_y.deck; do
echo "$(basename "$DECK"):"
OUT=$(mktemp -d)
"$BIN" "$DECK" output.dir="$OUT/np1" > "$OUT/np1.log"
for P in 2 3 4; do
  for LASER in true false; do
    mpirun $FLAGS -np $P "$BIN" "$DECK" output.dir="$OUT/np$P$LASER" laser.enabled=$LASER > "$OUT/np$P$LASER.log"
    [ "$LASER" = true ] || "$BIN" "$DECK" output.dir="$OUT/np1false" laser.enabled=false > /dev/null
    REF="$OUT/np1"; [ "$LASER" = true ] || REF="$OUT/np1false"
    python3 - "$REF/diag.txt" "$OUT/np$P$LASER/diag.txt" $P $LASER << 'PY' || fail=1
import sys, numpy as np
r = np.loadtxt(sys.argv[1]); x = np.loadtxt(sys.argv[2])
ok = r.shape == x.shape
d = np.abs(x - r) / (np.abs(r) + 1e-300) if ok else np.array([np.inf])
if ok: d[:, 0:3] = 0
print("mpi: %s ranks, laser %-5s  max relative difference to 1 rank %.1e  %s" % (sys.argv[3], sys.argv[4], d.max(), "PASS" if d.max() < 1e-9 else "FAIL"))
sys.exit(0 if d.max() < 1e-9 else 1)
PY
  done
done
rm -rf "$OUT"
done
# plasma (tests/mpi_plasma.deck): streaming electrons, mobile ions, field ionization, trapping and two bunches
# cross the block edges; diag.txt and ionization.txt must agree in every printed digit, the axis lineouts to
# float rounding (the order of the current sums differs)
DECK=$(dirname "$0")/mpi_plasma.deck
echo "$(basename "$DECK"):"
OUT=$(mktemp -d)
"$BIN" "$DECK" output.dir="$OUT/np1" > "$OUT/np1.log"
for P in 2 3 4; do
  mpirun $FLAGS -np $P "$BIN" "$DECK" output.dir="$OUT/np$P" > "$OUT/np$P.log"
  python3 - "$OUT/np1" "$OUT/np$P" $P << 'PY' || fail=1
import sys, glob, os, numpy as np
a, b, P = sys.argv[1], sys.argv[2], sys.argv[3]
r = np.loadtxt(a + "/diag.txt"); x = np.loadtxt(b + "/diag.txt")
d = np.abs(x - r) / (np.abs(r) + 1e-300) if r.shape == x.shape else np.array([np.inf])
w = 0.0
for f in glob.glob(a + "/axis_*.txt"):
    A = np.loadtxt(f); B = np.loadtxt(b + "/" + os.path.basename(f))
    w = max(w, (np.abs(A - B).max(axis=0) / (np.abs(A).max(axis=0) + 1e-300))[1:].max())
ia = [l.split()[3:] for l in open(a + "/ionization.txt") if l[0] != "#"]
ib = [l.split()[3:] for l in open(b + "/ionization.txt") if l[0] != "#"]
ok = d.max() < 1e-9 and w < 1e-5 and ia == ib
print("mpi plasma: %s ranks  diag %.1e, axis %.1e, ionization.txt %s  %s" % (P, d.max(), w, "equal" if ia == ib else "differs", "PASS" if ok else "FAIL"))
sys.exit(0 if ok else 1)
PY
done
rm -rf "$OUT"
[ $fail = 0 ] && echo "MPI: ALL PASSED" || { echo "MPI: FAILED"; exit 1; }
