#!/bin/sh
# Checkpoints through the binary: tests/mpi_plasma.deck (plasma, ions, ionization, two bunches) runs 12 steps with
# a checkpoint every 4 steps; a copy of its directory is cut back to step 4 and continued with --restart latest
# (diag.txt and ionization.txt must equal those of the uninterrupted run), and a run with a wall-time limit of 0.1 s
# must stop with exit code 3 after a checkpoint.  Usage: sh tests/checkpoint.sh [binary] [mpirun -np P]
set -e
BIN=${1:-build/gempic3d}
RUN=${2:-}
DECK=$(dirname "$0")/mpi_plasma.deck
OUT=$(mktemp -d)
$RUN "$BIN" "$DECK" output.dir="$OUT/a" output.every=1 output.checkpoint_every=4 > "$OUT/a.log"
mkdir -p "$OUT/b"
cp "$OUT"/a/checkpoint_000004_r*.bin "$OUT/a/diag.txt" "$OUT/a/ionization.txt" "$OUT/b/"
$RUN "$BIN" "$DECK" output.dir="$OUT/b" output.every=1 --restart latest > "$OUT/b.log"
fail=0
for t in diag.txt ionization.txt; do
  if cmp -s "$OUT/a/$t" "$OUT/b/$t"; then echo "checkpoint: restart after step 4, $t equal  PASS"
  else echo "checkpoint: restart after step 4, $t differs  FAIL"; diff "$OUT/a/$t" "$OUT/b/$t" | head -5; fail=1; fi
done
set +e
$RUN "$BIN" "$DECK" output.dir="$OUT/c" output.walltime_hours=3e-5 > "$OUT/c.log"
rc=$?
set -e
if [ $rc = 3 ] && ls "$OUT"/c/checkpoint_*.bin > /dev/null 2>&1; then echo "checkpoint: wall-time limit, exit code 3 and a checkpoint  PASS"
else echo "checkpoint: wall-time limit: exit code $rc  FAIL"; fail=1; fi
rm -rf "$OUT"
[ $fail = 0 ] && echo "CHECKPOINT: ALL PASSED" || { echo "CHECKPOINT: FAILED"; exit 1; }
