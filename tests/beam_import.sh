#!/bin/sh
# Bunches from files, bunches.txt and slices (needs a build with make HDF5=1), see tests/beam_import_check.py.
# Exit code 77 (skipped) without python3 / h5py / numpy.   usage: sh tests/beam_import.sh <gempic3d built with HDF5=1>
command -v python3 > /dev/null 2>&1 || { echo "SKIP: python3 not found"; exit 77; }
exec python3 "$(dirname "$0")/beam_import_check.py" "${1:-build-h5/gempic3d}"
