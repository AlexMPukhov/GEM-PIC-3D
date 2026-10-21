#!/bin/sh
# GUI (tools/gempic3d_gui.py) through its HTTP API, see tests/gui_check.py. Exit code 77 (skipped)
# without python3 / numpy.   usage: sh tests/gui.sh <gempic3d>
command -v python3 > /dev/null 2>&1 || { echo "SKIP: python3 not found"; exit 77; }
exec python3 "$(dirname "$0")/gui_check.py" "${1:-build/gempic3d}"
