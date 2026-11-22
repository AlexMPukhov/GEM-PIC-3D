#!/bin/sh
# HDF5 1.12 or newer: the HDF5 check of the Makefile, run as `make -n` (nothing is compiled)
# against stand-in HDF5 installations with the version lines of H5public.h / H5pubconf.h:
#   1.10.6        -> warning "too old (need 1.12 or newer)", build without HDF5 (no -DGEMPIC_HDF5, no -lhdf5)
#   1.14.6 serial -> -DGEMPIC_HDF5, -I<dir>/include, -lhdf5, no MPI paths
#   1.12.0 parallel, no MPI=1 -> in addition the include and library paths of $(MPICXX) (here a stand-in)
#   1.12.0 parallel, MPI=1    -> no extra MPI paths (mpicxx compiles)
# usage: sh tests/hdf5_version.sh   (from the repository root; part of make test)
set -u
command -v make >/dev/null && command -v awk >/dev/null || { echo "hdf5_version: skipped (needs make and awk)"; exit 77; }
ROOT=$(cd "$(dirname "$0")/.." && pwd)
T=$(mktemp -d "${TMPDIR:-/tmp}/g3d_h5v.XXXXXX")
trap 'rm -rf "$T"' EXIT
fail=0

fake() {   # fake <dir> <major> <minor> <release> <parallel 0|1>
    mkdir -p "$1/include" "$1/lib"
    printf '#define H5_VERS_MAJOR\t%s\t/* major */\n#define H5_VERS_MINOR\t%s\t/* minor */\n#define H5_VERS_RELEASE\t%s\t/* release */\n' \
        "$2" "$3" "$4" > "$1/include/H5public.h"
    if [ "$5" = 1 ]; then echo '#define H5_HAVE_PARALLEL 1' > "$1/include/H5pubconf.h"
    else echo '/* #undef H5_HAVE_PARALLEL */' > "$1/include/H5pubconf.h"; fi
}
fake "$T/h5old" 1 10 6 1
fake "$T/h5new" 1 14 6 0
fake "$T/h5par" 1 12 0 1
cat > "$T/mpicxx" <<'EOF'
#!/bin/sh
echo "g++ -I/fake/mpi/include -I/fake/mpi/include/openmpi -pthread -L/fake/mpi/lib -Wl,-rpath -Wl,/fake/mpi/lib -lmpi"
EOF
chmod +x "$T/mpicxx"

dry() {   # dry <HDF5_DIR> [make args]: the commands of make for build/gempic3d, warnings included
    d=$1; shift
    env -u MAKEFLAGS -u MFLAGS -u MAKELEVEL -u HDF5_CFLAGS -u HDF5_LIBS CPLUS_INCLUDE_PATH= CPATH= \
        make -n -B -C "$ROOT" HDF5=1 HDF5_DIR="$d" MPICXX="$T/mpicxx" BUILD="$T/b" "$@" "$T/b/gempic3d" 2>&1
}
has()  { if printf '%s\n' "$OUT" | grep -q -e "$2"; then echo "  ok    $1"; else echo "  FAIL  $1 (missing: $2)"; fail=1; fi; }
hasnt(){ if printf '%s\n' "$OUT" | grep -q -e "$2"; then echo "  FAIL  $1 (found: $2)"; fail=1; else echo "  ok    $1"; fi; }

echo "HDF5 1.10.6 (too old):"
OUT=$(dry "$T/h5old")
has   "warning with the version"        "HDF5 1.10.6 in .* is too old (need 1.12 or newer)"
hasnt "no -DGEMPIC_HDF5"                   "-DGEMPIC_HDF5"
hasnt "no -lhdf5"                       "-lhdf5"
has   "still builds build/gempic3d"         "src/main.cpp"

echo "HDF5 1.14.6, serial:"
OUT=$(dry "$T/h5new")
hasnt "no warning"                      "too old"
has   "-DGEMPIC_HDF5 and its include path" "-DGEMPIC_HDF5 -I$T/h5new/include"
has   "-lhdf5"                          "-L$T/h5new/lib .*-lhdf5"
hasnt "no MPI paths"                    "/fake/mpi"

echo "HDF5 1.12.0, parallel, build without MPI=1:"
OUT=$(dry "$T/h5par")
has   "-DGEMPIC_HDF5"                      "-DGEMPIC_HDF5"
has   "MPI include paths (mpi.h)"       "-I/fake/mpi/include -I/fake/mpi/include/openmpi"
has   "MPI library"                     "-L/fake/mpi/lib -lmpi"

echo "HDF5 1.12.0, parallel, MPI=1:"
OUT=$(dry "$T/h5par" MPI=1)
has   "-DGEMPIC_HDF5"                      "-DGEMPIC_HDF5"
hasnt "no extra MPI paths (mpicxx)"     "/fake/mpi"

[ $fail = 0 ] && echo "hdf5_version: all checks passed" || echo "hdf5_version: FAILED"
exit $fail
