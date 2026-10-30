# GEM-PIC-3D: notes for Claude

Read this first in every session. It is version-controlled, so it persists between chats and containers.

## Workflow

- Zero-touch: Claude writes the code, runs the tests and commits/pushes; Alexander does not edit code or run
  commands. Work on branch `plasma`; merge into `main` only when Alexander asks.
- Pushing from claude.ai goes through the GitHub connector, which sends whole files that pass through Claude's
  own output. Keep every source file below ~30 KB (split a file before it grows past that), push a few files
  per commit, and verify each push with `git fetch` + `git diff HEAD origin/plasma -- <file>` (must be empty).
  Push in an order in which the branch builds after every commit (new files first, the Makefile switch last).
- The container may be reset between turns: commit locally before anything long.
- Terminology: never call GEM-PIC "quasi-static". The forward field keeps its slow derivative ∂_s R, so the
  laser carrier and the beam's radiation are resolved.
- Style (as GEM-PIC-RZ): C++17, no external libraries (FFTW, HDF5, MPI, CUDA optional), a structured deck
  with units (unknown keys are errors), tests against independent analytic references, README with the
  equations.

## Build and test

```
make && make test                               # CPU, OpenMP; a few minutes on one core
make physics                                    # ion collapse on a 320^2 x 640 grid: run it on the GPU
make MPI=1 BUILD=build-mpi && sh tests/mpi.sh build-mpi/gempic3d
make EMU=1 BUILD=build-emu test emu-selftest    # GPU code paths on the CPU
make HDF5=1 BUILD=build-h5 && sh tests/openpmd.sh build-h5/gempic3d   # openPMD, needs h5py + numpy
sh tests/gui.sh build-mpi/gempic3d              # GUI
```

GPU machine (from atto: `ssh cuda`, no password): RTX 5090, sm_120, CUDA 13.0 in /opt/cuda-13.0, gcc 14.1 in
/opt/gcc-14.1.0, system OpenMPI 4.1.1 in /usr/lib64/openmpi; /storage2 is mounted as /.netmount/storage2.
Non-interactive shells there need `LD_LIBRARY_PATH=/opt/isl-0.24/lib:/opt/mpfr-4.2.1/lib:/opt/gmp-6.2.1/lib:/opt/mpc-1.2.1/lib`
for gcc 14's cc1plus (otherwise every compile fails silently); put the environment in a script and run it with
`ssh cuda script` (example: /storage2/pukhov/g3d/env.sh, build_tests.sh, mpi_gpu.sh). MPI there: `OMPI_CXX=g++`
and `mpirun --mca pml ob1 --mca btl self,vader` (the UCX layer crashes now and then in MPI_Finalize). Large
files on atto go to /storage2/pukhov (builds and runs: /storage2/pukhov/g3d), never into the home directory.
atto's `cp` is `cp -i`: use `\cp -f`. The atto nodes (Slurm partition `atto`, 24 cores) have no GPU.

Container setup: `apt-get update && apt-get install -y libhdf5-dev pkg-config openmpi-bin libopenmpi-dev`,
`pip install h5py numpy --break-system-packages`. Detached runs need `setsid nohup`.

## Code map

| file | content |
|---|---|
| `src/sim.hpp` | Config (deck values, grids, output groups), Particle, Simulation |
| `src/sim_internal.hpp` | includes, physical constants, MPI tags, `is_half` (staggering of the free-hand y) |
| `src/sim_kernels.hpp` | device code: TSC weights, field gather, RK4 push in z, march kernels per mode / k_x line |
| `src/sim_config.cpp` | units, grids, `Config::from_deck` (incl. output groups), the run summary |
| `src/sim_setup.cpp` | constructor, y and ζ setup, `init`, `step`, transverse window |
| `src/sim_particles.cpp` | push, MPI handover, current deposition and its spectra and filters |
| `src/sim_fields.cpp` | march over the slices, co-moving field, sponge on R − R_b, real-space fields, grid lasers |
| `src/sim_plasma.cpp` | streaming plasma (electrons, mobile ions), trapping, re-streaming, slice-by-slice march |
| `src/sim_ions.cpp` | ion blocks of the deck, ADK tables on the device, `ionization.txt` |
| `src/sim_bunches.cpp` | bunch blocks (several bunches, positrons, flat-top), loading, co-moving field per bunch |
| `src/ionization.*`, `src/atomic_data.*` | ADK rates (code units, device), counter-based RNG; NIST energies and masses (from RZ) |
| `src/sim_output.cpp` | `diag.txt`, field energy, dumps, GUI frames / native field files, axis, beams, `write_outputs` |
| `src/openpmd.cpp`, `src/openpmd_writer.hpp` | openPMD 1.1.0 writer (HDF5 C library), its HDF5 helpers |
| `src/deck.*`, `src/grid.*`, `src/comm.*` | deck parser, free-hand step mappings, thin MPI layer |
| `src/fft.hpp`, `src/fftmany.hpp`, `src/portable.hpp`, `src/emu/` | FFTs, CPU/GPU portability, GPU emulator |
| `tests/tests.cpp`, `tests/test_*.cpp`, `tests/test_util.hpp` | the checks of `make test` |
| `tests/physics.cpp` | `make physics`: checks on larger grids (ion collapse), seconds on the GPU |
| `tests/openpmd_check.py`, `tests/mpi.sh`, `tests/gui.sh` | output, MPI and GUI checks |
| `tools/gempic3d_gui.py`, `tools/compare.py` | browser GUI, laser vs reference run |
| `docs/output_spec.md` | output groups, common to QUARZ, GEM-PIC-RZ and GEM-PIC-3D |
| `docs/input_deck.md` | all deck keys (moved out of the README) |
| `docs/GEM-PIC-3D_report.*` | report (describes v1 only) |

## Status

- Plasma stage 1 (single rank), free-hand ζ and y, any FFT size, FFTW, GPU emulator, MPI along ζ (no plasma).
- Real GPU (2026-10-30, RTX 5090 on cuda.tp1): `make test`, `tests/emu_wake.cpp`, the examples and `tests/mpi.sh`
  (CUDA + MPI) agree with the CPU build to rounding; 9-12x faster than 20 CPU threads (README, Performance).
- Ions (2026-10-30): mobile ion species (`ion` blocks, RZ keys), separate n_e / Z n_i output, ADK field ionization
  (Monte Carlo, counter-based RNG, freed electrons stream, trapped ones are `inj_<name>`), `ionization.txt`,
  `plasma.free_electrons`. Checks: immobile limit, neutral pairs, ADK = Landau, H and N against rate equations
  (`make test`); ion collapse against a radial test-ion model (`make physics`, converged only at dx = sigma/10 and
  n_b = 100 n0; with n_b = 20 n0 plasma electrons return to the compressed ions and the model fails).
- Output groups as in QUARZ (`output`, `diag.<name>`), native and openPMD. GEM-PIC-RZ is to follow
  `docs/output_spec.md`; QUARZ is the reference.
- 3D beam wake: the earlier ~9 % deviation from the continuum theory is the TSC shape of the plasma
  response (ω² = ω_p² S²(k)); the check compares with the linear theory of the discrete scheme.

## Next

1. (done) GPU build on a real GPU.
2. (done) Plasma: mobile ions, field ionization, several bunches (electrons and positrons).
3. MPI with plasma (streaming particles handed between ranks): Alexander runs 4 MPI ranks, OpenMP helps little.
4. External fields, undulators, chicanes; profiles as expressions; checkpoints.
5. Radiation energy balance check (coherent emission vs far-field integral, plane wave as undulator).
6. Report update (it describes v1).
