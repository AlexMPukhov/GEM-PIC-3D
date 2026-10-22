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
make MPI=1 BUILD=build-mpi && sh tests/mpi.sh build-mpi/gempic3d
make EMU=1 BUILD=build-emu test emu-selftest    # GPU code paths on the CPU
make HDF5=1 BUILD=build-h5 && sh tests/openpmd.sh build-h5/gempic3d   # openPMD, needs h5py + numpy
sh tests/gui.sh build-mpi/gempic3d              # GUI
```

Container setup: `apt-get update && apt-get install -y libhdf5-dev pkg-config openmpi-bin libopenmpi-dev`,
`pip install h5py numpy --break-system-packages`. Detached runs need `setsid nohup`.

## Code map

| file | content |
|---|---|
| `src/sim.hpp` | Config (deck values, grids, output groups), Particle, Simulation |
| `src/sim_internal.hpp` | includes, physical constants, MPI tags, `is_half` (staggering of the free-hand y) |
| `src/sim_kernels.hpp` | device code: TSC weights, field gather, RK4 push in z, march kernels per mode / k_x line |
| `src/sim_config.cpp` | units, grids, `Config::from_deck` (incl. output groups), the run summary |
| `src/sim_setup.cpp` | constructor, y and ζ setup, bunch loading, `init`, `step`, transverse window |
| `src/sim_particles.cpp` | push, MPI handover, current deposition and its spectra and filters |
| `src/sim_fields.cpp` | march over the slices, co-moving field, sponge on R − R_b, real-space fields |
| `src/sim_plasma.cpp` | streaming plasma, trapping, re-streaming, grid lasers, slice-by-slice march |
| `src/sim_output.cpp` | `diag.txt`, field energy, dumps, GUI frames / native field files, axis, beams, `write_outputs` |
| `src/openpmd.cpp` | openPMD 1.1.0 writer (HDF5 C library) |
| `src/deck.*`, `src/grid.*`, `src/comm.*` | deck parser, free-hand step mappings, thin MPI layer |
| `src/fft.hpp`, `src/fftmany.hpp`, `src/portable.hpp`, `src/emu/` | FFTs, CPU/GPU portability, GPU emulator |
| `tests/tests.cpp`, `tests/test_*.cpp`, `tests/test_util.hpp` | the checks of `make test` |
| `tests/openpmd_check.py`, `tests/mpi.sh`, `tests/gui.sh` | output, MPI and GUI checks |
| `tools/gempic3d_gui.py`, `tools/compare.py` | browser GUI, laser vs reference run |
| `docs/output_spec.md` | output groups, common to QUARZ, GEM-PIC-RZ and GEM-PIC-3D |
| `docs/GEM-PIC-3D_report.*` | report (describes v1 only) |

## Status

- Plasma stage 1 (single rank), free-hand ζ and y, any FFT size, FFTW, GPU emulator, MPI along ζ (no plasma).
- Output groups as in QUARZ (`output`, `diag.<name>`), native and openPMD. GEM-PIC-RZ is to follow
  `docs/output_spec.md`; QUARZ is the reference.
- 3D beam wake: the earlier ~9 % deviation from the continuum theory is the TSC shape of the plasma
  response (ω² = ω_p² S²(k)); the check compares with the linear theory of the discrete scheme.

## Next

1. Run the GPU build on a GPU (only the emulator so far).
2. Plasma: mobile ions, field ionization (ADK, NIST data, ported from GEM-PIC-RZ), several bunches.
3. MPI with plasma (streaming particles handed between ranks): Alexander runs 4 MPI ranks, OpenMP helps little.
4. External fields, undulators, chicanes; profiles as expressions; checkpoints.
5. Radiation energy balance check (coherent emission vs far-field integral, plane wave as undulator).
6. Report update (it describes v1).
