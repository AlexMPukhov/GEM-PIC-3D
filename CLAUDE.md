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
sh tests/beam_import.sh build-h5/gempic3d       # bunches from files, bunches.txt, slices (HDF5 build, h5py)
make physics                                    # ion collapse, undulator radiation: run it on the GPU (13 s)
make MPI=1 BUILD=build-mpi && sh tests/mpi.sh build-mpi/gempic3d
make EMU=1 BUILD=build-emu test emu-selftest    # GPU code paths on the CPU
make HDF5=1 BUILD=build-h5 && sh tests/openpmd.sh build-h5/gempic3d   # openPMD, needs h5py + numpy
sh tests/gui.sh build-mpi/gempic3d              # GUI
sh tests/checkpoint.sh build/gempic3d           # restart through the binary (add "mpirun -np 2" for MPI)
```

GPU machine (from atto: `ssh cuda`, no password): RTX 5090, sm_120, CUDA 13.0 in /opt/cuda-13.0, gcc 14.1 in
/opt/gcc-14.1.0, system OpenMPI 4.1.1 in /usr/lib64/openmpi; /storage2 is mounted as /.netmount/storage2.
Non-interactive shells there need `LD_LIBRARY_PATH=/opt/isl-0.24/lib:/opt/mpfr-4.2.1/lib:/opt/gmp-6.2.1/lib:/opt/mpc-1.2.1/lib`
for gcc 14's cc1plus (otherwise every compile fails silently); put the environment in a script and run it with
`ssh cuda script` (example: /storage2/pukhov/g3d/env.sh, build_tests.sh, mpi_gpu.sh). MPI there: `OMPI_CXX=g++`
and `mpirun --mca pml ob1 --mca btl self,vader` (the UCX layer crashes now and then in MPI_Finalize). Large
files on atto go to /storage2/pukhov (builds and runs: /storage2/pukhov/g3d), never into the home directory.
atto's `cp` is `cp -i`: use `\cp -f`. The atto nodes (Slurm partition `atto`, 24 cores) have no GPU.

Container setup: `apt-get update && apt-get install -y libhdf5-dev pkg-config openmpi-bin libopenmpi-dev` (HDF5 1.12 or
newer is required: Ubuntu 24.04 has 1.10, the Makefile then builds without HDF5; Ubuntu 25.04 / Debian 13 have 1.14),
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
| `src/sim_plasma.cpp` | streaming plasma: loading, spectra, the march slice by slice and down the MPI ranks, trapping, re-streaming |
| `src/sim_stream.hpp` | device code of the streaming particles: Boris push, deposit, ADK ionization, one half-cell |
| `src/sim_ions.cpp` | ion blocks of the deck, ADK tables on the device, `ionization.txt` |
| `src/external.hpp`, `src/sim_external.cpp` | undulators and transport elements: fields, deck blocks, push sub-steps |
| `src/expr.*` | expressions of the deck (from RZ); `Deck::expr`, `constants`, values as expressions in `Deck::num` |
| `src/laser_file.*`, `src/h5read.hpp` | LASY envelope files (from RZ): reader, writer (tests), exact propagation to z = 0; HDF5 read helper |
| `src/sim_checkpoint.cpp` | checkpoints per rank, restart, `latest` |
| `src/sim_bunches.cpp` | bunch blocks (several bunches, positrons, flat-top), loading, co-moving field per bunch |
| `src/ionization.*`, `src/atomic_data.*` | ADK rates (code units, device), counter-based RNG; NIST energies and masses (from RZ) |
| `src/sim_output.cpp` | `diag.txt`, field energy, dumps, GUI frames / native field files, axis, beams, `write_outputs` |
| `src/openpmd.cpp`, `src/openpmd_writer.hpp` | openPMD 1.1.0 writer (HDF5 C library), its HDF5 helpers |
| `src/deck.*`, `src/grid.*`, `src/comm.*` | deck parser, free-hand step mappings, thin MPI layer |
| `src/fft.hpp`, `src/fftmany.hpp`, `src/portable.hpp`, `src/emu/` | FFTs, CPU/GPU portability, GPU emulator |
| `tests/tests.cpp`, `tests/test_*.cpp`, `tests/test_util.hpp` | the checks of `make test` |
| `tests/physics.cpp`, `tests/test_radiation.cpp` | `make physics`: ion collapse, undulator radiation energy balance; seconds on the GPU |
| `tests/openpmd_check.py`, `tests/mpi.sh`, `tests/gui.sh` | output, MPI and GUI checks |
| `tools/gempic3d_gui.py`, `tools/compare.py` | browser GUI, laser vs reference run |
| `docs/output_spec.md` | output groups, common to QUARZ, GEM-PIC-RZ and GEM-PIC-3D |
| `docs/input_deck.md` | all deck keys (moved out of the README) |
| `docs/GEM-PIC-3D_report.*` | report (describes v1 only) |

## Status

- Plasma, free-hand ζ and y, any FFT size, FFTW, GPU emulator, MPI along ζ (also with plasma: the march runs down
  the ranks with packages down/up, README Parallelism; `tests/mpi.sh` covers it with `tests/mpi_plasma.deck`).
- Real GPU (2026-10-30, RTX 5090 on cuda.tp1): `make test`, `tests/emu_wake.cpp`, the examples and `tests/mpi.sh`
  (CUDA + MPI) agree with the CPU build to rounding; 9-12x faster than 20 CPU threads (README, Performance).
- Ions (2026-10-30): mobile ion species (`ion` blocks, RZ keys), separate n_e / Z n_i output, ADK field ionization
  (Monte Carlo, counter-based RNG, freed electrons stream, trapped ones are `inj_<name>`), `ionization.txt`,
  `plasma.free_electrons`. Checks: immobile limit, neutral pairs, ADK = Landau, H and N against rate equations
  (`make test`); ion collapse against a radial test-ion model (`make physics`, converged only at dx = sigma/10 and
  n_b = 100 n0; with n_b = 20 n0 plasma electrons return to the compressed ions and the model fails).
- LASY laser files (`laser.type = file`, HDF5 build): README section Laser files; `inject = box` only.
- Output groups as in QUARZ (`output`, `diag.<name>`), native and openPMD. GEM-PIC-RZ is to follow
  `docs/output_spec.md`; QUARZ is the reference.
- 3D beam wake: the earlier ~9 % deviation from the continuum theory is the TSC shape of the plasma
  response (ω² = ω_p² S²(k)); the check compares with the linear theory of the discrete scheme.

## Next

1. (done) GPU build on a real GPU.
2. (done) Plasma: mobile ions, field ionization, several bunches (electrons and positrons).
3. (done) MPI with plasma. Possible later: pipelining over steps on the GPU, CUDA-aware MPI.
4. (done) Undulators, quadrupole, plasma lens, solenoid; expressions (constants, values, density_function,
   K_function, laser profile/envelope, bunch profile_z/profile_r/gamma/ux/uy functions); checkpoints
   (`tests/checkpoint.sh`); thin chicane (one rank). Note: a frozen bunch's field changes by ~1-2 % in the first step from the
   co-moving start (E_z ~3 % of E_r near the bunch centre), partly macroparticle noise, partly systematic: to be
   understood (the chicane check therefore compares field changes).
5. (done) Radiation energy balance: coherent undulator emission against the radiation integral (`make physics`,
   tests/test_radiation.cpp; the scan in /storage2/pukhov/g3d/rad). Possible: the same with the plane wave.
6. (done, 2026-10-30) Report: section "Version 2" added to docs/GEM-PIC-3D_report.tex; the PDF must be rebuilt
   where LaTeX works (atto's TeX Live 2018 lacks microtype and the ecrm fonts).
7. (done, 2026-11-10, branch `lasy`) LASY laser files: `laser { type = file }` (src/laser_file.*, reader and
   conventions from GEM-PIC-RZ, h5read.hpp copied from there), cartesian and thetaMode, exact vacuum propagation
   from `file_z` to z = 0 by a 3D FFT (t, y, x), injected as R = 2 E_perp like the Gaussian grid laser; check 28 of
   `make test` in the HDF5 build (README Validation). Every rank propagates the whole file at the start (memory
   nt x ny x nx x padding x 16 B); restrict to the rank's time range if files get large.
   Still missing against GEM-PIC-RZ (its docs/input_deck.md): `inject = boundary` (through the sponge) and LASY
   output; the flying focus; bunches from files (openPMD import); FEL diagnostics (fel.txt, fel_power,
   fel_harmonics); bunch z_start, quiet start, axis_refine; ion `profile = own` (dopants); plasma radial profile
   options; numerics.beam_field = mean (undulators with intense beams); a free-hand z grid; the chicane with MPI.
   Also: understand the 1-2 % first-step change of a rigid bunch's field from the co-moving start.

## Development towards the state of GEM-PIC-RZ (from 2026-11-15)

Alexander (2026-11-15): develop GEM-PIC-3D to the state of GEM-PIC-RZ; follow the choices made there (model,
numerics, deck syntax, output conventions), decide the rest and record each decision here with its reason; report at
milestones with numbers. Features on a branch with a pull request into `main`. First application: beam-driven wakes
in plasma channels of any cross-section (`density_function`), metres of propagation.

Rules taken over from the GEM-PIC-RZ work: no physics may be dropped to make a problem go away; every fix is tested
against an independent reference; compare with the theory of the DISCRETE scheme where the discretization matters;
never analyse the first steps of a plasma wake (start-up transient); keep MPI = 1 rank; compile-check the CUDA build
and keep `make EMU=1` green.

Branch `rz-benchmarks` (2026-11-15):
- **Start field** (`numerics.start_relax`, default on; `Simulation::relax_start_field`): the v = c part of the
  co-moving start field is replaced by the steady state of the discrete scheme (three backward-Euler marches with a
  step of 100 dz). This was the open "1-2 % change of a rigid bunch's field in the first step": the continuum
  co-moving field is not the scheme's steady state, and the trapezoidal rule keeps the mismatch of the stiff modes as
  an odd-even mode in s (1.5 % of E_perp; +-10 ... 35 % of the wake's E_z at the bunch in a plasma channel); now
  <= 1.4e-5 per step without alternation (relaxation step 100 dz: 1e4 dz gave 6e-8 but amplified rounding, 2 MPI
  ranks then differed from 1 in the last printed digit on the free-hand y). The
  discrete steady state itself is clean (E_z, E_x - B_y < 1e-8 E_perp also for sigma_z = 2 h): no static j_z
  mismatch as in GEM-PIC-RZ. The chicane uses the same relaxed field. Decision: fix the start, change no equation.
- **Start field without the k_perp = 0 current of the bunch**, also with plasma (it was kept there and started the
  bunch in the uniform E_z of its sheet charge in the periodic box: 0.023 MeV lost in the first step of the benchmark).
- **Periodic images:** decision: the box stays periodic, the error is stated by a box-size scan (benchmarks/rz:
  0.7 % in the energy loss between 3.2 and 6.4 mm with a wall of 4 c/omega_p). `numerics.box_mean = remove` exists
  for checks; it is worse for channels (their E_z has a physical box average) and stays off.
- **Bunches from files** (`bunch.file`, src/beam_import.* from RZ, src/sim_bunch_file.cpp), **bunches.txt** and
  **slices_NNNNNN.txt** (src/sim_bunch_diag.cpp; centred second moments in two passes, as RZ's energy spread).
  Decision: bunches.txt follows the period of diag.txt (`output.every`), not every step as in RZ: the beam list
  is downloaded from the GPU for it.
- **Deck:** several assignments on a line separated by `;` (the README examples used it, the parser did not).
- **Summary warning** when dz * dzeta * omega_p^2 > 8 (the plasma march fails at 25; the long-wake check runs 6.25).
- **Checks:** 29 static bunch, 30 beam in a drift (E_r / gamma^2), 31 long wake with large steps (tests/test_lessons.cpp;
  `make build/lessons` builds them alone), tests/beam_import.sh, benchmarks/rz (GEM-PIC-RZ, offset bunch in a round
  hollow channel: energy loss +1.7 %, E_z 5 %, dipole kick 0.6 %).
- atto: clone and builds in /storage2/pukhov/g3d-mature (login node: compile; nodes: `sbatch -p atto`, binaries
  linked with -static-libstdc++; HDF5 in /storage2/pukhov/gpu/opt/hdf5; python 3.14 with PYTHONPATH
  /storage2/pukhov/g3d/pylib). GEM-PIC-RZ binary for the atto nodes: /storage2/pukhov/gempic-combo/build_ivb.

Next (order agreed with the lessons of GEM-PIC-RZ):
1. non-axisymmetric channel example (plasma on one side of the beam) over ~1 m with an imported bunch and many
   quadrupoles; cost and resolution (uniform x grid: dx ~ 0.6 sigma_r gave 1-2 %);
2. `beam_field = mean` for undulators (planar step instability), undulator kick/current timing proven or fixed,
   edge-averaged transport elements + quad_edges check, GENESIS benchmark;
3. output: volume and peak-memory report, openPMD position-snapshot attributes, SI density check;
4. real-GPU rerun of all checks; then the remaining list above (boundary LASY injection, flying focus, FEL
   diagnostics, z_start, quiet start, dopant profiles, free-hand z, MPI chicane).
