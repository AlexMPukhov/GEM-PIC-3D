# GEM-PIC-3D

Galilean electromagnetic particle-in-cell code in three dimensions, written from scratch on the
model of GEM-PIC-RZ. It is **not** a quasi-static code: the forward field keeps its slow derivative
and the laser carrier and the radiation of the beam are resolved, while the propagation step is set by
the slow evolution, not by a Courant condition. By default the transverse directions are Fourier modes,
so the field solve of a slice is a closed-form scalar problem per (k_x, k_y). Optionally y is a free-hand
finite-difference grid between conducting walls (fine near the bunch, coarse far out); the solve is then
one tridiagonal system along y per k_x. The ζ grid is uniform or free-hand in either case.

Plasma (stage 1 of the GEM-PIC-RZ physics): streaming plasma electrons with an immobile ion background,
lasers on the grid, trapping into the beam list and the bubble closure `plasma.smooth_length`; see
"Plasma". A browser GUI (`tools/gempic3d_gui.py`) edits and runs decks and shows the fields live.

The first application is a dense electron bunch overtaken at a small angle by a plane-wave laser,
with the radiation of the bunch resolved on the grid and acting back on it
(`examples/oblique_plane_wave.deck`).

## Build, test, run

```
make                      # build/gempic3d and build/tests (C++17, OpenMP, no external libraries)
make test                 # 68 checks against analytic references, a few minutes on one core
make MPI=1 BUILD=build-mpi                 # MPI (mpicxx); test: sh tests/mpi.sh build-mpi/gempic3d
make CUDA=1 NVCC=nvcc CUDA_ARCH=-arch=sm_80 BUILD=build-gpu    # GPU (nvcc, cuFFT); add MPI=1 for both
make EMU=1 BUILD=build-emu test emu-selftest   # the GPU code paths on the CPU (GPU emulator, see Parallelism)
make FFTW=1 BUILD=build-fftw               # optional: FFTW 3 instead of the own FFT (CPU builds)
make HDF5=1 BUILD=build-h5                 # optional: openPMD output (HDF5); test: sh tests/openpmd.sh build-h5/gempic3d
build/gempic3d examples/oblique_plane_wave.deck --check            # summary and memory estimate
build/gempic3d examples/oblique_plane_wave.deck                    # run, results in oblique/
build/gempic3d examples/oblique_plane_wave.deck laser.enabled=false output.dir=ref   # reference
tools/compare.py oblique ref                                       # <pz> with and without the laser
python3 tools/gempic3d_gui.py examples/lwfa_plasma.deck            # GUI in the browser
```

Any key can be overridden on the command line as `block.key=value`. Unknown keys, missing units and
units of the wrong kind are errors with file and line.

## Model

Units: c = 1, lengths in 1/k0, times in 1/ω0 (λ0 = `grid.lambda0`), momenta in m c, fields in
m c ω0/e, densities in n_c. Coordinates ζ = z − t (fast) and s = z (slow, GT2 for the particles):
a level s is the plane z = s of the laboratory and the slice ζ of that level is the time t = s − ζ.

**Fields.** With R = (E_x + B_y, E_y − B_x), L = (E_x − B_y, E_y + B_x), a = i k_x, b = i k_y, Maxwell's
equations reduced as in GEM-PIC [Pukhov, Elkina, Wilson, J. Comput. Phys. 567, 115331 (2026)] read

```
∂_s R_x = a E_z + b B_z − j_x            ∂_ζ L_x = ½ (a E_z − b B_z + j_x)
∂_s R_y = b E_z − a B_z − j_y            ∂_ζ L_y = ½ (a B_z + b E_z + j_y)
∂_ζ B_z = a E_y − b E_x                  ∂_ζ E_z = −(a B_y − b B_x) + j_z
```

This is the Cartesian form of the azimuthal-mode system of GEM-PIC-RZ: ∂_r → ik_x, im/r → ik_y. The
slow derivative is dropped only where it appears together with ∂_ζ (backward and longitudinal
fields), so the backward wave is not described. In vacuum a mode e^{iκζ − iωs} has
ω = 2κk⊥²/(4κ² − k⊥²), the paraxial limit of √(κ² + k⊥²) − κ.

**March.** R is integrated in s with the trapezoidal rule, the ζ-equations with the trapezoidal rule
over each half-cell (length h = h_n between slices n and n+1), from the front slice (L = E_z = B_z = 0)
to the rear. Per (k_x, k_y), R and L of slice n are linear in its E_z and B_z; inserted into the B_z and
E_z equations the cross terms cancel exactly, and

```
B_z = [B_z(n+1) − (h/2) C(n+1) − (h/4) S_B] / [1 + (h/4)(Δ/2 + h/4) k⊥²]
E_z = [E_z(n+1) − (h/2) Q(n+1) − (h/4) S_E − h j_z] / [same]
```

with C = a E_y − b E_x, Q = −a B_y + b B_x, and S_B, S_E the same combinations of the known parts of R
and L (verified symbolically). R and L follow. The current at a slice is the linear interpolation of the
two adjacent half-cells, (h_n j_{n−1/2} + h_{n−1} j_{n+1/2})/(h_{n−1} + h_n), at both levels (the mean on a
uniform grid).

**Free-hand ζ** (`zeta_grid = table`). The step h(ζ) is given at reference points and is linear between
them (as in GEM-PIC-RZ); the slices follow from the exact inverse of the mapping. The march above holds
slice by slice with the local h. The (1,2,1)/4 ζ-filter acts on the half-cell charges h_m j_m, so it
conserves charge on a graded grid. Keep the change of h below ~20 % per cell (`--check` prints the
largest change); the fine part must cover the bunch and the region its radiation slips through.

**Free-hand y** (`y_grid = table`, optional). y becomes a staggered finite-difference grid between
conducting walls at ±`y_max` (rounded up to a whole cell), mirrored about y = 0: E_x, B_y (R_x, L_x), E_z,
j_x and j_z on the nodes; E_y, B_x (R_y, L_y), B_z and j_y on the half-nodes. b = i k_y becomes the
difference operators b₊f = (f_{j+1} − f_j)/Δ_j (nodes → half-nodes) and b₋g = (g_j − g_{j−1})/A_j
(half-nodes → nodes), with Δ_j the cell and A_j the dual cell. a = i k_x is a number and commutes with
b₊ and b₋, so the cross terms cancel exactly as before, and the denominators become operators:

```
[1 + (h/4)(Δ/2 + h/4)(k_x² − b₋b₊)] E_z = ...    (nodes, E_z = 0 at the walls)
[1 + (h/4)(Δ/2 + h/4)(k_x² − b₊b₋)] B_z = ...    (half-nodes)
```

two tridiagonal (Thomas) solves along y per k_x and slice. x stays Fourier, so the window still follows
the bunch in x by an exact phase factor; the y grid is fixed (no `move_y`). This suits configurations
whose drift lies in the x–z plane: the fine cells hold the bunch and its quiver, and stretched cells
push the walls far out at little cost.

**Laser.** The plane wave of infinite transverse size is an external, analytic field,
A = a0 g(η) cos(k_L(η − η_c) + φ) e_pol with η = t − k̂·r, k̂ = (sin θ, 0, cos θ), E = −dA/dη,
B = k̂ × E, and a Gaussian g with the intensity FWHM `tau_fwhm`. It is exact at every particle position
and needs no grid resolution, no transverse periodicity and no dispersion correction. The grid holds
the self-fields of the bunch and its radiation; the scattered field interferes with the analytic laser
in the force on the particles, so the energy exchange with the laser is part of the dynamics.
Polarization `s` has E along y, `p` has E in the x–z plane, `circular` both with a0 per component.

**Particles.** The characteristics are integrated with z as the independent variable by RK4:
dr⊥/dz = p⊥/p_z, dζ/dz = −(γ − p_z)/p_z with γ − p_z = (1 + p⊥²)/(γ + p_z), and
dp/dz = q(γE + p × B)/p_z, with the laser at the exact (x, z, t = z − ζ) and the grid fields of level
s0 (TSC transverse, linear in ζ). A sub-step is at most `dz_push` and at most `push_phase` of the laser
phase seen by the particle, k_L(γ − p·k̂)/p_z · dz, and the sub-steps land exactly on the next plane.

Why not Boris or Vay? On steps set by the slow scale, q Δt E and q Δt v × B of a co-propagating wave
are each ~100 times the net force, and the magnetic rotation angle is far from small. The split
pushers then fail in a plane wave (the first version did: the electron left a 5-FWHM pulse with
γ = 310 instead of 50). Integrating the total force along z has no such splitting error.

**Deposition.** A particle on the plane z = s carries the charge q w/v_z per unit ζ, so j = q w v/v_z:
NGP over the half-cell in ζ, TSC transverse, a (1,2,1)/4 filter along ζ (it removes the source of the
ζ-Nyquist Cherenkov mode, as in GEM-PIC-RZ), and optionally the transverse filter
cos²(k_x Δx/2) cos²(k_y Δy/2). The k⊥ = 0 component of the current is removed: a net current in a
periodic box would drive uniform fields. With the free-hand y, TSC acts in the grid index (the same
weights deposit and gather, so a particle exerts no force on itself), the y-filter is (1,2,1)/4 on the
cell charges, and the k_x = 0 current is removed together with the y-mean of j_y (no TEM current between
the walls). Particles within one cell of a wall are lost.

**Initial field.** The co-moving field of the bunch, −(∇⊥² + γ⁻²∂_ζ²)ψ = ρ, R = (1 + β)E⊥,
E⊥ = −∇⊥ψ: one tridiagonal solve in ζ per (k_x, k_y). With the free-hand y, the y-Laplacian is first
diagonalized once (symmetric tridiagonal eigenproblem in the dual-cell metric), and each (k_x, eigenvalue)
is one tridiagonal solve in ζ. E_z, B_z and L then follow from the march with Δ = 0.

**Boundaries.** The transverse box is periodic (in x, and in y unless y is free-hand; then y ends at
conducting walls). An absorbing sponge at the edges (`sponge` in x, `sponge_y` in y) damps the radiation
part R − R_b of the forward field, where R_b = (1 + β)E⊥ with Tψ = ρ + γ⁻²T⁻¹∂_ζ²ρ is the bound field of
the charge of the slice (T = k⊥², or k_x² − ∂_y² on the free-hand y: the slice charge with the first
1/γ² correction of the co-moving field). Damping all of R instead drains the bunch through its own
Coulomb field at the box edges, like a resistive wall: the first version lost 6 × 10⁻⁴ of ⟨p_z⟩ per
millimetre this way. The transverse window follows the bunch centroid in whole cells (in x, and in y for
Fourier y); the shift is an exact phase factor in Fourier space. Particle coordinates stay continuous, since the analytic laser
is not periodic in the box; only grid indices wrap.

## Plasma

Stage 1 of the physics of GEM-PIC-RZ, with the same algorithm (`src/sim_plasma.cpp`):

- **Plasma block:** `density` (cm^-3 or n_c), a longitudinal profile (`z_points`, `z_values`, linear between
  the points, constant beyond them), uniform or `parabolic` channel (`channel_depth` at `r_channel`), column
  radius `r_max`, `ppc_x` × `ppc_y` electrons per cell on a regular lattice. The ions are an immobile background.
- **Streaming electrons:** fresh plasma is loaded at the front of the window in every step and streams to the
  rear. Per half-cell n, in `numerics.iterations` passes (default 2; 1 is unstable): the electrons are pushed
  from ζ_{n+1} to ζ_n by Boris with dt = h/(1 − v_z) in the fields of slice n+1 (final) and slice n, their
  current W = q w dt/Δ is deposited (TSC), smoothed, transformed, and slice n is solved. The analytic plane
  wave acts on them too. The march goes slice by slice when there is plasma.
- **Prediction of R:** R at slice n needs the current of half-cell n − 1 before the electrons get there. GEM-PIC-RZ
  predicts it with its value of the previous level (`numerics.predictor = previous`). In a laser pulse that
  current rotates with the carrier phase from level to level, and the prediction error of R is then about
  (ω_p² Δ / 2ω)² of the laser field: 6 % in the wake amplitude at n0 = 0.01 n_c and Δ = 50 k0⁻¹, and the scheme
  grows without bound at Δ = 200 k0⁻¹. The default `extrapolate` adds the change from the previous level,
  extrapolated linearly in ζ from the half-cells n and n+1: 0.6 % at Δ = 50, stable at Δ = 200 (with damping).
  R at slice n+1 is corrected once half-cell n is final, as in GEM-PIC-RZ.
- **Trapping:** an electron that reaches z = s1 + (`trap_advance` − 1) Δ inside the march joins the beam list if
  v_z ≥ `vz_beam` (its beam current is deposited at once), and streams again from its ζ in the next step
  otherwise. Beam-list particles slower than `vz_min` leave it and stream in the current march from their
  (z, ζ). `diag.txt` keeps the moments of the bunch only and adds the number and charge of trapped electrons.
- **Bubble closure** (`plasma.smooth_length`, as in GEM-PIC-RZ): the streaming current of every half-cell is
  smoothed, j ← (1 − a²∇⊥²)⁻¹ j: a factor 1/(1 + a²k⊥²) for Fourier y; on the free-hand y one tridiagonal solve
  per k_x for the node components (j_x, j_z, E_z-like) and for the half-node component (j_y, B_z-like). Both
  commute with the discrete divergence, so the smoothed current conserves the smoothed charge.
- **Lasers on the grid** (`laser` blocks with `type = gaussian`; more blocks: `laser`, `laser2`, …): paraxial
  Gaussian pulses, R = 2E⊥ with the complex beam parameter (focus at `z_focus`), `a0`, `waist`, `tau_fwhm`
  (intensity), `zeta_center`, `cep`, linear (`pol_angle`) or circular (`helicity`) polarization. The analytic
  plane wave (`type = plane_wave`, the default) stays as before; it is not changed by the plasma.
- **Net current:** with plasma the k⊥ = 0 current is kept (a 1D plasma wave, not a net current in a periodic box),
  and the sponge uses the bound field of the beam list only.
- **Not yet:** MPI (plasma runs on one rank), mobile ions, field ionization, several bunches, external fields
  and undulators, profiles as expressions and checkpoints (the next stages).

## GUI

`tools/gempic3d_gui.py` is the browser front end of GEM-PIC-RZ adapted to 3D: a small local web server
(Python standard library and numpy) and a page without external dependencies.

```
python3 tools/gempic3d_gui.py examples/lwfa_plasma.deck          # opens the browser
python3 tools/gempic3d_gui.py my.deck --no-browser --port 8767   # remote: ssh -L 8767:127.0.0.1:8767 host
```

- **Deck:** editor with the example decks, Save, Check (`gempic3d deck --check`), extra command-line
  overrides that are not saved in the deck.
- **Run:** threads (OpenMP), MPI ranks, Stop, the log and the progress. A deck with errors is refused with the
  message.
- **Live frames:** the GUI runs with `output.frames = true` on the command line (and `output.every` if "frames
  every" is given); every output step then writes `frame_NNNNN.bin` (written as `.tmp` and renamed): E, B and,
  with plasma, n_e/n0 of the streaming electrons in the planes y = 0 and x = 0 through the axis, assembled from
  all MPI ranks.
- **Plots:** 2D map of either plane with a robust colour range, cuts along ζ and across (click the map), frame
  slider and playback, and the bunch moments from `diag.txt` against z.
- **Security:** the server listens on 127.0.0.1 only, and every request needs the random token of its URL.
- **Test:** `sh tests/gui.sh build-mpi/gempic3d` runs decks through the HTTP API: frames and cuts against the
  frame files, a 2-rank run against 1 rank (to rounding), a plasma run with n_e/n0, token and deck errors.

## Parallelism

**CPU and GPU from one source.** `src/portable.hpp` writes every kernel once as a lambda that runs as an OpenMP
loop or as a CUDA kernel (`--extended-lambda`); fields and particles live in `Buf<T>` device buffers;
`src/fftmany.hpp` batches the FFTs over slices: 2D, or along x only for the free-hand y (own FFT with OpenMP
over rows and columns, or cuFFT). The march is local in k: the FFTs are taken out of it (currents forward
before, fields inverse after), so every (k_x, k_y), or every k_x line of the free-hand y, is one recurrence
over the slices (GPU: one thread per mode or line; CPU: OpenMP over them). The sponge on R − R_b is
therefore applied after the march of a step, not slice by slice. `src/grid.*` holds the free-hand step
mapping, the node lookup and the tridiagonal eigensolver.

**MPI along zeta** (`src/comm.*`). Rank q owns the slices [a, b), rank P−1 at the front, and the particles of
the half-cells [a+1, b]. Per step: push; particles that enter the half-cells of the rank below are handed over
during or at the end of the push with the rest of the step; the three top half-cells of unfiltered current go
to the rank above; the march receives slice b (k-space) from above and sends slice a below; two ghost slices of
real fields go below for the next push. The co-moving initial field is a pipelined tridiagonal solve. Each rank
stores only its own slices (about 256 bytes per grid point). The transverse window shift of step k uses the
centroid of step k − `window_lag` (non-blocking reduction, also in serial runs). The coupling march(q+1) →
push(q) → march(q) limits the speedup to about P/2; the main gain is the memory per rank.

**FFT sizes.** `nx` and `ny` can be any size. The own FFT (`src/fft.hpp`) uses radix 2 for powers of two
(unchanged), a self-sorting mixed-radix Stockham transform for sizes whose prime factors are ≤ 31, and
Bluestein's algorithm otherwise; sizes of the form 2^a 3^b 5^c 7^d are the fast ones. `make FFTW=1` uses FFTW 3
(with its OpenMP threads) instead; cuFFT on the GPU takes any size as well. The default stays free of external
libraries.

**GPU emulator** (`make EMU=1`, `src/emu/`). No CUDA emulator runs current CUDA with cuFFT at a usable speed
(GPGPU-Sim, the closest, stops at CUDA 11, simulates cycles and has no cuFFT). The emulator therefore compiles the
GPU code paths of the sources (GEMPIC_GPU = 1: device buffers, kernel launches through `par_for`, the per-mode
march, cuFFT, device atomics) with the host compiler against a CPU stand-in for the CUDA runtime and cuFFT.
Device memory is kept apart as on a GPU: every `cudaMalloc` is its own memory map, open only during a kernel,
a copy, a memset or an FFT, and closed (PROT_NONE) otherwise, so host code that touches device memory stops with
a backtrace. Kernel threads run in a scrambled order. `make EMU=1 emu-selftest` checks the emulator itself.
Not covered: device code that reads host memory and the compile-time rules of nvcc; for the latter compile
with the real nvcc (`sh tools/install_cuda_pip.sh` installs nvcc and cuFFT from PyPI without root rights).

**Status.** `tests/mpi.sh`: 2, 3 and 4 ranks reproduce the single-rank run exactly (laser, dense bunch, Fawley
noise, sponge, window shifts, block-edge crossings in a graded ζ grid), with Fourier y and with the free-hand y. The CUDA build (also with MPI) compiles with nvcc 13.4 for
sm_80 but has not been run on a GPU yet: run `build-gpu/tests` and `tests/mpi.sh` there first. With several
ranks on GPUs the particle exchange goes through host memory each step. In the GPU emulator all tests and
`tests/mpi.sh` pass, and nvcc 13.4 compiles the CUDA build for sm_80 (also with the free-hand grids); a run on a
real GPU is still to be done.

## Input deck

| block | key | default | meaning |
|---|---|---|---|
| grid | `lambda0` | laser wavelength | reference wavelength (code units) |
| | `nx`, `Lx` | — | x: modes (any size; 2^a 3^b 5^c 7^d are fast) and period |
| | `y_grid` | fourier | `fourier`, or `table` for the free-hand y between walls |
| | `ny`, `Ly` | —, `Lx` | Fourier y: modes (any size) and period |
| | `y_points`, `dy_points`, `y_max` | — | free-hand y: step `dy_points` at the distances `y_points` from the axis (linear between, constant beyond), mirrored; walls at ±`y_max` |
| | `zeta_grid` | uniform | `uniform`, or `table` for the free-hand ζ |
| | `nzeta` | — | uniform ζ: number of slices |
| | `zeta_points`, `dzeta_points` | — | free-hand ζ: step `dzeta_points` at ζ = `zeta_points` from the rear |
| | `window_length` | — | length of the ζ window (front at ζ = window_length; a table rounds it up to a whole cell) |
| | `dz`, `z_end` | — | field step along z, end of the run |
| | `dz_push` | `dz` | largest particle sub-step |
| | `sponge`, `sponge_y`, `sponge_rate` | 0.1 min(Lx, Ly), `sponge`, 0.1 | width of the absorbing layer in x and in y, damping per step at the edge |
| | `move_x`, `move_y` | true, true | the transverse window follows the bunch (`move_y`: Fourier y only) |
| | `window_lag` | 4 | steps between the centroid and the window shift it causes |
| laser | `enabled` | true | false: reference run without the laser |
| | `a0`, `wavelength`, `angle` | —, λ0, 0 | amplitude, wavelength, angle of k to z in the x–z plane |
| | `polarization` | s | s, p, circular |
| | `tau_fwhm`, `cep` | —, 0 | intensity FWHM, carrier phase |
| | `start_fwhm`, `stop_fwhm` | 3.5, = start | the bunch centre starts this many FWHM before the peak; stop this many after it (0: run to z_end) |
| bunch | `charge`, `gamma`, `energy_spread` | —, —, 0 | |
| | `sigma_x`, `sigma_y`, `sigma_z` | —, σ_x, — | rms sizes (σ_z in the lab at t = 0) |
| | `emittance_x`, `emittance_y` | 0, ε_x | normalized rms emittance |
| | `zeta_center` | 0.75 window | position of the centre in the window |
| | `macroparticles`, `seed` | 100000, 1 | |
| | `shot_noise`, `noise_wavelength`, `beamlet` | random, —, 4 | `fawley`: bunching noise of the real electrons at noise_wavelength (Fawley, PRST-AB 5, 070701 (2002)) |
| numerics | `self_fields` | true | false: test particles in the laser only |
| | `zeta_filter`, `k_filter` | true, true | current filters |
| | `push_phase` | 0.05 rad | laser phase per RK4 sub-step |
| | `type` | plane_wave | `plane_wave` (analytic, above) or `gaussian` (on the grid): `a0`, `wavelength`, `waist`, `tau_fwhm`, `zeta_center`, `z_focus` (0), `cep`, `polarization` (linear, circular), `pol_angle`, `helicity`, `enabled` |
| plasma | `density` | — | cm^-3 or n_c |
| | `z_points`, `z_values` | — | longitudinal profile (relative), linear between the points |
| | `profile`, `channel_depth`, `r_channel` | uniform | `parabolic`: n0 (1 + depth r²/r_channel²) |
| | `r_max`, `ppc_x`, `ppc_y` | ∞, 2, ppc_x | column radius, electrons per cell |
| | `smooth_length` | 0 | bubble closure: smoothing length of the plasma currents |
| numerics | `iterations`, `predictor` | 2, extrapolate | passes per slice; prediction of the streaming current (`previous`: as GEM-PIC-RZ) |
| | `vz_min`, `vz_beam`, `trap_advance` | 0.8, 0.9, 1 | trapping and return to streaming, as in GEM-PIC-RZ |
| output | `dir`, `every`, `particles`, `frames` | out, 10, true, false | `every`: period of `diag.txt` and of the main output's fields; `particles`: start/end dumps; `frames`: frame_NNNNN.bin for the GUI |
| | `fields`, `beam_every`, `beams`, `axis` | all, 0, all, false | main output group (see Output) |
| | `format`, `xi_stride`, `particle_stride` | native, 1, 1 | `native`, `openpmd` or `both` |
| | `openpmd_file`, `openpmd_grid`, `openpmd_dy`, `openpmd_dzeta`, `hdf5_single`, `hdf5_compression` | openpmd/data_%06T, uniform, smallest step, true, 0 | openPMD settings |
| diag | `names` | — | diagnostic groups, each a block `diag.<name>` with the keys of the output group (`every` 0, `beam_every` = every, format and openPMD settings from `output`) |

`diag.txt` has one line per output step: z, the centroid (x, y, ζ), the rms sizes, ⟨p_x⟩, ⟨p_y⟩,
⟨p_z⟩, ⟨γ⟩, ⟨p·k̂⟩ (along the laser), the rms of p_z, the position of the bunch in the pulse
(η − η_c in FWHM), the local amplitude a0 g, and the field and kinetic energy in J. `particles_start.txt`
and `particles_end.txt` hold the macroparticles.

## Output

The output follows the common specification of QUARZ, GEM-PIC-RZ and GEM-PIC-3D (`docs/output_spec.md`).
The main output (block `output`) and the diagnostic groups (`diag.names`, blocks `diag.<name>`, written to
`<output.dir>/<name>/`) each write fields every `every` and beam particles every `beam_every` steps, restricted
to their `fields` (`ex ey ez bx by bz ne`, `all`, `none`) and `beams` (`bunch trapped`, `all`, `none`), in their
`format`. Example: all fields every 100 steps in openPMD, E_z and the axis every 5 steps:

```
output {
  every      = 100
  beam_every = 100
  format     = openpmd
}
diag {
  names = light
}
diag.light {
  every  = 5
  fields = ez
  beams  = none
  axis   = true
}
```

- **`fields_NNNNNN.bin`** (native): the selected components in the planes y = 0 and x = 0, in the format of
  the GUI frames (one JSON header line with the grids and the component list, then float32 arrays).
- **`beam_<species>_NNNNNN.txt`** (native): x, y, ζ [µm], p_x, p_y, p_z [m c], weight [real electrons].
- **`axis_NNNNNN.txt`** (`axis = true`): ζ [µm], E_z, E_x − B_y, E_y + B_x [m c ω0/e] on the axis, and n_e/n0
  with plasma.
- **openPMD** (`format = openpmd` or `both`, build with `make HDF5=1`): openPMD 1.1.0, one HDF5 file per output
  step (`openpmd/data_%06T.h5`), basePath `/data/%T/`. Meshes `E`, `B` (x, y, z) and `n_e` (with plasma),
  geometry cartesian, data [ζ][y][x], SI through `unitSI`. A level s is shown at t = s/c with z = s + ζ
  (`gridGlobalOffset`); in the GT2 frame its data lie on the plane z = s at the times t = s − ζ. On uniform grids
  (Fourier y, uniform ζ) the records are the code's own nodes; on free-hand grids `openpmd_grid = uniform`
  interpolates to a uniform grid (spacing `openpmd_dy`, `openpmd_dzeta`, default the smallest step), `native`
  writes `native_E_x`, … on the code's staggered nodes with their positions in `gridCoordinates` (the
  rectilinear meshes of GEM-PIC-RZ and QUARZ), `both` writes both. Particles `bunch` and `trapped`: position
  (z = ζ, positionOffset z = s), momentum per real particle, weighting, charge, mass, one particle patch. The
  input deck after the overrides is stored in `/data/<T>/input_deck`. The fields are those of the end of the
  step, as used by the next push.

## Validation (`make test`, `tests/mpi.sh`)

| check | result |
|---|---|
| FFT: pair transforms against the direct DFT, round trip; sizes 6, 15, 37, 60, 97, 120 (mixed radix, Bluestein); batched 2D and x-only transforms, 16 × 8 and 12 × 10 | 1e-15; 2e-15; 1e-14 |
| free-hand grids: inverse of the step mapping; eigenvectors of the y-Laplacian | 0; 1e-13 |
| vacuum: ω of a forward wave tilted by 0.1 rad against ω = 2κk⊥²/(4κ² − k⊥²), amplitude over 40 steps; default grids, free-hand ζ (4:1), free-hand y, 15 × 6 modes | 0.35 % (32 points per λ), 4e-5, the same for all four |
| diffraction of a Gaussian beam (w0 = 10/k0) over 3 Rayleigh lengths, rms width against the phase of the reduced system: Fourier y; free-hand y at constant step; free-hand y 1 → 4 against constant step | 0.26 %; 0.7 %; 0.09 % |
| one electron, a0 = 5, θ = 0.1, γ = 50, s- and p-polarized: h = γ − p·k̂ conserved; peak γ from h and p⊥ + qA; p after the pulse (Lawson–Woodward) | 1e-8 / 2e-6; 0.04 % / 0.06 %; 6e-8 / 2e-6 |
| field of a long round Gaussian bunch against Gauss's law with the TSC widening σ² + Δ²/4: E_x, E_y, B_y at r = σ, 2σ, mean of 5 slices; default grids, free-hand y, free-hand ζ, 120 × 120 modes | ≤ 1.4 % (macroparticle noise) |
| γ = 1000 bunch in its own field over 100 steps: ⟨γ⟩, field energy; default, and free-hand y and ζ | 3e-8, 0.1 %; 5e-8, 0.1 % |
| dense γ = 50, 160 pC bunch with and without the sponge; Fourier y, free-hand y | 2e-7; 6e-7 |
| constant-step ζ table against the uniform ζ grid, with Fourier y and with the free-hand y | bit for bit |
| MPI, 2/3/4 ranks against 1 rank (`tests/mpi.sh`), Fourier y and free-hand y | identical |
| plasma, 1D linear wake of a Gaussian laser pulse (a0 = 0.05, n0 = 0.01 n_c) against cold-fluid theory: amplitude, plasma wavelength; predictor extrapolate / previous | 0.8 %, 3e-5; 2.5 %, 3e-5 |
| plasma, 3D linear wake of a narrow rigid bunch on the axis after 30 steps against the linear theory of the discrete scheme (plasma frequency k_p S(k) per transverse mode, S = TSC shape); free-hand y against Fourier y | 1.9 %; 1.7 % (the continuum theory differs by 9.7 %: the TSC shape, not an error) |
| plasma, smoothing commutes with the divergence (free-hand y) | 4e-16 |
| streaming in vacuum: a fast electron is trapped at ζ0 − (1 − v_z) Δ/v_z and moves on in the beam list; a slow one returns to streaming | 1e-9 |
| GUI (`tests/gui.sh`): frames, cuts, 2 ranks = 1 rank, plasma frames | pass |
| output groups and openPMD (`tests/openpmd.sh`): periods and selections of two groups, openPMD attributes and input deck, particles = native beam file, fields = native field file and axis file on a uniform grid | pass; 1.5e-7; exact; 1.5e-11 |
| all of the above except the 3D beam wake in the GPU emulator (`make EMU=1 test`); the 3D beam wake there on a reduced grid (`tests/emu_wake.cpp`: 32 × 32 modes, Δζ = 1, 10 steps, Fourier and free-hand y) against the CPU build; `tests/mpi.sh build-emu-mpi/gempic3d`; FFTW build | pass; E_z on the axis identical bit for bit; pass, pass (both before the output groups; to be rerun) |

With the default grids (Fourier y, uniform ζ) the code reproduces the previous version (commit cf3ee50)
up to rounding, except in the sponge, whose bound field now includes the factor (1 + β) and the 1/γ²
correction.

The p-polarized case needs `push_phase` ≈ 0.02: γ drops to about 2 once per cycle there (p⊥0 ≈ a0),
and at 0.05 the momentum after the pulse is off by 5e-5. The pulse must start and stop at ≥ 3.5 FWHM.
At 2.5 FWHM the residual A ~ 10⁻³ a0, amplified by the cross term 2p⊥0·A/h, leaves 4e-4 in p_z.

## Limits and next steps

- **Plasma, transverse resolution:** the TSC shape lowers the plasma frequency of a transverse mode k to
  k_p S(k), S = sinc³(k_x Δx/2) sinc³(k_y Δy/2). A bunch with σ/Δx = 2.7 (the 3D wake check) gets a wake
  about 10 % off the continuum theory; resolve the wake modes accordingly (or compare with the theory of the
  scheme, as the check does).
- **Plasma, performance:** the march goes slice by slice with plasma (FFTs of single planes); the LWFA example
  takes about 25 s per step on one core.

- **Backward radiation is not described** (∂_s L dropped). A momentum gain from the recoil of backward
  or large-angle emission is outside the model by construction.
- **No energy-balance test for the radiation yet.** The next check is coherent emission of a short bunch
  against the far-field radiation integral, as `tests/undulator.sh` in GEM-PIC-RZ, but with the
  analytic plane wave as the undulator. Energy diagnostics of the exchange with the external laser
  (interference term) and of the flux into the sponge are also still missing.
- **Self-fields for slow particles.** The reduced ζ-equations assume sources moving with v ≈ c. In
  p-polarization γ dips to about 2 once per cycle (v_z ≈ 0.87), where the error of the self-field is
  O(1 − v_z) for that part of the orbit.
- **Periodic images.** The bound field of a relativistic bunch reaches γσ_z ≈ 0.5 mm. The images in the
  periodic box need a box-size scan; with the free-hand y the walls can be moved far out cheaply, while x
  stays periodic.
- **Free-hand y accuracy.** Second-order differences: the relative error of k_y² is about (k_y Δy)²/12, so
  structure in y needs ~20 cells per wavelength where it matters. The y grid is fixed, so it suits motion
  in the x–z plane; use Fourier y when the bunch moves in y.
- **Grid fields over a step.** The grid fields are frozen at level s0 during the particle step, as in
  GEM-PIC-RZ; the laser is exact. Halving `dz` checks this.
- **Performance.** No checkpoints yet; the GPU build has run only in the emulator. The
  production deck needs about 13 GB in total (256 bytes per point), split over the MPI ranks.
