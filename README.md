# GEM-PIC-3D

Galilean electromagnetic particle-in-cell code in three dimensions, written from scratch on the
model of GEM-PIC-RZ. It is **not** a quasi-static code: the forward field keeps its slow derivative
and the laser carrier and the radiation of the beam are resolved, while the propagation step is set by
the slow evolution, not by a Courant condition. By default the transverse directions are Fourier modes,
so the field solve of a slice is a closed-form scalar problem per (k_x, k_y). Optionally y is a free-hand
finite-difference grid between conducting walls (fine near the bunch, coarse far out); the solve is then
one tridiagonal system along y per k_x. The ζ grid is uniform or free-hand in either case.

Plasma (the GEM-PIC-RZ physics): streaming plasma electrons, mobile ions and ADK field ionization with the
NIST ionization energies, lasers on the grid, trapping into the beam list and the bubble closure
`plasma.smooth_length`; see "Plasma" and "Ions and field ionization". A browser GUI (`tools/gempic3d_gui.py`) edits and runs decks and shows the fields live.

The first application is a dense electron bunch overtaken at a small angle by a plane-wave laser,
with the radiation of the bunch resolved on the grid and acting back on it
(`examples/oblique_plane_wave.deck`).

## Build, test, run

```
make                      # build/gempic3d and build/tests (C++17, OpenMP, no external libraries)
make test                 # 105 checks against analytic references, a few minutes on one core
make physics              # ion collapse, undulator radiation energy: 13 s on an RTX 5090, 4.5 min on 24 cores
make MPI=1 BUILD=build-mpi                 # MPI (mpicxx); test: sh tests/mpi.sh build-mpi/gempic3d
make CUDA=1 CUDA_ARCH=-arch=sm_120 BUILD=build-gpu  # GPU (nvcc, cuFFT; sm_80 A100, sm_90 H100/GH200); add MPI=1 for both
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
  radius `r_max`, `ppc_x` × `ppc_y` electrons per cell on a regular lattice. Without `ion` blocks the ions are an
  immobile background (see "Ions and field ionization").
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
  (intensity), `zeta_center`, `cep`, linear (`pol_angle`) or circular (`helicity`) polarization; or any pulse
  from a LASY file (`type = file`, section Laser files). The analytic
  plane wave (`type = plane_wave`, the default) stays as before; it is not changed by the plasma.
- **Net current:** with plasma the k⊥ = 0 current is kept (a 1D plasma wave, not a net current in a periodic box),
  and the sponge uses the bound field of the beam list only.
- **Several bunches** (`bunch`, `bunch2`, …): electrons or positrons (`species`), Gaussian or flat-top, with
  offsets, each from its own seed (e.g. a drive and a witness bunch). The co-moving initial field
  −(∇⊥² + γ⁻²∂ζ²)ψ = ρ is solved for each bunch with its own γ and the forward fields are summed. `diag.txt`
  holds the moments of all bunches together; the beam files and openPMD have one species per bunch name.
- **MPI:** the plasma march runs down the ranks along ζ (see Parallelism).
- **Not yet:** profiles as expressions, chicanes and checkpoints (the next stages).

## Undulators and transport elements

As in GEM-PIC-RZ (`src/external.hpp`, `src/sim_external.cpp`, deck blocks `undulator`, `quadrupole`,
`plasma_lens`, `solenoid`, each repeatable): static magnetic fields at the lab position of a particle, added to the
gathered fields of the bunch push and of the streaming particles; the field solver does not see them.

- **Undulator** (planar with natural focusing, or helical): B0 = K k_u, linear ramps of `ramp_periods` at both
  ends (whole periods: no net angle). The summary prints the resonant wavelength λ_u (1 + K²/2)/(2γ²) (helical:
  1 + K²) of every bunch, and warns if `dz` > λ_u/16 (the transverse current oscillates with the period).
- **Quadrupole, plasma lens, solenoid** (with linear fringes): as in GEM-PIC-RZ.
- **Push:** the RK4 sub-steps of the bunch push resolve the elements by themselves: at most λ_u/32 inside an
  undulator (slippage to 1e-5) and 0.1/k_β inside a magnet (k_β of the bunch with the lowest energy), and they end
  on every breakpoint of an element (entrance, ends of the ramps, exit), where the field or its derivative jumps;
  the end stages of such a sub-step take the field just inside it. Without these rules RK4 drops to first order
  at a fringe (3 % in the canonical angular momentum of the solenoid check at L/50 steps).
- **Taper:** `K_function` (an expression of the lab position z) replaces `K`; it is tabulated with 32 points per
  period, as in GEM-PIC-RZ.
- **Chicane** (block `chicane`, thin, at `z`): every beam particle moves in ζ by r56 (γ/γ_ref − 1) − `delay`
  (r56 > 0: more energy moves towards the head; `delay` default |r56|/2, reference `gamma` or `energy`). The co-moving
  field of the beam moves with it: the co-moving fields R of the old and the new positions are subtracted and added
  (vacuum at the chicane), E_z, B_z and L follow from R as at the start; the radiation and the part of the field that
  is not co-moving stay. One rank only, as in GEM-PIC-RZ.

## Laser files (LASY)

`laser { type = file; file = pulse.h5; zeta_center = 100 um }` puts a pulse from a LASY / openPMD envelope file
(https://github.com/LASY-org/lasy, extension LaserEnvelope) on the grid, as `laser.file` of GEM-PIC-RZ
(`src/laser_file.*`, conventions shared with it). The file holds the complex envelope of E⊥ at a plane z = z_f as a
function of the lab time t, E_{x,y} = Re[p_{x,y} env e^{−iω_L t}] with the polarization (p_x, p_y), in `cartesian`
(axes t, y, x) or `thetaMode` geometry (axes t, r; components m = 0, cos mθ, sin mθ, summed on the Cartesian
points), as `electric_field` (V/m) or `normalized_vector_potential` (E = iω_L a − ∂_t a). Any spatio-temporal
structure is kept (chirps, pulse-front tilt, axiparabola or other structured light prepared in LASY).

Placement (GT2): the start level s = 0 is the plane z = 0 and its slice ζ is the time t = −ζ, so the file is a level
of the code; its time `t_center` (default: the intensity centroid) is put at `zeta_center`, i.e. the slice ζ holds
the file time τ = t_center + zeta_center − ζ, with the carrier e^{−iω_L τ} (the CEP of the file). With `file_z` = 0
the envelope is interpolated (trilinear in x, y, τ) to the points of the grid (and the half-nodes of the free-hand
y). With `file_z` ≠ 0 the field at z = 0 is the exact vacuum solution: the envelope is transformed in (t, y, x)
(FFT lengths `padding` × the file's time axis and `padding_xy` × its transverse axes), every plane wave
(ω = ω_L − Ω, k_x, k_y) gets the phase e^{i(k_z − ω)(0 − z_f)} at fixed retarded time, k_z = √(ω² − k⊥²), and the
evanescent part (k⊥ > ω) is dropped; no paraxial or envelope approximation. E.g. `file_z = 3 mm` starts a pulse
given at its focus 3 mm before the focus. The forward field is R = 2E⊥, L = 0 (as the Gaussian grid laser); the
grid then propagates it with the GEM scheme. Every rank fills only its own slices (MPI); a restart takes R from
the checkpoint.

The summary lists the file, its grid and field type, ω_L, the polarization, the placement, the energy fraction
dropped as evanescent, and warns when the envelope reaches the edge of the propagation grid (wrap-around: increase
`padding` / `padding_xy` or the file's box). The periodic box of the code must hold the pulse; the envelope is
zero outside the file's box. Not yet: feeding the pulse through the sponge at every step (`inject = boundary` of
GEM-PIC-RZ, needed for the flying focus) and LASY output.

## Checkpoints and restart

`output.checkpoint_every = N` writes the complete state after every N-th step, one binary file per rank
(`<output.dir>/checkpoint_NNNNNN_rRRRR.bin`, `src/sim_checkpoint.cpp`): level, step, transverse window, the forward
fields R, E_z, B_z, L, the real fields with the ghost slices, the currents of level s⁰ (beam and streaming
prediction), the beam list, the particles that join the next march, and the pending centroid sums of the window.
`gempic3d deck --restart latest` (or `--restart NNNNNN`) continues from the newest complete checkpoint in
`output.dir` with the same number of ranks and the same grid (the deck may change the output and the end of the
run); `diag.txt` and `ionization.txt` are cut back to the restart step and continue. `output.walltime_hours`
stops the run 1.5 mean step times before the limit, writes a checkpoint and exits with code 3, so a job script
can resubmit:

```
R=""; ls out/checkpoint_*_r0000.bin > /dev/null 2>&1 && R="--restart latest"
mpirun -np 4 gempic3d run.deck $R; [ $? = 3 ] && sbatch job.sh
```

## Functions and expressions

As in GEM-PIC-RZ (`src/expr.*`, ported): a block `constants` defines named values (a number with unit, or an
expression of the constants above it, in code units), and any single value of the deck may be an expression of
them, e.g. `waist = "2 * w0"` (quotes when it contains blanks). Numbers in expressions carry their units and are
converted to code units (lengths in k0⁻¹, densities in n_c, times in ω0⁻¹; also the units `k0^-1` and `n_c`);
operators `+ − * / ^` (also `**`, right-associative, −x² = −(x²)), comparisons, `&& || !`, `cond ? a : b`, and the
functions sin … atan2, exp, log (= ln), sqrt, abs, floor, ceil, min, max, pow, sign, step, if, gauss(x, s), rect(x, w).
A number with a unit of the wrong kind stays an error. Keys that take functions so far:

| block | key | variables | meaning |
|---|---|---|---|
| `plasma` | `density_function` | x, y, z, r | factor of the density (x, y relative to the axis, z the lab position; < 0 counts as 0) |
| `undulator` | `K_function` | z | taper K(z), replaces `K` |
| `laser` (`type = gaussian`) | `profile_function` | r | transverse field profile at z = 0, replaces `waist` and `z_focus` (the field is given where the run starts) |
| `laser` (`type = gaussian`) | `envelope_function` | zeta | longitudinal envelope of the field, replaces `tau_fwhm`; `zeta_center` (the reference of the carrier phase) defaults to the centre of the intensity |
| `bunch` | `profile_z` (with `distribution_z = function`) | zeta | current profile on [`zeta_min`, `zeta_max`], any normalization, ≥ 0, sampled by the inverse of its cumulative distribution |
| `bunch` | `profile_r` (with `distribution_r = function`), `ux_function`, `uy_function` | x, y, r, zeta | transverse density, sampled by rejection in r < `r_extent`; mean transverse momenta [m c] (the energy is kept) |
| `bunch` | `gamma_function` | zeta | γ(ζ) (energy chirp), replaces `gamma`/`energy`; `energy_spread` is relative to the local value |

```
constants { n0 = 3.5e18 cm^-3;  Lup = 100 um }
plasma    { density = n0;  density_function = "min(z / Lup, 1) * (1 + 0.2 * (r / (20 um))^2)" }
undulator { period = 2 cm;  periods = 100;  K_function = "1.2 * (1 - 0.05 * max(z - 1 m, 0) / (1 m))" }
```

## Ions and field ionization

As in GEM-PIC-RZ (`src/sim_ions.cpp`, `src/ionization.*`, `src/atomic_data.cpp`; deck keys in
`docs/input_deck.md`, block `ion`):

- **Mobile ions:** every `ion` block is one species, loaded with the fresh plasma at the positions of the plasma
  electrons (neutral pairs), with charge Z0, q/m = Z0/M and n_atom = fraction·n_e/Z0 (or `density`). The rest of
  the positive charge is an immobile background, which needs no source term: the field equations take only the
  currents. The ions stream through the window like the electrons (the same Boris push with their q/m, current
  q w dt/Δ · v). Each loaded species adds the number of plasma macroparticles.
- **Densities:** the charge of negative and positive streaming particles is deposited separately; the output has
  n_e/n0 and the ion charge density Z n_i/n0 (`ni` in the field files and the GUI frames, `n_i` in openPMD,
  a column `Zni/n0` in `axis_NNNNNN.txt`; `diag.txt` adds its maximum on the axis).
- **Field ionization** (`ionization = adk`): the ADK tunnel rate of each level (Ammosov, Delone, Krainov 1986;
  s-state form with l* = n* − 1 as in Chen et al., JCP 236, 220 (2013) and FBPIC)

  w = ω_a C²_{n*} (U/2U_H) [2 (U/U_H)^{3/2} E_a/E]^{2n*−1} exp[−(2/3)(U/U_H)^{3/2} E_a/E],
  n* = (k+1)(U_H/U)^{1/2}, C²_{n*} = 2^{2n*}/(n* Γ(n*+1) Γ(n*)),

  with the NIST ionization energy U of the level and the instantaneous |E| at the ion (grid fields and the
  analytic plane wave; the carrier is resolved, so there is no cycle averaging). Beyond the maximum of the
  expression (barrier suppression) the rate stays at its maximum. Per half-cell the rate is averaged over the
  fields at its two ends, (w(E_{n+1}) + w(E_n))/2, and sequential ionization k → k+1 → … is sampled with
  exponential waiting times (Monte Carlo per macroparticle). The random numbers are counter-based (a hash of
  seed, step, slice, pass and particle), so CPU and GPU draw the same numbers whatever the thread order.
- **Freed electrons:** each event frees one electron per represented atom at the ion, with its velocity; it
  streams from the next half-cell on. Trapped ones join the beam list as `inj_<name>` (beam files, openPMD;
  `beams = injected`). `ionization.txt` gives per step the charge-state distribution of the ions that left the
  window and the charge freed. `mobile = false` keeps ionizing atoms at rest (heavy dopants);
  `plasma.free_electrons = false` starts from a neutral gas. The energy taken from the laser by the ionization
  is neglected (dilute dopants), as in GEM-PIC-RZ.
- **Elements:** H, D, He, Li, Be, B, C, N, O, F, Ne, Na, Mg, Al, Si, Ar, Cu, Kr, Ag, Xe and Au, with all levels
  from the NIST Atomic Spectra Database (provenance in `src/atomic_data.cpp`); `ionization_potentials` replaces
  them.

```
plasma { density = 3.5e18 cm^-3; ppc_x = 2 }
ion    { element = N; charge_state = 5; density = 3.5e16 cm^-3; ionization = adk }   # dopant
ion    { element = He; charge_state = 2; fraction = 0.9 }                              # mobile He2+
```

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

**MPI with plasma.** The streaming particles cross the whole window in one march, so the march of a step runs
down the ranks: rank q marches its half-cells b−1 … a after rank q+1. The package down (q+1 → q) holds the
streaming particles at ζ_b, slice b of level s1 (k-space for the march, real space for the gather), the
prediction of the streaming current of half-cell b−1 that the solve of slice b used, and the change s0 → s1 of
half-cell b (extrapolation). The package up (q → q+1, after the first half-cell of q) holds the correction of R at
slice b and the final streaming current of half-cell b−1, which rank q+1 needs before its sponge and as the
prediction of the next step. Electrons trapped in half-cell a go to the beam list of the rank below; beam
particles that fall below `vz_min` in half-cell b stream on the rank above. The rest of the step (push, currents,
sponge) is as without plasma, and the densities and `ionization.txt` are gathered for the output. A beam-list
sub-step moves a particle by at most half the smallest half-cell in ζ, which keeps the RK stages within the
slices of the rank (and resolves slow trapped electrons). `tests/mpi.sh` (`tests/mpi_plasma.deck`: a0 = 2.5 grid
laser, ramp, mobile H⁺, N⁵⁺ ionized by ADK, a witness and a positron driver, electrons trapped by ionization
injection): 2, 3 and 4 ranks reproduce `diag.txt` and `ionization.txt` of one rank in every printed digit and the
axis lineouts to 1e-7 (float output).

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
noise, sponge, window shifts, block-edge crossings in a graded ζ grid), with Fourier y and with the free-hand y.
**First runs on a real GPU** (NVIDIA RTX 5090, CUDA 13.0, `make CUDA=1 CUDA_ARCH=-arch=sm_120`, 2026-10-30):
all checks of `make test` pass and agree with the CPU build line by line to the printed digits, except at
rounding level (cuFFT against the own FFT: round trips 2e-15 instead of 7e-16; the p-polarized p_x residual
−9.27e-6 instead of −9.14e-6, tolerance 1e-4; the constant-step ζ table 5e-13 instead of 3e-14). The 3D beam
wake of `tests/emu_wake.cpp` is identical bit for bit to the CPU build; `diag.txt` of 50 steps of
`examples/lwfa_plasma.deck` and 40 steps of `examples/oblique_plane_wave.deck` agree in every printed digit, the
axis lineouts to 1e-7 (E_z) and 7e-8 (n_e) of their maxima. `tests/mpi.sh` with the CUDA + MPI build
(OpenMPI 4.1.1, several ranks on one GPU): identical to one rank for 2, 3 and 4 ranks. With several ranks on GPUs
the particle exchange goes through host memory each step. Note: OpenMPI 4.1 with its UCX layer crashed now and
then inside `MPI_Finalize` after a complete run (in `opal_common_ucx_del_procs`); `mpirun --mca pml ob1` avoids it.

**Performance** (s per step):

| run | RTX 5090 | Xeon, 20 threads (same host) | atto node, 2 × 12-core Ivy Bridge, 24 threads | atto node, 4 ranks × 6 threads |
|---|---|---|---|---|
| `examples/lwfa_plasma.deck` (64 × 64 × 1201, 4 electrons per cell, 2 passes) | 0.42 | 3.9 | 3.05 | — |
| `examples/oblique_plane_wave.deck` (256 × 128 × 1601, 400 000 macroparticles) | 0.23 | 2.7 | 2.65 | 2.8 |

With plasma the march runs down the ranks one after the other (see MPI with plasma), so MPI gives memory per rank,
not speed: the LWFA example takes 3.3 s per step with 1 rank × 24 threads, 4.3 s with 4 ranks × 6 threads and 22 s
with 4 ranks × 1 thread on one atto node.

The GeForce card has 1/64 of its FP32 rate in FP64; a data-centre GPU (A100, H100, GH200) should gain several
times more. The small grids of `make test` take longer on the GPU (2.2 min) than on 20 threads (1.4 min).

## Input deck

The keys of all blocks are in [`docs/input_deck.md`](docs/input_deck.md). A deck is a sequence of blocks
`name { key = value }`; dimensional values need units; unknown keys, missing units and units of the wrong kind are
errors with file and line. Repeated `laser` and `ion` blocks become `laser2`, `ion2`, … (also for overrides).

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
| ions: H-like ions of mass 1e15 u against no ions (1D laser wake, a0 = 0.3); neutral pairs H+ / He2+ at rest in a parabolic channel with a ramp: fields, momenta, Z n_i = n_e | 1e-13; 0, 9e-16 |
| ADK rate of hydrogen-like levels against the Landau rate; code-unit tables against the SI formula | 2e-14; 7e-15 |
| field ionization in a plane-wave pulse (τ = 30 k0⁻¹, 4096 atoms) against the rate equations with E = −dA/dη: neutral H, fixed atoms, a0 = 0.01; N⁵⁺ → N⁶⁺ → N⁷⁺, mobile, a0 = 2 | 0.514 / 0.517; 0.220 / 0.213, 0.742 / 0.749, 0.0383 / 0.0380 (Monte Carlo scatter ~1 %); CPU and GPU identical |
| ion collapse (`make physics`, as the check of GEM-PIC-RZ): rigid bunch n_b = 100 n0, σ_r = 2.5 k0⁻¹ (dx = σ_r/10), hydrogen and deuterium, on-axis Z n_i against a radial test-ion model with full blowout and the ion self-field, in the compression phase (3 … 30 n0) | H 1.7 %, D 0.7 % (before, while the blowout forms: 3.6 %, 3.9 %); peak 209 n0 (H), 186 n0 (D) |
| several bunches: an electron and a positron bunch from the same seed; fields of two frozen bunches (γ = 200 electrons, γ = 50 positrons, offset) against the sum of each alone; flat-top rms length L/√12; transverse offset | 9e-17; 1e-7 (float output); 0.05 %; 0.005 |
| radiation energy balance (`make physics`, as `tests/undulator.sh` of GEM-PIC-RZ): coherent emission of a 1 fC, γ = 20 bunch (σ_z = 0.785, σ_r = 200 k0⁻¹) in 4 periods of a planar undulator (K = 0.5, λ_r = λ0), W_x − W_y = ½∫(E_x² + B_y² − E_y² − B_x²) after the undulator against the far-field radiation integral (Jackson 14.65) with the form factor of the bunch | −1.9 % at Δζ = λ/31, L_x = 6400 k0⁻¹; converged (second order in Δζ, 1/L_x² in the box) +0.6 % |
| undulators (one electron, γ = 100, K = 1): planar u_x amplitude K, no drift; slippage over 6 periods against the exact integral of (γ − p_z)/p_z; helical \|u⊥\| = K; natural focusing k_β = K k_u/(√2 γ) | 1e-6; 6e-17; 7e-6; 5e-7; 3e-4 |
| quadrupole x0 cos(√k L) and y0 cosh(√k L), plasma lens x0 cos(√k L) (γ = 1000, √k L = 1); solenoid with fringes: canonical angular momentum x p_y − y p_x − B r²/2 | 1e-7; 4e-8 (relative to B r²/2) |
| expressions: operators, functions, units and constants against direct values; a deck value as an expression; wrong unit kind | 1e-16; exact; error |
| grid laser `profile_function`, `envelope_function` = built-in Gaussian (start fields); bunch functions: flat-top `profile_z` rms, `gamma_function` per particle, ring `profile_r` mean radius (5 + σ²/5), rigid-rotation `ux/uy_function` per particle with the energy kept | 6e-14; 0.2 %; 3e-14; 0.03 %; 7e-15 |
| `density_function` = the built-in parabolic channel (3 steps of a laser wake); `K_function = 1` = `K = 1`; linear taper: slippage against ∫⟨(γ − p_z)/p_z⟩ at the local K | 6e-10 (float output); 0; 4e-4 |
| chicane (frozen γ = 1000 bunch): change of E⊥, B⊥ by a delay of 20 k0⁻¹ against the change of the co-moving start field; R56 = 50 k0⁻¹ with 1 % spread: shift of every particle | 8e-8 (float output); 7e-15 |
| checkpoint (plasma ramp, laser wake, mobile H⁺, ionizing N⁵⁺, bunch, moving window): 2 steps, checkpoint, restart, 2 steps against 4 steps; `tests/checkpoint.sh`: `--restart latest` from step 4 of 12 (1 and 2 ranks), wall-time stop | fields 5e-10 (thread sums), beam list and window exact; `diag.txt`, `ionization.txt` identical; exit code 3 |
| GUI (`tests/gui.sh`): frames, cuts, 2 ranks = 1 rank, plasma frames | pass |
| LASY files (`type = file`, HDF5 build): cartesian file of the built-in Gaussian (a0 = 1, w = 12, τ = 25, `pol_angle` 0.3, `cep` 0.4) at its plane = built-in (start fields); `normalized_vector_potential` = `electric_field` of the same pulse; `a0` rescaling; thetaMode (C_0) = built-in (radial interpolation, dr = dx/4); thetaMode with m = 1 (cos, sin) = cartesian file of the same field; file at the focus with `file_z` = 3 z_R (k w = 6): envelope at z = 0 against an independent exact reference (Hankel transform per frequency); 20 steps in vacuum = built-in; CPU = RTX 5090 (CUDA build) | 1e-12; 2e-5; 0; 5e-4; 6e-4; 3e-5 (the paraxial one-frequency Gaussian differs by 5 %); 6e-12; same digits |
| LASY file laser in `tests/mpi.deck` (a0 = 0.5, w = 40 µm, 0.16 ps, free-hand ζ): 2/3/4 ranks against 1 rank, CPU and CUDA + MPI builds; against the built-in Gaussian (τ interpolated between the file's samples on the free-hand ζ) | `diag.txt` identical; 1.3e-5 |
| output groups and openPMD (`tests/openpmd.sh`): periods and selections of two groups, openPMD attributes and input deck, particles = native beam file, fields = native field file and axis file on a uniform grid | pass; 1.5e-7; exact; 1.5e-11 |
| all of the above on a real GPU (RTX 5090, `make CUDA=1`); `tests/emu_wake.cpp` GPU against CPU; `tests/mpi.sh` with CUDA + MPI | pass, equal to the CPU build to rounding; bit for bit; identical |
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
  takes about 25 s per step on one core, 3 s on a 24-core node and 0.4 s on an RTX 5090 (see Parallelism).

- **Backward radiation is not described** (∂_s L dropped). A momentum gain from the recoil of backward
  or large-angle emission is outside the model by construction.
- **Radiation energy balance:** checked for coherent undulator emission (`make physics`, Validation). A periodic
  box samples the angular spectrum of the radiation at k⊥ = 2π m/L_x (the images of the bunch): the box must be
  several times 2π/(k Δθ) for an angular width Δθ of the emission. Energy diagnostics of the exchange with the
  external plane wave (interference term) and of the flux into the sponge are still missing.
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
