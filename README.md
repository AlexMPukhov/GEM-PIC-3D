# GEM-PIC-3D

Galilean electromagnetic particle-in-cell code in three dimensions, written from scratch on the
model of GEM-PIC-RZ. It is **not** a quasi-static code: the forward field keeps its slow derivative
and the laser carrier and the radiation of the beam are resolved, while the propagation step is set by
the slow evolution, not by a Courant condition. The transverse directions are Fourier modes, so the
field solve of a slice is a closed-form scalar problem per (k_x, k_y).

The first application is a dense electron bunch overtaken at a small angle by a plane-wave laser,
with the radiation of the bunch resolved on the grid and acting back on it
(`examples/oblique_plane_wave.deck`).

## Build, test, run

```
make                      # build/gempic3d and build/tests (C++17, OpenMP, no external libraries)
make test                 # 19 checks against analytic references, a few minutes on one core
make MPI=1 BUILD=build-mpi                 # MPI (mpicxx); test: sh tests/mpi.sh build-mpi/gempic3d
make CUDA=1 NVCC=nvcc CUDA_ARCH=-arch=sm_80 BUILD=build-gpu    # GPU (nvcc, cuFFT); add MPI=1 for both
build/gempic3d examples/oblique_plane_wave.deck --check            # summary and memory estimate
build/gempic3d examples/oblique_plane_wave.deck                    # run, results in oblique/
build/gempic3d examples/oblique_plane_wave.deck laser.enabled=false output.dir=ref   # reference
tools/compare.py oblique ref                                       # <pz> with and without the laser
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
over each half-cell, from the front slice (L = E_z = B_z = 0) to the rear. Per (k_x, k_y), R and L of
slice n are linear in its E_z and B_z; inserted into the B_z and E_z equations the cross terms cancel
exactly, and

```
B_z = [B_z(n+1) − (h/2) C(n+1) − (h/4) S_B] / [1 + (h/4)(Δ/2 + h/4) k⊥²]
E_z = [E_z(n+1) − (h/2) Q(n+1) − (h/4) S_E − h j_z] / [same]
```

with C = a E_y − b E_x, Q = −a B_y + b B_x, and S_B, S_E the same combinations of the known parts of R
and L (verified symbolically). R and L follow. The current at a slice is the mean of the two adjacent
half-cells at both levels.

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
periodic box would drive uniform fields.

**Initial field.** The co-moving field of the bunch, −(∇⊥² + γ⁻²∂_ζ²)ψ = ρ, R = (1 + β)E⊥,
E⊥ = −∇⊥ψ: one tridiagonal solve in ζ per (k_x, k_y). E_z, B_z and L then follow from the march with
Δ = 0.

**Boundaries.** The transverse box is periodic. An absorbing sponge at the edges damps the radiation
part R − R_b of the forward field, where R_b = 2E⊥ with k⊥²ψ = ρ is the bound field of the charge of
the slice (v = c limit, local in the slice). Damping all of R instead drains the bunch through its own
Coulomb field at the box edges, like a resistive wall: the first version lost 6 × 10⁻⁴ of ⟨p_z⟩ per
millimetre this way. The transverse window follows the bunch centroid in whole cells; the shift is an
exact phase factor in Fourier space. Particle coordinates stay continuous, since the analytic laser
is not periodic in the box; only grid indices wrap.

## Parallelism

**CPU and GPU from one source.** `src/portable.hpp` writes every kernel once as a lambda that runs as an OpenMP
loop or as a CUDA kernel (`--extended-lambda`); fields and particles live in `Buf<T>` device buffers;
`src/fftmany.hpp` batches the 2D FFTs over slices (own FFT with OpenMP over planes, or cuFFT). The march is
local in k: the FFTs are taken out of it (currents forward before, fields inverse after), so every
(k_x, k_y) is one recurrence over the slices (GPU: one thread per mode; CPU: OpenMP over the modes). The
sponge on R − R_b is therefore applied after the march of a step, not slice by slice.

**MPI along zeta** (`src/comm.*`). Rank q owns the slices [a, b), rank P−1 at the front, and the particles of
the half-cells [a+1, b]. Per step: push; particles that enter the half-cells of the rank below are handed over
during or at the end of the push with the rest of the step; the three top half-cells of unfiltered current go
to the rank above; the march receives slice b (k-space) from above and sends slice a below; two ghost slices of
real fields go below for the next push. The co-moving initial field is a pipelined tridiagonal solve. Each rank
stores only its own slices (about 256 bytes per grid point). The transverse window shift of step k uses the
centroid of step k − `window_lag` (non-blocking reduction, also in serial runs). The coupling march(q+1) →
push(q) → march(q) limits the speedup to about P/2; the main gain is the memory per rank.

**Status.** `tests/mpi.sh`: 2, 3 and 4 ranks reproduce the single-rank run exactly (laser, dense bunch, Fawley
noise, sponge, window shifts, block-edge crossings). The CUDA build (also with MPI) compiles with nvcc 13.4 for
sm_80 but has not been run on a GPU yet: run `build-gpu/tests` and `tests/mpi.sh` there first. With several
ranks on GPUs the particle exchange goes through host memory each step.

## Input deck

| block | key | default | meaning |
|---|---|---|---|
| grid | `lambda0` | laser wavelength | reference wavelength (code units) |
| | `nx`, `ny` | — | transverse modes, powers of 2 |
| | `Lx`, `Ly` | —, `Lx` | periodic box |
| | `nzeta`, `window_length` | — | slices and length of the ζ window (front at ζ = window_length) |
| | `dz`, `z_end` | — | field step along z, end of the run |
| | `dz_push` | `dz` | largest particle sub-step |
| | `sponge`, `sponge_rate` | 0.1 min(Lx, Ly), 0.1 | width of the absorbing layer, damping per step at the edge |
| | `move_x`, `move_y` | true | the transverse window follows the bunch |
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
| output | `dir`, `every`, `particles` | out, 10, true | |

`diag.txt` has one line per output step: z, the centroid (x, y, ζ), the rms sizes, ⟨p_x⟩, ⟨p_y⟩,
⟨p_z⟩, ⟨γ⟩, ⟨p·k̂⟩ (along the laser), the rms of p_z, the position of the bunch in the pulse
(η − η_c in FWHM), the local amplitude a0 g, and the field and kinetic energy in J. `particles_start.txt`
and `particles_end.txt` hold the macroparticles.

## Validation (`make test`, `tests/mpi.sh`)

| check | result |
|---|---|
| FFT: pair transforms against the direct DFT, round trip | 1e-15 |
| vacuum: ω of a forward wave tilted by 0.1 rad against ω = 2κk⊥²/(4κ² − k⊥²), amplitude over 40 steps | 0.35 % (32 points per λ), 4e-5 |
| one electron, a0 = 5, θ = 0.1, γ = 50, s- and p-polarized: h = γ − p·k̂ conserved; peak γ from h and p⊥ + qA; p after the pulse (Lawson–Woodward) | 1e-8 / 2e-6; 0.04 % / 0.06 %; 6e-8 / 2e-6 |
| field of a long round Gaussian bunch against Gauss's law, E_r and B_y at r = σ, 2σ | 0.4–1.7 % (macroparticle noise) |
| γ = 1000 bunch in its own field over 100 steps: ⟨γ⟩, field energy | 3e-8, 0.1 % |
| dense γ = 50, 160 pC bunch with and without the sponge | 4e-7 |
| batched FFT against the plane-by-plane FFT | exact |
| MPI, 2/3/4 ranks against 1 rank (`tests/mpi.sh`) | identical |

The p-polarized case needs `push_phase` ≈ 0.02: γ drops to about 2 once per cycle there (p⊥0 ≈ a0),
and at 0.05 the momentum after the pulse is off by 5e-5. The pulse must start and stop at ≥ 3.5 FWHM.
At 2.5 FWHM the residual A ~ 10⁻³ a0, amplified by the cross term 2p⊥0·A/h, leaves 4e-4 in p_z.

## Limits and next steps

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
  periodic box need a box-size scan.
- **Grid fields over a step.** The grid fields are frozen at level s0 during the particle step, as in
  GEM-PIC-RZ; the laser is exact. Halving `dz` checks this.
- **Performance.** No FFTW option, checkpoints or HDF5/openPMD output yet; the GPU build is untested. The
  production deck needs about 13 GB in total (256 bytes per point), split over the MPI ranks.
