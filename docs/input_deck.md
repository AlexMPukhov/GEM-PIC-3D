# GEM-PIC-3D input deck

A block `constants` defines named values; any single value may be an expression of them (README, Functions and
expressions).

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
| bunch (0 … N, `bunch`, `bunch2`, …) | `name`, `species` | `bunch`, `bunch2`, …; electron | label (beam files, openPMD species); `electron` or `positron` |
| | `charge` | — | total charge (its magnitude; the sign comes from `species`) |
| | `gamma` *or* `energy` | — | Lorentz factor, or the kinetic energy (`MeV`, `GeV`, …) |
| | `energy_spread` | 0 | relative rms spread of p_z |
| | `sigma_x`, `sigma_y` | —, σ_x | rms transverse sizes |
| | `distribution_z`, `sigma_z`, `length` | gaussian | `gaussian` with rms `sigma_z` (in the lab at t = 0), `flattop` of full `length`, or `function`: `profile_z(zeta)` on [`zeta_min`, `zeta_max`] |
| | `distribution_r`, `profile_r`, `r_extent` | gaussian | `function`: transverse density `profile_r(x, y, r, zeta)` in r < `r_extent` (replaces `sigma_x`, `sigma_y`) |
| | `gamma_function`, `ux_function`, `uy_function` | — | γ(ζ) (replaces `gamma`); mean transverse momenta (x, y, r, zeta), the energy kept |
| | `emittance_x`, `emittance_y` | 0, ε_x | normalized rms emittance |
| | `zeta_center`, `x0`, `y0` | 0.75 window, 0, 0 | position of the centre in the window; transverse offset |
| | `macroparticles`, `seed` | 100000, k (block number) | each bunch draws from its own seed |
| | `shot_noise`, `noise_wavelength`, `beamlet` | random, —, 4 | `fawley`: bunching noise of the real electrons at noise_wavelength (Fawley, PRST-AB 5, 070701 (2002)) |
| bunch (`file`) | `file` | — | the bunch is read from an openPMD particle file (this code, GEM-PIC-RZ, FBPIC, HiPACE++, WarpX, …; HDF5 build). Path relative to the working directory or to the deck. The shape keys (`energy`, `sigma_*`, `emittance_*`, `distribution_*`, `species`) are then not allowed: everything comes from the file (README, Bunches from files) |
| | `file_species`, `file_iteration` | the only non-empty species, the last | species and iteration of the file |
| | `select` | all | expression of x, y, z, ux, uy, uz, gamma: particles with a non-zero value are kept (z: position relative to the charge centroid of the species, momenta in m_e c) |
| | `zeta_center` | — | the charge centroid of the selected particles is placed here (required) |
| | `x0`, `y0` | 0 | transverse shift |
| | `charge` | from the file | rescales the weights to this total charge |
| | `macroparticles`, `resample_smoothing`, `seed` | as in the file, 0, k | resample to this number (systematic resampling with probabilities ∝ charge, equal weights); Gaussian jitter of the resampled particles in units of the rms of each coordinate |
| numerics | `self_fields` | true | false: test particles in the laser only |
| | `start_relax` | true | the start field of the bunches is the steady state of the discrete scheme (README, Initial field); false: the co-moving field of the continuum equations as it is (versions before 2026-11-15: an odd-even mode in s of 1–2 % of E⊥) |
| | `box_mean` | keep | Fourier y with plasma: `keep` the currents at k⊥ = 0 (the box average: a 1D plasma wave, the return current of a beam), or `remove` them, so that no field has a box average (README, Periodic images) |
| | `zeta_filter`, `k_filter` | true, true | current filters |
| | `push_phase` | 0.05 rad | laser phase per RK4 sub-step |
| laser (`type`) | `type` | plane_wave | `plane_wave` (analytic, above), `gaussian` (on the grid): `a0`, `wavelength`, `waist` *or* `profile_function(r)`, `tau_fwhm` *or* `envelope_function(zeta)`, `zeta_center`, `z_focus` (0), `cep`, `polarization` (linear, circular), `pol_angle`, `helicity`, `enabled`; or `file` (on the grid, below) |
| laser (`type = file`) | `file` | — | LASY / openPMD envelope file (`laserEnvelope`, cartesian (t, y, x) or thetaMode (t, r)); HDF5 build (`make HDF5=1`). The pulse is given by the file: `waist`, `tau_fwhm`, `polarization`, … are not allowed (README, Laser files) |
| | `zeta_center` | — | window position of the file time `t_center` (slice ζ holds the file time τ = t_center + zeta_center − ζ) |
| | `t_center` | intensity centroid | time of the file put at `zeta_center` (the file's time base) |
| | `file_z` | 0 | plane of the file in the coordinates of the run (z = 0: its start); ≠ 0: exact vacuum propagation of every (ω, k_x, k_y) to z = 0 |
| | `a0` | from the file | rescales the field so that the peak of \|env\| is a0 ω_L |
| | `padding`, `padding_xy` | 2, 1 | FFT length of the propagation in t and transversally, in units of the file's axes (`file_z` ≠ 0 only; the summary warns when the envelope reaches the edge) |
| | `file_mesh`, `file_iteration`, `inject`, `enabled` | laserEnvelope, last, box, true | mesh and iteration of the file; `inject = box` (the field at the start; `boundary` not yet) |
| plasma | `density` | — | cm^-3 or n_c |
| | `free_electrons` | true | false: a neutral gas, all electrons come from field ionization (ion blocks with `charge_state = 0`) |
| | `z_points`, `z_values` | — | longitudinal profile (relative), linear between the points |
| | `density_file` | — | the profile from a file in the format of [density_file.md](density_file.md) (text table or openPMD mesh; the header defines the geometry): n(x, y, z), n(r, z) (rotated about the axis) or n(z), the latter with an optional transverse function in the file. A density with a unit in the file is the absolute density (`density` then only the reference n0), without unit a factor on `density`. Replaces `z_points`; `profile` only with a plain n(z) file; `r_max` and `density_function` still apply |
| | `file_z`, `file_outside` | 0, zero | with `density_file`: z of the run at the file's z = 0; the density beyond the file, `zero` or `hold` (edge values) |
| | `file_mesh`, `file_iteration` | density, last | with an openPMD `density_file`: mesh and iteration |
| | `profile`, `channel_depth`, `r_channel` | uniform | `parabolic`: n0 (1 + depth r²/r_channel²) |
| | `r_max`, `ppc_x`, `ppc_y` | ∞, 2, ppc_x | column radius, electrons per cell |
| | `smooth_length` | 0 | bubble closure: smoothing length of the plasma currents |
| | `density_function` | — | expression of x, y, z, r: a factor of the density (README, Functions and expressions) |
| ion (0 … N, `ion`, `ion2`, …) | `element` *or* `mass` | — | `H D He Li Be B C N O F Ne Na Mg Al Si Ar Cu Kr Ag Xe Au` (standard atomic mass, NIST ionization energies), or a mass (`1 m_p`, `4.0026 u`, `m_e`, `kg`) |
| | `charge_state` | 1 (H, D, mass); 0 with ionization | initial charge state Z0 (required for heavier elements without ionization) |
| | `fraction` *or* `density` | fraction = 1 | Z0 n_atom = fraction · n_e (needs Z0 ≥ 1), or the atom density at the reference density of the plasma profile (required for neutral atoms) |
| | `mobile` | true | false: the ions stay at rest (an immobile background, or fixed atoms that are still ionized) |
| | `ionization`, `max_charge_state` | none, atomic number | `adk`: ADK field ionization of the levels Z0 … max_charge_state − 1 |
| | `ionization_potentials` | NIST | energies of the levels 0, 1, … (`eV`, `keV`); all levels of an element, or the levels of an ion given by mass |
| | `name` | e.g. `H`, `N5+`, `ion2` | label (`inj_<name>`: its ionization electrons in the beam list) |
| undulator (0 … N) | `type` | planar | `planar`: B_y = B0 cosh(k_u y) cos ψ, B_z = −B0 sinh(k_u y) sin ψ; `helical`: B_x = B0 cos ψ, B_y = B0 sin ψ; ψ = k_u (z − z_start), B0 = K k_u m_e c/e |
| | `period`, `K` *or* `K_function` | — | undulator period, deflection parameter (K = 0.934 B0[T] λ_u[cm]), or a taper K(z) (expression of the lab position z) |
| | `periods` *or* `length`, `z_start` | —, 0 | number of periods or magnetic length (with the ramps); entrance |
| | `ramp_periods`, `natural_focusing` | 1, true | linear ramps of the amplitude at both ends; planar: the off-axis field (focusing in y, k_β = K k_u/(√2 γ)) |
| quadrupole, plasma_lens (0 … N) | `z_start`, `length`, `gradient` (T/m) | 0, —, — | quadrupole B_x = −G y, B_y = −G x (G > 0 focuses electrons in x); plasma lens B_φ = −G r (focuses electrons) |
| | `radius` | ∞ | plasma lens: B_φ = −G R²/r outside |
| solenoid (0 … N) | `z_start`, `length`, `field` (T), `ramp` | 0, —, —, 0 | B_z = B f(z), B_r = −(r/2) B f′(z), linear fringes of length `ramp` (use ramp > 0: the fringes focus) |
| chicane (0 … N) | `z`, `r56`, `delay`, `gamma` *or* `energy` | —, —, \|r56\|/2, — | thin chicane at z: ζ += r56 (γ/γ_ref − 1) − delay for every beam particle (README) |
| numerics | `iterations`, `predictor` | 2, extrapolate | passes per slice; prediction of the streaming current (`previous`: as GEM-PIC-RZ) |
| | `vz_min`, `vz_beam`, `trap_advance` | 0.8, 0.9, 1 | trapping and return to streaming, as in GEM-PIC-RZ |
| output | `dir`, `every`, `particles`, `frames` | out, 10, true, false | `every`: period of `diag.txt` and of the main output's fields; `particles`: start/end dumps; `frames`: frame_NNNNN.bin for the GUI |
| | `slice_length` | 0 | > 0: `slices_NNNNNN.txt` with the moments of every beam species per slice of this length along ζ, at the steps of `diag.txt` |
| | `checkpoint_every`, `walltime_hours` | 0, 0 | checkpoint every N steps; stop with a checkpoint (exit code 3) before this wall time (README, Checkpoints) |
| | `fields`, `beam_every`, `beams`, `axis` | all, 0, all, false | main output group (README, Output): fields `ex ey ez bx by bz ne ni`, beams `bunch trapped injected` |
| | `format`, `xi_stride`, `particle_stride` | native, 1, 1 | `native`, `openpmd` or `both` |
| | `openpmd_file`, `openpmd_grid`, `openpmd_dy`, `openpmd_dzeta`, `hdf5_single`, `hdf5_compression` | openpmd/data_%06T, uniform, smallest step, true, 0 | openPMD settings |
| diag | `names` | — | diagnostic groups, each a block `diag.<name>` with the keys of the output group (`every` 0, `beam_every` = every, format and openPMD settings from `output`) |

`diag.txt` has one line per output step: z, the centroid (x, y, ζ), the rms sizes, ⟨p_x⟩, ⟨p_y⟩,
⟨p_z⟩, ⟨γ⟩, ⟨p·k̂⟩ (along the laser), the rms of p_z, the position of the bunch in the pulse
(η − η_c in FWHM), the local amplitude a0 g, and the field and kinetic energy in J. `particles_start.txt`
and `particles_end.txt` hold the macroparticles.

`bunches.txt` (as in GEM-PIC-RZ) has one line per output step and beam species (every bunch by its name, `trapped`,
`inj_<ion>`): step, z [mm], name, number of macroparticles, charge [pC], mean kinetic energy [MeV], rms energy spread
dE/E (of γ, accumulated centred), rms sizes σ_x, σ_y [µm], normalized rms emittances ε_nx, ε_ny [µm]
(√(⟨x²⟩⟨u_x²⟩ − ⟨x u_x⟩²), centred), ⟨ζ⟩ and σ_z [µm], and the centroid ⟨x⟩, ⟨y⟩ [µm]. `slices_NNNNNN.txt`
(`output.slice_length`) has the same columns per slice of ζ (name, slice centre [µm], then N … ⟨y⟩); empty slices are
left out.

`ionization.txt` (with ionizing ion blocks): per step and ionizing species, the step, z [mm], the name, the mean
charge state and the charge freed in the step [pC], then the charge-state fractions f(0) … f(Zmax) of the ions that
left the window in this step.
