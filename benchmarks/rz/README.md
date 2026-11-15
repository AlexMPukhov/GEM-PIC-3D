# GEM-PIC-3D against GEM-PIC-RZ: offset bunch in a round hollow plasma channel

An axisymmetric structure with a beam off its axis is the case both codes can do: GEM-PIC-RZ with the azimuthal
modes m = 0, 1 about the channel axis, GEM-PIC-3D on its Cartesian grid with the channel given by
`plasma.density_function`. The monopole wake (energy loss, E_z) and the dipole wake (the transverse kick along the
bunch, which drives the beam break-up) are compared.

- Plasma 10¹⁵ cm⁻³ (c/ω_p = 168 µm), bore radius 0.5 mm with a 50 µm linear edge, wall out to r = 1.2 mm.
- Bunch 1 GeV, 2 nC, σ_r = 20 µm, σ_z = 200 µm, offset x0 = 20 µm; window 3 mm, Δζ = 10 µm.
- 10 steps of 2 mm (Δ h ω_p² = 0.7), beam sub-steps 0.2 mm.
- GEM-PIC-3D: 256 × 256 modes, box 3.2 mm (dx = 12.5 µm), 2 × 2 electrons per cell. GEM-PIC-RZ: dr = 5 µm.

```
gempic3d  channel_3d.deck          # 75 s on 24 cores (2 x 10^7 grid points, 7 s per step)
gempic_rz channel_rz.deck          # 10 s
python3 compare.py chan_3d chan_rz # exit code 1 beyond the tolerances (3 %, 6 %, 10 %)
```

## Result (atto, 2026-11-15; GEM-PIC-RZ main with PR #38)

| quantity at z = 20 mm | GEM-PIC-3D | GEM-PIC-RZ | difference |
|---|---|---|---|
| energy loss of the bunch | 0.8116 MeV | 0.7981 MeV | +1.7 % |
| E_z on the axis, bunch and first wake period behind it (largest difference / largest \|E_z\|) | | 3.30 × 10⁻⁴ m c ω0/e | 5.2 % |
| transverse kick ⟨u_x⟩ of 8 slices of the bunch, tail … head | 0.1148 0.0958 0.0717 0.0469 0.0274 0.0136 0.0049 0.0021 | 0.1137 0.0960 0.0719 0.0480 0.0275 0.0131 0.0057 0.0022 | 0.6 % (charge-weighted rms / largest) |

Both codes lose nothing in the first step (the bunch starts with its vacuum field) and then 0.08–0.11 MeV per step.
The wake behind the first period is not steady at these 10 steps in either code (the plasma starts abruptly at
z = 0; E_z at a fixed ζ changes by ±15 % from step to step, in both codes alike), and the codes differ there by up
to 23 % of the largest E_z at the rear end of the window, mostly a phase difference that falls with the transverse
resolution of the edge (table below, last column).

## Scans

Energy loss over the 20 mm [MeV]; GEM-PIC-RZ: 0.7979 (dr = 2.5 µm), 0.7981 (5 µm), 0.7998 (12.5 µm).

| GEM-PIC-3D | box | dx | energy loss | vs RZ | E_z at ζ = 0 (RZ: −2.94 × 10⁻⁴) |
|---|---|---|---|---|---|
| `box_mean = keep` (default) | 3.2 mm | 12.5 µm | 0.8116 | +1.7 % | −2.14 × 10⁻⁴ |
| | 3.2 mm | 6.25 µm | 0.8082 | +1.3 % | −2.44 × 10⁻⁴ |
| | 4.8 mm | 12.5 µm | 0.8081 | +1.3 % | −2.20 × 10⁻⁴ |
| | 6.4 mm | 12.5 µm | 0.8061 | +1.0 % | −2.21 × 10⁻⁴ |
| `box_mean = remove` | 4.8 mm | 12.5 µm | 0.7727 | −3.2 % | −1.74 × 10⁻⁴ |

(The rows with 4.8 and 6.4 mm were run before the start field was relaxed, `numerics.start_relax`; at 3.2 mm that
changed the loss by 0.0008 MeV.)

- **Periodic images:** with the wall 4 c/ω_p thick, the box size changes the energy loss by 0.7 % between 3.2 and
  6.4 mm. The wall screens the images.
- **`box_mean = remove` is worse here:** the wake of a channel has an E_z that is uniform over the bore, so its
  box average is physical (the bore is 8 % of the 3.2 mm box). With a sharp wall the removal lowered the wake behind
  the bunch by 30 % (3.2 mm), 11 % (4.8 mm) and 6 % (6.4 mm). Keep the default for channels.
- **Transverse resolution:** 12.5 µm (0.6 σ_r of the bunch, 4 cells in the wall edge) gives the energy loss and the
  dipole kick to 1–2 %; the phase of the wake several periods behind the bunch needs the finer grid.
- **What was wrong before this comparison:** the bunch started with the continuum co-moving field, which is not the
  steady state of the discrete scheme, and with plasma also with the uniform E_z of its sheet charge in the periodic
  box. E_z at the bunch then alternated from step to step by ±10 % (dx = 12.5 µm) to ±35 % (6.25 µm) of the wake
  field, and the first step cost 0.023 MeV. Both are fixed (README, Initial field; check 29).
