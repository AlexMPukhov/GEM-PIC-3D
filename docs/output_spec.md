# Output groups: the common specification of QUARZ, GEM-PIC-RZ and GEM-PIC-3D

The three codes configure their output the same way, following the diagnostic groups of QUARZ. This file
states the common part and, per code, what differs. GEM-PIC-3D implements it as described here (README,
section Output); GEM-PIC-RZ is to follow.

## Groups

An output group writes fields every `every` steps and beam particles every `beam_every` steps, restricted
to its selection of fields, beams, slices and particles, in its format, into its own directory.

- **The main output** is the group of block `output`, written into `output.dir`.
- **Diagnostic groups** are named in `diag.names` (blank or comma separated). Group `<n>` is the block
  `diag.<n>` and writes into `<output.dir>/<n>/`. Its keys default to those of the main output where that
  makes sense (format and openPMD settings); `every` defaults to 0 (no fields), `beam_every` to `every`.
- A group writes nothing at a step where neither of its periods is due. Steps are counted from 0; the
  initial state (step 0) is written by every group whose period is non-zero.
- `output.every = 0` with named groups gives only the groups.

| key | main output | `diag.<n>` | meaning |
|---|---|---|---|
| `every` | code default (GEM-PIC-3D: 10, also the period of `diag.txt`) | 0 | field output period [steps] |
| `beam_every` | 0 | `every` | beam particle output period [steps] |
| `fields` | `all` | `all` | `all`, `none`, or a list of the code's component names |
| `beams` | `all` | `all` | `all`, `none`, or a list of beam (species) names |
| `axis` | false | false | also write the on-axis lineouts `axis_NNNNNN.txt` with the fields |
| `format` | `native` | main output | `native`, `openpmd` or `both` |
| `xi_stride` | 1 | 1 | write every n-th slice (openPMD) |
| `particle_stride` | 1 | 1 | write every n-th particle of each beam |
| `openpmd_file` | `openpmd/data_%06T` | main output | file pattern in the group directory; `%T` or `%0<w>T` is the step |
| `openpmd_grid` | `uniform` | main output | `uniform`, `native` or `both`: records on a uniform grid and/or on the code's own nodes |
| `openpmd_*` | | main output | further openPMD settings of the code (spacing of the uniform grid, precision, compression) |

The deck syntax follows each code (QUARZ: `diag.light.every = 5`; GEM-PIC-RZ and GEM-PIC-3D: a block
`diag.light { every = 5 }`). Command-line overrides `diag.light.every=5` work in all three: the block name is
everything before the last dot.

## Files

- **Native fields** `fields_NNNNNN.*` in the code's own format, holding the selected components only.
- **Native beams** `beam_<species>_NNNNNN.*`.
- **Axis lineouts** `axis_NNNNNN.txt`: one line per slice on the axis, ξ first, then the code's on-axis
  quantities (always the longitudinal field and the transverse force on a particle moving at c).
- **openPMD** (standard 1.1.0, file based, one file per output step, basePath `/data/%T/`): meshes `E`, `B`
  and the densities, particles per species with `position`, `positionOffset`, `momentum` (per real
  particle), `weighting` (real particles per macroparticle), `charge`, `mass`. Every file stores the input
  deck after the command-line overrides (GEM-PIC-3D: dataset `/data/<T>/input_deck`). Non-uniform grids are
  written either interpolated to a uniform grid (readable by every openPMD reader) or exactly on the code's
  nodes as rectilinear meshes (`gridType = "rectilinear"`, node positions in `<basePath>gridCoordinates/…`,
  the proposal for openPMD 2.0 that QUARZ and GEM-PIC-RZ use; geometry `other` until it is standard).
- Particles are written with the full data dumps; for groups with selected fields the user decides with
  `beams` and `beam_every` whether particles are written too.

## Per code

| | QUARZ | GEM-PIC-RZ | GEM-PIC-3D |
|---|---|---|---|
| field names | `psi ez er eth br bth bz ne ni rhob nz_<species> a` | as QUARZ where they exist | `ex ey ez bx by bz ne` |
| beams | beam block names | beam names | `bunch`, `trapped` (plasma electrons in the beam list) |
| native fields | `fields_NNNNNN.bin` (r–ξ planes) | text or HDF5 (sub-choice) | `fields_NNNNNN.bin`: the planes y = 0 and x = 0, the GUI frame format |
| native beams | `beam_<n>_NNNNNN.bin` | as before | `beam_<species>_NNNNNN.txt`: x, y, ζ [µm], p [m c], weight [real particles] |
| axis | ξ, E_z, ψ, n_e, ρ_b, slope | as QUARZ | ζ [µm], E_z, E_x − B_y, E_y + B_x [m c ω0/e], n_e/n0 with plasma |
| openPMD geometry | thetaMode | thetaMode | cartesian, axes (z, y, x), data [ζ][y][x] |
| openPMD build | openPMD-api | HDF5 C library | HDF5 C library (`make HDF5=1`) |
| openPMD keys | `openpmd_backend`, `openpmd_dr`, `openpmd_rmax`, `openpmd_options` | as RZ | `openpmd_dy`, `openpmd_dzeta` (uniform grid spacing; default the smallest step), `hdf5_single` (float32, default true), `hdf5_compression` (0–9) |
