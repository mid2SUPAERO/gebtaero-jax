# gebtaero-jax

**Differentiable nonlinear aeroelasticity of high-aspect-ratio wings, in pure Python/JAX.**

`gebtaero-jax` is a modern re-implementation (thanks claude) of two research codes:

* [**GEBTAero**](https://github.com/mid2SUPAERO/GEBTAero) (ISAE-Supaero / French Air Force Academy): a
  geometrically exact beam (GEBT, W. Yu) coupled with strip aerodynamics, built for fast flutter/divergence
  computation and aeroelastic tailoring of composite wings. Fortran + MUMPS + ARPACK + CalculiX.
* [**Aeroelasticity.jl**](https://github.com/byuflowlab/Aeroelasticity.jl) (BYU FLOW Lab): a modular framework that
  couples implicit submodels (2D aerodynamics, typical section, GXBeam, rigid body) into monolithic systems.

Both are rebuilt around one idea: **write the residual `R(ẋ, x, p, t) = 0`, let JAX do the calculus.**
Tangent stiffness, mass matrices, the implicit derivatives of nonlinear equilibria, eigenvalue sensitivities and
the gradient of the flutter speed with respect to *any* design parameter (ply angles, stiffnesses, CG position,
controller gains…) all come from automatic differentiation. No Fortran, no compiled dependencies, no hand-derived
Jacobians.

```python
import gebtaero_jax as gx

cs = gx.CrossSection.from_isotropic(EI2=9.77e6, GJ=0.99e6).with_mass(mu=35.71, i22=8.64e-4, i33=8.64, nu=0.18288)
goland = gx.straight_wing(cs, span=6.096, chord=1.8288, a=-0.34, n_elements=10, aero=gx.Aero.PETERS)

gx.flutter_speed(goland, gx.FlightCondition(rho=1.225), v_min=100, v_max=160)
# CriticalPoint(flutter: V=136.5067 m/s, f=11.1914 Hz)
```

```python
import jax.numpy as jnp

def build(p, V):  # any pytree of design variables -> (model, flight)
    cs = gx.CrossSection.from_isotropic(9.77e6, 0.0, p["GJ"]).with_mass(35.71, 8.64e-4, 8.64, nu=p["nu"])
    return gx.straight_wing(cs, 6.096, 1.8288, -0.34, 10), gx.FlightCondition(velocity=V, rho=1.225)

r = gx.flutter_speed_and_grad(build, {"GJ": 0.99e6, "nu": 0.18288}, 100, 160)
r.dV_dparams  # {'GJ': 1.0092e-04, 'nu': -95.39}  -- matches finite differences to ~7 digits
```

## Features

**Structure** — geometrically exact beam in GEBT's mixed formulation (displacements, Rodrigues rotations, internal
forces/moments, momenta): large deflections, arbitrary 6×6 flexibility (zero entries = rigid DOFs) and mass
matrices, multi-section wings with sweep/dihedral/twist, gravity with CG offsets, tip and distributed loads
(dead or follower), moving root frame, stiffness-proportional structural damping (GXBeam style).

**Aerodynamics** (strip theory, shared with the 2D section models) — steady, quasi-steady (with and without apparent
mass), Wagner (Jones), Peters finite-state inflow; lift slope, zero-lift angle, `c_m0`, `c_d0`, Prandtl–Glauert.

**Cross sections** — isotropic values, solid rectangle, and **composite plates from Classical Laminate Theory**
(differentiable w.r.t. ply angles; replaces GEBTAero's CalculiX homogenisation for plate wings).

**Analyses** — nonlinear static aeroelasticity (`custom_root` ⇒ implicitly differentiable), modal analysis about
the deformed state, V-g sweeps with mode tracking, flutter/divergence search, exact critical-speed gradients,
nonlinear transient simulation (generalised-α in `lax.scan`, differentiable), trim by Newton on AD derivatives.

**Coupled-model framework** — Aeroelasticity.jl's `CoupledModel` in JAX: plug any `jax.numpy` residuals together
with a coupling function and get `linearize`, `steady_state`, `dense_eigen`, `correlate_eigenmodes`,
critical-value search, gradients and time marching. Typical section with all four aero models included.

**Mode tracking + eigen-adjoint** (`gx.tracking`) — S. He's bordered eigenvalue adjoint generalised to the complex,
non-symmetric pencil of the beam: MAC tracking against reference shapes (Hungarian assignment), adjoint gradients of the
MAC loss, tracked frequencies/damping and the flutter speed with **one bordered solve per tracked function** and
matrix-free AD partials — cost independent of the number of design variables.

**Concentrated masses** — `BeamModel.with_point_mass` (motors, booms, pods, stores), differentiable.

**HALE examples** — `examples/hale_wings.py`: beam models of the EcoHALE and Dawn solar-HALE wings reconstructed from
their published designs (assumptions documented), with an AD trim (`trim_alpha`) at any load factor.

**Migration** — reader for GEBTAero's Fortran `.dat` input files.

## Validation

| Case | gebtaero-jax | Reference |
|---|---|---|
| Patil wing structural modes (Hz) | 0.357, 2.258, 4.941, 5.046, 6.428 | GEBTAero: 0.357, 2.257, 4.941, 5.048, 6.428 |
| Goland flutter, Peters / QS + added mass / QS (m/s) | 136.51 / 64.33 / 35.16 | GEBTAero: 136.5 / 64.34 / 35.26 (lit. 137) |
| Goland divergence (m/s) | 253.2 | GEBTAero: 253 |
| Goland, swept 15° + β, dihedral 15°, twist 45° + α | 136.51 (all) | GEBTAero: 136.5 |
| Goland, Aeroelasticity.jl set-up | 135.1 m/s, 69.01 rad/s | AJ: 140 m/s on a 5 m/s grid, 69.0 rad/s; Palacios: 141 m/s, 69.8 rad/s |
| Patil flutter / divergence (m/s) | 32.15 / 37.23 | GEBTAero: 32.12 / 37.25; Patil et al.: 32.21 / 37.29 |
| Patil, gravity-deformed flutter (m/s) | 23.13 | GEBTAero: 23.35 (approximate hand-coded aero Jacobian) |
| Typical section V_f: Wagner / Peters | 2.170 / 2.165 | AJ: "around 2.2"; exact Theodorsen (in-repo): 2.184 |
| Typical section divergence (all aero models) | 2.82843 | analytic √8 |
| AD vs finite differences (static, eigen, flutter) | 6–8 digits | — |

All of the above are in the test suite (`pytest`, ~6 min).
The tracking adjoint is checked against the left/right-eigenvector formula (1e-7) and finite differences.

## Install

```bash
pip install -e ".[dev]"     # jax, numpy, scipy, matplotlib (+ pytest, jupyter for dev)
pytest -n auto              # 35 tests
```

64-bit floats are enabled on import (`jax_enable_x64`). CPU is the expected target; systems are small and dense.

## Notebooks

| Notebook | Content | Origin |
|---|---|---|
| `01_patil_wing_modes_and_flutter` | modes and shapes, V-g diagram, flutter/divergence, aero models | GEBTAero `IsotropicWing.py`, `cas_test/Patil.py` |
| `02_goland_wing_benchmarks` | 5 aero models, invariance tests, Aeroelasticity.jl set-up, legacy `.dat`, flutter mode | GEBTAero `cas_test/Goland.py`, AJ `goland-stability` |
| `03_nonlinear_static_and_deformed_flutter` | large static deflections, implicit gradients, trim, flutter vs AoA, gravity | GEBTAero `StaticLoads`, `EquilibriumAoA`; AJ `cantilever-steady` |
| `04_time_domain_simulation` | flutter onset in time, growth-rate check, 1−cos gust on a deformed wing | GEBTAero `FlutterVtk`/`TemporalFlutterSpeed` |
| `05_composite_tailoring_with_gradients` | CLT plate, ply-angle sweep, exact dV_f/dθ, SLSQP tailoring | GEBTAero `PlateComposite.py`, `examples/optimisation` |
| `06_typical_section_aeroelasticity_jl` | 4 aero models, root loci, Theodorsen check, simulation, sensitivities | AJ `section-stability`, `section-simulation` |
| `07_custom_coupled_models` | user-defined submodels: cubic spring + flap actuator, LCO, gradient through a simulation | AJ coupling framework |
| `09_hale_aeroelasticity_ecohale_dawn` | two solar HALE designs (EcoHALE CO₂ optimum, DawnDesignTool point design): mass closure, modes, trimmed nonlinear deflection, flutter vs. root incidence, gust in time, battery placement, adjoint wall redistribution at constant mass | `EcoHALE_AeroSandbox_1.ipynb`, [DawnDesignTool](https://github.com/peterdsharpe/DawnDesignTool) |
| `08_crm_mode_tracking_adjoint` | CRM-like swept wing, mode crossing and MAC tracking, He's eigen-adjoint (complex, matrix-free) vs FD and left/right formula, cost vs n_x, 32-variable wingbox sizing with flutter + tracked-mode constraints | S. He's note, `sensitivity_modetracking.ipynb` |

## Package layout

```
src/gebtaero_jax/
  beam.py           GEBT mixed-formulation residual (element + system), FlightCondition
  aerodynamics.py   2D strip models (steady, quasi-steady, Wagner, Peters), Prandtl–Glauert
  cross_section.py  flexibility / mass matrices, CLT composite plate
  materials.py      isotropic & orthotropic materials, plies, laminates (ABD)
  wing.py           Frame, WingSection, Wing -> BeamModel discretisation, Aero enum
  solvers.py        static (custom_root), modal, transient simulation
  stability.py      flutter/divergence search, V-g sweeps, eigenvalue & critical-speed gradients
  coupled.py        generic CoupledModel framework (Aeroelasticity.jl)
  section.py        typical section + aero submodels, exact Theodorsen reference
  tracking.py       MAC mode tracking, bordered eigen-adjoint (He), tracked-mode & flutter gradients
  integrators.py    generalised-alpha (lax.scan)
  legacy.py         GEBTAero .dat reader
  plotting.py       V-g plots, 3D wing shapes
```

See [`docs/MAPPING.md`](docs/MAPPING.md) for a line-by-line mapping of both original projects onto this package,
including what differs and what is not ported.

## Conventions

Frame *a* (aircraft): `x_a` forward, `y_a` along the right wing, `z_a` down. Cross-section frame *b*: `x_b` beam
axis, `y_b` towards the leading edge, `z_b = x_b × y_b`. Strains `(γ11, 2γ12, 2γ13, κ1, κ2, κ3)`.
`a` is Theodorsen's reference-axis position in semi-chords (positive aft); `nu` the CG offset (positive aft).
Root reactions follow GEBT's sign convention (`gx.lift_force` returns the net upward load). Eigenvalue tables
use GEBTAero's `[Re(s)/2π, Im(s)/2π]`.

## License

GPL-3.0-or-later, like GEBTAero's Python code. See [`NOTICE.md`](NOTICE.md) for the GEBT (W. Yu) and
Aeroelasticity.jl (MIT) attributions.
