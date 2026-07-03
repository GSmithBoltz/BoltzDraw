# WaterKit (Biotite edition)

A reimplementation of the **WaterKit** method — grid-based sampling of explicit
water molecules in protein binding sites — rebuilt on a modern, pip-installable
Python stack.

> Original method: Eberhardt, J. & Forli, S. *WaterKit: Thermodynamic Profiling
> of Protein Hydration Sites*, **JCTC** 2023.
> Reference implementation: https://github.com/forlilab/waterkit

## Why this fork

The upstream tool depends on OpenBabel and a **special, hand-compiled AutoGrid
binary** (patched to support 56 atom types). This port removes both:

| Concern | Upstream | This package |
| --- | --- | --- |
| Structure I/O, bonds, hydrogens | OpenBabel | **Biotite** + **hydride** |
| SMARTS perception (HB / rotatable) | OpenBabel | **RDKit** |
| Energy grids | external AutoGrid binary | **pure-Python AutoDock4** force field + grid generator |
| Packaging / CLI | setup.py + scripts | hatchling + **Typer**/**Rich** |

The sampling algorithm itself (Boltzmann placement order, position/orientation
optimization against the grids, incremental map updates, TIP3P/TIP5P explicit
water construction) is ported faithfully.

## Status

🟢 **Implemented and runnable.** Every module is ported (geometry/Boltzmann
utils, Biotite-backed `Molecule` + I/O, RDKit SMARTS perception, the pure-Python
AutoDock4 force field + grid generator, the TIP3P/TIP5P water model, the
`WaterBox`/`WaterSampler` engine, `hydrate()` and the CLI). `ruff check` is clean,
the test suite passes, and an end-to-end hydration of a small test receptor
places waters at its hydrogen-bond anchors and writes a trajectory PDB.

Known follow-ups before large-scale production use:

- **RDKit bond perception.** `Molecule.to_rdkit()` builds bonds by proximity
  (`MolFromPDBBlock(..., proximityBonding=True)`), so explicit double bonds /
  aromaticity are not always recovered; some waterfield SMARTS (e.g. carbonyl
  acceptors) under-match. Feeding correct bond orders would close this gap.
- **Receptor atom typing.** The bundled `AD4_parameters.dat` is keyed by Amber
  ff14SB atom types, so `run` expects an Amber-typed receptor PDBQT (as upstream
  produced with AmberTools). `prepare-receptor` is a pragmatic Biotite/hydride
  stand-in, not full AmberTools parity.
- **Water self-interaction maps.** Upstream shipped these as large precomputed
  grids; here they are regenerated in pure Python and cached. The first
  uncached build of the spherical-water (`SW`) map is the main cost — worth
  precomputing/vectorizing for large boxes.
- **`hydrate(n_jobs>1)`** uses `multiprocessing`; callers must guard entry with
  `if __name__ == "__main__":` (standard on spawn-based platforms).

## Layout

```
src/waterkit/
├── api.py            # hydrate(): high-level orchestration
├── cli.py            # `waterkit run` / `waterkit prepare-receptor`
├── core/
│   ├── molecule.py   # Biotite-backed Molecule
│   ├── water.py      # spherical / TIP3P / TIP5P water model
│   ├── water_box.py  # receptor + hydration shells, KD-tree
│   ├── sampling.py   # WaterSampler (the algorithmic heart)
│   ├── forcefield.py # AutoDock4 scoring (pure numpy)
│   ├── grid.py       # Map + pure-Python AutoGrid replacement + SphericalWaterMap
│   └── types.py
├── perception/       # RDKit SMARTS: hydrogen_bonds.py, rotatable_bonds.py
├── analysis/         # hydration_sites.py, gist.py
├── utils/            # geometry.py, boltzmann.py, io.py, constants.py, exceptions.py
└── data/             # waterfield.par, disordered_hydrogens.par, AD4 params, water models
```

## Development

```bash
just sync     # uv sync --all-extras
just test     # pytest
just all      # types + lint + format + docstring coverage + tests
```
