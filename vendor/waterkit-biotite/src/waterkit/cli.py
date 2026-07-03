"""Command-line interface for waterkit.

Two commands, modelled on the upstream WaterKit scripts:

* ``run`` -- the ``scripts/run_waterkit.py`` pipeline, reimplemented end-to-end
  in pure Python (no AutoGrid binary, no Vina). It loads a receptor PDBQT,
  builds the AutoDock4 affinity grids, bakes the water electrostatics, computes
  the spherical-water (``SW``) map and hands everything to
  :func:`waterkit.api.hydrate`.
* ``prepare-receptor`` -- a pragmatic, Biotite/hydride-based stand-in for
  ``scripts/wk_prepare_receptor.py``. Full AmberTools (pdb4amber/tleap) parity is
  intentionally out of scope; see :func:`prepare_receptor`.
"""

from __future__ import annotations

import time
from pathlib import Path

import typer
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich.table import Table

app = typer.Typer(
    name="waterkit",
    help="Grid-based explicit-water sampling in protein binding sites, on Biotite.",
    add_completion=False,
    rich_markup_mode="rich",
)
console = Console()

# Supported water models.
_WATER_MODELS = ("tip3p", "tip5p")

# Explicit (non-``SW``) water atom types whose receptor affinity maps must be
# built before ``prepare_water_map`` / ``SphericalWaterMap`` can run. ``SW`` is
# deliberately excluded: it is added afterwards by ``SphericalWaterMap.run``.
_WATER_ATOM_TYPES: dict[str, list[str]] = {
    "tip3p": ["OW", "HW"],
    "tip5p": ["OT", "HT", "LP"],
}

# Input formats accepted by ``prepare-receptor`` (handled by the Biotite loader).
_PREPARE_FORMATS = (".pdb", ".cif", ".mmcif", ".bcif", ".pdbqt")


def _validate_input_file(structure_file: str) -> Path:
    """Validate that an input structure file exists.

    Args:
        structure_file: Path to the input structure.

    Returns:
        The resolved :class:`~pathlib.Path`.

    Raises:
        typer.Exit: If the file does not exist.
    """
    path = Path(structure_file)
    if not path.exists():
        console.print(f"[bold red]Error: file '{structure_file}' does not exist[/bold red]")
        raise typer.Exit(1)
    return path


def _spinner(description: str) -> Progress:
    """Return a transient Rich spinner with a single running task description."""
    progress = Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console,
        transient=True,
    )
    progress.add_task(description, total=None)
    return progress


@app.command()
def run(
    receptor: str = typer.Argument(
        ..., help="Receptor file (PDBQT with AutoDock atom types and partial charges)"
    ),
    center: tuple[float, float, float] = typer.Option(
        ..., "--center", "-c", help="Grid box center X Y Z (Å)"
    ),
    size: tuple[float, float, float] = typer.Option(
        ..., "--size", "-s", help="Grid box size X Y Z (Å)"
    ),
    out_dir: str = typer.Option("traj", "--out-dir", "-o", help="Output directory for frames"),
    n_frames: int = typer.Option(1, "--n-frames", "-n", help="Number of frames to sample"),
    n_layers: int = typer.Option(3, "--n-layers", "-l", help="Hydration layers per frame"),
    water_model: str = typer.Option("tip3p", "--water-model", "-w", help="tip3p or tip5p"),
    temperature: float = typer.Option(300.0, "--temperature", "-t", help="Temperature (K)"),
    n_jobs: int = typer.Option(-1, "--jobs", "-j", help="Parallel workers (-1 = all cores)"),
    sw_orientations: int = typer.Option(
        2000,
        "--sw-orientations",
        help="Water orientations for the spherical-water map (0 = all 9897, exact). "
        "Fewer is faster with minimal accuracy loss.",
    ),
    cache_dir: str = typer.Option(
        ".waterkit_cache",
        "--cache-dir",
        help="Directory for the spherical-water map cache; re-runs of the same "
        "receptor/box reuse it. Pass an empty string to disable caching.",
    ),
) -> None:
    """Sample water-molecule placements in a receptor binding site.

    Reimplements the ``run_waterkit.py`` pipeline without any external binary:
    read the receptor, evaluate the AutoDock4 force field on the grid in pure
    Python, bake the TIP3P/TIP5P electrostatics into the water maps, build the
    Boltzmann-averaged spherical-water (``SW``) map and then grow hydration
    shells with :func:`waterkit.api.hydrate`.

    Args:
        receptor: Receptor PDBQT (AutoDock atom types + partial charges).
        center: Grid box center (X, Y, Z) in Angstrom.
        size: Grid box size (X, Y, Z) in Angstrom.
        out_dir: Directory the per-frame water PDBs are written to.
        n_frames: Number of independent replicas (frames) to generate.
        n_layers: Number of hydration layers to add per frame.
        water_model: ``"tip3p"`` or ``"tip5p"``.
        temperature: Sampling temperature in Kelvin.
        n_jobs: Parallel worker processes (``-1`` uses all cores).
        sw_orientations: Number of water orientations used to build the
            spherical-water map (``0`` uses all 9897 for an exact map). The
            spherical-water map dominates runtime, so subsampling here is the
            main speed/accuracy knob.
        cache_dir: Directory for the spherical-water map cache (empty string
            disables it). Subsequent runs on the same receptor/box/settings load
            the map from disk instead of recomputing it.
    """
    water_model = water_model.lower()
    if water_model not in _WATER_MODELS:
        console.print(
            f"[bold red]Error: unknown water model '{water_model}'. "
            f"Choose one of: {', '.join(_WATER_MODELS)}[/bold red]"
        )
        raise typer.Exit(1)
    _validate_input_file(receptor)

    # Heavy imports are deferred so ``--help`` stays cheap.
    from .api import hydrate
    from .core.grid import SphericalWaterMap, generate_maps
    from .core.molecule import Molecule
    from .utils import prepare_water_map
    from .utils.constants import DEFAULT_SPACING
    from .utils.exceptions import WaterKitError

    center = tuple(float(c) for c in center)
    size = tuple(float(s) for s in size)
    atom_types = _WATER_ATOM_TYPES[water_model]
    out_path = Path(out_dir)

    start_time = time.time()

    try:
        with _spinner("Reading receptor..."):
            rec = Molecule.from_file(receptor)
        n_atoms = int(rec.atoms.size)
        console.print(f"Loaded receptor with [cyan]{n_atoms}[/cyan] atoms")

        with _spinner(f"Generating AD4 maps ({', '.join(atom_types)})..."):
            ad_map = generate_maps(rec, atom_types, center, size, DEFAULT_SPACING, n_jobs=n_jobs)

        with _spinner("Baking water-map electrostatics..."):
            prepare_water_map(ad_map, water_model)

        with _spinner("Building spherical-water (SW) map..."):
            n_orientations = None if sw_orientations <= 0 else sw_orientations
            sw_cache = cache_dir if cache_dir else None
            SphericalWaterMap(
                water_model,
                temperature,
                n_jobs,
                n_orientations=n_orientations,
                cache_dir=sw_cache,
            ).run(ad_map)

        out_path.mkdir(parents=True, exist_ok=True)

        with _spinner(f"Hydrating ({n_frames} frame(s), up to {n_layers} layer(s))..."):
            hydrate(
                rec,
                ad_map,
                out_path,
                temperature=temperature,
                water_model=water_model,
                n_layers=n_layers,
                n_frames=n_frames,
                n_jobs=n_jobs,
            )
    except (WaterKitError, KeyError, ValueError) as exc:
        console.print(f"[bold red]Error: {exc}[/bold red]")
        raise typer.Exit(1) from exc

    elapsed = time.time() - start_time
    n_written = len(sorted(out_path.glob("*.pdb")))
    map_labels = " ".join(ad_map.info()["maps"])

    console.print(f"\n[bold green]Hydration complete in {elapsed:.2f} seconds[/bold green]\n")

    table = Table(
        title="WaterKit run",
        show_header=True,
        header_style="bold magenta",
        border_style="blue",
    )
    table.add_column("Parameter", style="cyan")
    table.add_column("Value", style="green")
    table.add_row("Receptor", receptor)
    table.add_row("Receptor atoms", str(n_atoms))
    table.add_row("Water model", water_model)
    table.add_row("Box center (Å)", " ".join(f"{c:g}" for c in center))
    table.add_row("Box size (Å)", " ".join(f"{s:g}" for s in size))
    table.add_row("Grid spacing (Å)", f"{DEFAULT_SPACING:g}")
    table.add_row("Affinity maps", map_labels)
    sw_label = "9897 (exact)" if sw_orientations <= 0 else str(sw_orientations)
    table.add_row("SW orientations", sw_label)
    table.add_row("SW cache", cache_dir or "disabled")
    table.add_row("Temperature (K)", f"{temperature:g}")
    table.add_row("Frames", str(n_frames))
    table.add_row("Layers / frame", str(n_layers))
    table.add_row("Parallel jobs", str(n_jobs))
    table.add_row("PDB frames written", str(n_written))
    console.print(table)
    console.print(f"\n[cyan]Results written to {out_dir}/[/cyan]")


@app.command()
def prepare_receptor(
    structure: str = typer.Argument(..., help="Input protein structure (PDB/mmCIF)"),
    out: str = typer.Option("receptor", "--out", "-o", help="Output basename"),
    keep_hydrogens: bool = typer.Option(False, "--keep-hydrogens", help="Keep input hydrogens"),
) -> None:
    """Protonate and AutoDock-type a receptor (Biotite + hydride), writing PDBQT.

    A pragmatic stand-in for ``wk_prepare_receptor.py``. It loads the structure
    with Biotite (optionally re-adding hydrogens with ``hydride``), assigns
    best-effort AutoDock4 atom types, and writes ``<out>.pdbqt`` and
    ``<out>.pdb``.

    Note:
        Full AmberTools parity (pdb4amber/tleap protonation-state assignment,
        disulfide detection, residue renumbering and ff14SB partial charges) is
        out of scope. In particular, partial charges are not computed here and
        are written as zero; for production runs prepare the receptor with the
        original AmberTools-based tool.

    Args:
        structure: Input PDB or mmCIF/BinaryCIF structure file.
        out: Output basename (writes ``<out>.pdbqt`` and ``<out>.pdb``).
        keep_hydrogens: If set, keep the input hydrogens instead of re-adding
            them with ``hydride``.
    """
    path = _validate_input_file(structure)
    if path.suffix.lower() not in _PREPARE_FORMATS:
        console.print(
            f"[bold red]Error: unsupported format '{path.suffix}'. "
            f"Supported: {', '.join(_PREPARE_FORMATS)}[/bold red]"
        )
        raise typer.Exit(1)

    # Heavy imports are deferred so ``--help`` stays cheap.
    import numpy as np
    from biotite.structure.io import pdb

    from .utils.io import assign_autodock_types, load_structure, write_pdbqt

    start_time = time.time()

    try:
        with _spinner("Reading structure..."):
            atoms = load_structure(structure, add_hydrogens=not keep_hydrogens)
        n_atoms = atoms.array_length()
        console.print(f"Loaded structure with [cyan]{n_atoms}[/cyan] atoms")

        with _spinner("Assigning AutoDock atom types..."):
            atom_types = assign_autodock_types(atoms)
    except Exception as exc:
        console.print(f"[bold red]Error: {exc}[/bold red]")
        raise typer.Exit(1) from exc

    out_path = Path(out)
    parent = out_path.parent
    if str(parent) not in ("", "."):
        parent.mkdir(parents=True, exist_ok=True)
    pdbqt_file = f"{out}.pdbqt"
    pdb_file = f"{out}.pdb"

    charges = getattr(atoms, "charge", None)
    if charges is None:
        charges = np.zeros(n_atoms)

    console.print("Writing output files...")
    write_pdbqt(
        pdbqt_file,
        atoms.atom_name,
        atoms.res_name,
        atoms.res_id,
        atoms.chain_id,
        atoms.coord,
        charges,
        atom_types,
    )
    pdb_out = pdb.PDBFile()
    pdb_out.set_structure(atoms)
    pdb_out.write(pdb_file)

    elapsed = time.time() - start_time
    n_hydrogens = int(np.sum(np.char.upper(atoms.element.astype("U2")) == "H"))

    console.print(f"\n[bold green]Receptor prepared in {elapsed:.2f} seconds[/bold green]\n")

    table = Table(
        title="Prepared receptor",
        show_header=True,
        header_style="bold magenta",
        border_style="blue",
    )
    table.add_column("Field", style="cyan")
    table.add_column("Value", style="green")
    table.add_row("Input", structure)
    table.add_row("Atoms", str(n_atoms))
    table.add_row("Hydrogens", str(n_hydrogens))
    table.add_row("Hydrogens re-added", "no" if keep_hydrogens else "yes")
    table.add_row("PDBQT", pdbqt_file)
    table.add_row("PDB", pdb_file)
    console.print(table)


def main() -> None:
    """Entry point for the CLI."""
    app()


if __name__ == "__main__":
    main()
