"""BoltzDraw runner for the experimental Biotite-edition WaterKit.

Invoked as a subprocess from hydrate.run_biotite_pipeline() using this
package's own venv (Python >=3.12). Reads a JSON config and writes the sampled
waters as ``<out_dir>/water_000001.pdb``.

Config keys (JSON file passed as argv[1]):
    receptor        Amber-typed receptor PDBQT (from wk_prepare_receptor --amber_pdbqt)
    center          [x, y, z] grid box center (A)
    size            [x, y, z] grid box size (A)
    out_dir         output directory for the water frame
    water_model     "tip3p" | "tip5p"        (default "tip3p")
    temperature     sampling temperature K   (default 300.0)
    n_layers        hydration layers         (default 3)
    sw_orientations SW-map orientations      (default 2000)
    n_jobs          parallel workers, -1=all (default -1)
    cache_dir       SW-map cache dir, or null to disable

The SW-map axis-ordering fix (transpose bug on non-cubic boxes) lives in the
vendored source at src/waterkit/core/grid.py.
"""

import json
import sys


def main() -> None:
    cfg = json.load(open(sys.argv[1]))

    from waterkit.api import hydrate
    from waterkit.core.grid import SphericalWaterMap, generate_maps
    from waterkit.core.molecule import Molecule
    from waterkit.utils import prepare_water_map
    from waterkit.utils.constants import DEFAULT_SPACING

    center = tuple(float(c) for c in cfg["center"])
    size = tuple(float(s) for s in cfg["size"])
    water_model = cfg.get("water_model", "tip3p")
    temperature = float(cfg.get("temperature", 300.0))
    n_layers = int(cfg.get("n_layers", 3))
    sw_orientations = int(cfg.get("sw_orientations", 2000))
    n_jobs = int(cfg.get("n_jobs", -1))
    cache_dir = cfg.get("cache_dir") or None

    atom_types = {"tip3p": ["OW", "HW"], "tip5p": ["OT", "HT", "LP"]}[water_model]

    rec = Molecule.from_file(cfg["receptor"])
    ad_map = generate_maps(rec, atom_types, center, size, DEFAULT_SPACING, n_jobs=n_jobs)
    prepare_water_map(ad_map, water_model)
    SphericalWaterMap(
        water_model,
        temperature,
        n_jobs,
        n_orientations=(None if sw_orientations <= 0 else sw_orientations),
        cache_dir=cache_dir,
    ).run(ad_map)

    hydrate(
        rec,
        ad_map,
        cfg["out_dir"],
        temperature=temperature,
        water_model=water_model,
        n_layers=n_layers,
        n_frames=1,
        n_jobs=n_jobs,
    )


if __name__ == "__main__":
    main()
