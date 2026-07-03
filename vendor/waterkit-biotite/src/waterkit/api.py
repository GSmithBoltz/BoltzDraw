"""High-level API for water sampling.

Port of ``waterkit.waterkit.WaterKit`` as a functional/class entry point in
pocketeer style. Orchestrates: build a :class:`~waterkit.core.water_box.WaterBox`
from a receptor + grids, then for each requested frame add hydration shells and
write the sampled waters. Frames are independent and run in parallel.

The primary public entry point is :func:`hydrate`. :class:`WaterKit` is a thin
class wrapper kept for parity with the original API; it simply delegates to
:func:`hydrate`.
"""

from __future__ import annotations

import gc
import logging
import multiprocessing as mp
import os
from pathlib import Path

from tqdm import tqdm

from .core import Map, Molecule, WaterBox
from .utils import is_writable, split_list_in_chunks

logger = logging.getLogger("waterkit.api")


def _hydrate_single(
    water_box: WaterBox,
    n_layers: int = 0,
    start: int = 0,
    stop: int = 1,
    output_dir: str | Path = ".",
    position: int = 0,
) -> None:
    """Run a single hydration job over a contiguous range of frames.

    This is the multiprocessing worker. It is a module-level function so it can
    be pickled and dispatched to a :class:`multiprocessing.Process`. Each frame
    gets its own deep copy of ``water_box`` (sampling mutates the box in place),
    and the resulting waters are written to ``water_<frame>.pdb``.

    Args:
        water_box: The initialized water box to copy and sample from.
        n_layers: Number of hydration shells to build per frame (capped at 26).
        start: First frame index handled by this job (inclusive).
        stop: Last frame index handled by this job (inclusive).
        output_dir: Directory to write the per-frame PDB files into.
        position: Row used to position this job's tqdm progress bar.
    """
    progress = tqdm(
        total=stop - start,
        position=position,
        desc="job %02d" % (position + 1),
        bar_format="{l_bar}{bar:50}{r_bar}{bar:-10b}",
    )

    for frame_id in range(start, stop + 1):
        i = 1

        # Create a copy of the waterbox, because otherwise it won't work...
        w_copy = water_box.copy()

        while True:
            # build_next_shell returns True if it was able to place water
            # molecules, otherwise it returns False and we break.
            if w_copy.build_next_shell():
                pass
            else:
                break

            # Stop after building n_layers or when the number of layers is equal
            # to 26. In the PDB format, the segid is encoded with only one
            # character, so the maximum number of layers is 26 (the letter Z).
            if i in (n_layers, 26):
                break

            i += 1

        progress.update(1)

        output_filename = os.path.join(str(output_dir), "water_%06d.pdb" % (frame_id + 1))
        w_copy.to_pdb(output_filename, include_receptor=False)

        # Force Python to release the old box, otherwise we are going to have a
        # bad time (memory blows up across frames).
        del w_copy
        gc.collect()

    progress.close()


def hydrate(
    receptor: Molecule,
    ad_map: Map,
    output_dir: str | Path = ".",
    *,
    temperature: float = 300.0,
    water_model: str = "tip3p",
    spherical_water_map: str | None = None,
    n_layers: int = 1,
    n_frames: int = 1,
    n_jobs: int = 1,
) -> None:
    """Hydrate a receptor by adding successive water shells, writing one PDB/frame.

    The receptor is hydrated by adding successive layers of water molecules until
    the box is completely full (or ``n_layers``/26 layers have been built). Each
    of the ``n_frames`` replicas is independent and the work is spread across
    ``n_jobs`` worker processes, each handling a contiguous chunk of frames.

    Args:
        receptor: Receptor molecule (with AutoDock types, charges, HB anchors).
        ad_map: AutoDock affinity grids for the receptor (incl. the SW map).
        output_dir: Directory to write ``water_000001.pdb`` ... frames into.
        temperature: Sampling temperature in Kelvin.
        water_model: ``"tip3p"`` or ``"tip5p"``.
        spherical_water_map: Optional path to a precomputed spherical-water map.
        n_layers: Number of hydration shells to build per frame (max 26).
        n_frames: Number of independent replicas (frames) to generate.
        n_jobs: Parallel worker processes (-1 = all cores).
    """
    is_writable(output_dir)

    if n_jobs == -1:
        n_jobs = mp.cpu_count()
    # Never spawn more workers than there are frames: otherwise
    # ``split_list_in_chunks`` would produce empty chunks (one per surplus worker).
    n_jobs = max(1, min(n_jobs, n_frames))

    chunks = split_list_in_chunks(n_frames, n_jobs)

    logger.info("Hydrating %d frame(s) across %d job(s) into %s", n_frames, n_jobs, output_dir)

    # Initialize a box once in the parent (this builds the affinity / spherical
    # water maps, which can take some time).
    w = WaterBox(receptor, ad_map, temperature, water_model, spherical_water_map)

    # Serial fast path: no worker processes at all.
    if n_jobs == 1:
        for i, chunk in enumerate(chunks):
            _hydrate_single(w, n_layers, chunk[0], chunk[1], output_dir, i)
        return

    # Prefer fork: workers inherit the (large) water box via copy-on-write, so it
    # is never pickled and the package is not re-imported per worker (spawn would
    # do both, dwarfing the actual sampling). Fall back to the default context
    # (e.g. spawn on Windows), which requires an ``if __name__ == "__main__"``
    # guard in the caller.
    proc_ctx = mp.get_context("fork") if "fork" in mp.get_all_start_methods() else mp
    jobs: list[mp.Process] = []
    for i, chunk in enumerate(chunks):
        job = proc_ctx.Process(
            target=_hydrate_single,
            args=(w, n_layers, chunk[0], chunk[1], output_dir, i),
        )
        job.start()
        jobs.append(job)

    for job in jobs:
        job.join()


class WaterKit:
    """Thin wrapper around :func:`hydrate` kept for parity with the original API."""

    def __init__(
        self,
        temperature: float = 300.0,
        water_model: str = "tip3p",
        spherical_water_map: str | None = None,
        n_layer: int = 1,
        n_frames: int = 1,
        n_jobs: int = 1,
    ) -> None:
        """Initialize WaterKit.

        Args:
            temperature: Temperature in Kelvin, only used for Boltzmann sampling.
            water_model: Model used for the water molecule, ``"tip3p"`` or ``"tip5p"``.
            spherical_water_map: AutoDock map file for the water spherical model.
            n_layer: Number of hydration layers to add.
            n_frames: Number of replicas (frames) to generate.
            n_jobs: Number of parallel processes (-1 = all cores).
        """
        self._water_model = water_model
        self._spherical_water_map = spherical_water_map
        self._temperature = temperature
        self._n_layer = n_layer
        self._n_frames = n_frames

        if n_jobs == -1:
            self._n_jobs = mp.cpu_count()
        else:
            self._n_jobs = n_jobs

    def hydrate(self, receptor: Molecule, ad_map: Map, output_dir: str | Path = ".") -> None:
        """Hydrate the molecule with water molecules.

        Args:
            receptor: Receptor of the protein.
            ad_map: AutoDock map of the receptor.
            output_dir: Output directory for the trajectory.
        """
        hydrate(
            receptor,
            ad_map,
            output_dir,
            temperature=self._temperature,
            water_model=self._water_model,
            spherical_water_map=self._spherical_water_map,
            n_layers=self._n_layer,
            n_frames=self._n_frames,
            n_jobs=self._n_jobs,
        )
