"""
BoltzStar hydration pipeline.

Two modes
---------
quick  — Geometric bridging water placement (~3 s)
           PDBFixer prep → find protein–ligand polar pairs at bridge
           distance (4–6.2 Å) → place water O at 2.8 Å → clash filter.
           Fast, binding-site-specific, no external binaries beyond PDBFixer.

full   — WaterKit energy-scored placement (~40 s)
           PDBFixer prep → PDBQT conversion (OpenBabel + meeko) →
           AutoGrid4 energy maps → WaterKit TIP3P 3-shell hydration →
           ligand clash filter.
           Slower, physically rigorous, covers full protein surface.

Both return (pdb_string, water_molecule_count).
"""

import glob
import io
import os
import shutil
import subprocess
import tempfile

import numpy as np

# ----------------------------------------------------------------
# Polymer residue names (used to split protein from ligand)
# ----------------------------------------------------------------
_POLYMER = {
    'ALA', 'ARG', 'ASN', 'ASP', 'CYS', 'GLN', 'GLU', 'GLY', 'HIS', 'ILE',
    'LEU', 'LYS', 'MET', 'PHE', 'PRO', 'SER', 'THR', 'TRP', 'TYR', 'VAL',
    'HID', 'HIE', 'HIP', 'CYX', 'ACE', 'NME',
    # nucleotides
    'A', 'T', 'G', 'C', 'U', 'DA', 'DT', 'DG', 'DC', 'DU',
}

# Derive paths from the running Python interpreter so this works on
# any machine regardless of username or Miniconda install location.
import sys as _sys
_ENV        = os.path.dirname(os.path.dirname(_sys.executable))
_AUTOGRID4  = os.path.join(_ENV, 'bin', 'autogrid4')
_OBABEL     = os.path.join(_ENV, 'bin', 'obabel')
_PARAM_FILE = os.path.join(
    _ENV,
    'lib', 'python3.11', 'site-packages',
    'waterkit', 'data', 'water', 'tip3p', 'raw_data', 'AD4_parameters.dat'
)

# ----------------------------------------------------------------
# Experimental: Biotite-edition WaterKit (vendored, own venv).
# Runs in a separate Python (>=3.12) via subprocess because the server
# env is 3.11. Receptor typing still needs AmberTools (wk_prepare_receptor
# --amber_pdbqt), so we reuse the binaries already in this env.
# ----------------------------------------------------------------
_BIOTITE_DIR    = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                               'vendor', 'waterkit-biotite')
_BIOTITE_PY     = os.path.join(_BIOTITE_DIR, '.venv', 'bin', 'python')
_BIOTITE_RUNNER = os.path.join(_BIOTITE_DIR, 'boltz_runner.py')
_BIOTITE_CACHE  = os.path.join(_BIOTITE_DIR, '.sw_cache')
_WK_PREPARE     = os.path.join(_ENV, 'bin', 'wk_prepare_receptor.py')
_BIOTITE_PAD    = 20.0   # Å padding around the ligand extent (matches Full)


# ----------------------------------------------------------------
# Public entry point
# ----------------------------------------------------------------

def run_pipeline(input_data: str, input_format: str, mode: str = 'full'):
    """
    Run the hydration pipeline.

    Parameters
    ----------
    input_data   : CIF or PDB file content as a UTF-8 string
    input_format : 'mmcif' or 'pdb'
    mode         : 'quick' (geometric) | 'full' (WaterKit) |
                   'biotite' (experimental Biotite-edition WaterKit)

    Returns
    -------
    (pdb_string, water_molecule_count)
    """
    if mode == 'quick':
        return run_quick_pipeline(input_data, input_format)
    if mode == 'biotite':
        return run_biotite_pipeline(input_data, input_format)
    return run_waterkit_pipeline(input_data, input_format)


# ----------------------------------------------------------------
# Quick pipeline — geometric bridging water placement
# ----------------------------------------------------------------

# H-bond geometry constants
_HBOND_ELEMENTS = {'N', 'O', 'S', 'F'}
_WATER_DIST     = 2.80   # Å — ideal heavy-atom … O(water)
_BRIDGE_MIN     = 4.0    # Å — min protein–ligand dist to bridge
_BRIDGE_MAX     = 6.2    # Å — max protein–ligand dist to bridge
_SITE_RADIUS    = 6.0    # Å — binding-site definition
_CLASH_HEAVY    = 1.80   # Å — min water-O … heavy-atom
_CLASH_WATER    = 2.20   # Å — min water-O … water-O
_MAX_WATERS_Q   = 80     # hard cap


def run_quick_pipeline(input_data: str, input_format: str):
    """
    Geometric bridging water placement.
    PDBFixer → find polar protein–ligand pairs at 4–6.2 Å →
    place water O at 2.8 Å toward ligand → clash filter.
    """
    from pdbfixer import PDBFixer
    from openmm.app import PDBFile
    from openmm.unit import angstrom

    suffix = '.cif' if input_format == 'mmcif' else '.pdb'
    workdir = tempfile.mkdtemp(prefix='boltzstar_quick_')
    try:
        input_path = os.path.join(workdir, 'input' + suffix)
        with open(input_path, 'w', encoding='utf-8') as fh:
            fh.write(input_data)

        fixer = PDBFixer(filename=input_path)
        fixer.findMissingResidues()
        fixer.findNonstandardResidues()
        fixer.replaceNonstandardResidues()
        fixer.findMissingAtoms()
        fixer.addMissingAtoms()
        fixer.addMissingHydrogens(7.4)
    finally:
        shutil.rmtree(workdir, ignore_errors=True)

    topology  = fixer.topology
    positions = fixer.positions

    try:
        pos_ang = np.array(positions.value_in_unit(angstrom))
    except Exception:
        pos_ang = np.array([[p.x * 10, p.y * 10, p.z * 10]
                            for p in positions])

    water_coords = _place_geometric_waters(topology, pos_ang)

    pdb_buf = io.StringIO()
    PDBFile.writeFile(topology, positions, pdb_buf)
    pdb_str = pdb_buf.getvalue()

    n_waters = len(water_coords)
    if n_waters:
        pdb_str = _append_water_records(pdb_str, water_coords)

    return pdb_str, n_waters


def _place_geometric_waters(topology, pos_ang: np.ndarray) -> np.ndarray:
    """Return Nx3 array of water oxygen positions (Å)."""
    elements   = []
    is_polymer = []
    atom_indices = []

    for atom in topology.atoms():
        if atom.element is None:
            continue
        if atom.element.symbol == 'H':
            continue
        elements.append(atom.element.symbol)
        is_polymer.append(atom.residue.name.upper() in _POLYMER)
        atom_indices.append(atom.index)

    if not atom_indices:
        return np.empty((0, 3))

    coords    = pos_ang[atom_indices]
    poly_mask = np.array(is_polymer, dtype=bool)
    polar     = np.array([e in _HBOND_ELEMENTS for e in elements], dtype=bool)

    lig_mask  = ~poly_mask
    if not lig_mask.any():
        return np.empty((0, 3))

    lig_polar  = coords[lig_mask  & polar]
    prot_polar = coords[poly_mask & polar]

    if not len(lig_polar) or not len(prot_polar):
        return np.empty((0, 3))

    # Distance matrix: protein polar × ligand polar
    diffs    = prot_polar[:, None, :] - lig_polar[None, :, :]
    dist_mat = np.linalg.norm(diffs, axis=2)

    # Only protein atoms within SITE_RADIUS of any ligand atom
    in_site    = dist_mat.min(axis=1) < _SITE_RADIUS
    site_prot  = prot_polar[in_site]
    site_dists = dist_mat[in_site]

    candidates = []
    for i, pp in enumerate(site_prot):
        for j, lp in enumerate(lig_polar):
            d = site_dists[i, j]
            if _BRIDGE_MIN <= d <= _BRIDGE_MAX:
                direction = lp - pp
                direction /= np.linalg.norm(direction)
                wpos = pp + direction * _WATER_DIST
                if np.linalg.norm(wpos - lp) <= _WATER_DIST + 1.0:
                    candidates.append(wpos)

    if not candidates:
        return np.empty((0, 3))

    kept = _clash_filter_q(np.array(candidates), coords)
    return kept[:_MAX_WATERS_Q]


def _clash_filter_q(candidates: np.ndarray,
                    existing: np.ndarray) -> np.ndarray:
    kept = []
    for wpos in candidates:
        if np.linalg.norm(existing - wpos, axis=1).min() < _CLASH_HEAVY:
            continue
        if kept:
            if np.linalg.norm(np.array(kept) - wpos, axis=1).min() < _CLASH_WATER:
                continue
        kept.append(wpos)
    return np.array(kept) if kept else np.empty((0, 3))


def _append_water_records(pdb_str: str, water_coords: np.ndarray) -> str:
    """Append HETATM HOH oxygen records to a PDB string before END."""
    lines = pdb_str.splitlines(keepends=True)

    last_serial = 0
    last_resseq = 0
    for line in lines:
        if line[:6] in ('ATOM  ', 'HETATM'):
            try:
                last_serial = max(last_serial, int(line[6:11]))
                last_resseq = max(last_resseq, int(line[22:26]))
            except ValueError:
                pass

    water_lines = []
    for i, (x, y, z) in enumerate(water_coords):
        serial = (last_serial + i + 1) % 100000
        resseq = (last_resseq + i + 1) % 10000
        water_lines.append(
            f'HETATM{serial:5d}  O   HOH W{resseq:4d}    '
            f'{x:8.3f}{y:8.3f}{z:8.3f}  1.00  0.00          '
            f' O  \n'
        )

    insert_at = len(lines)
    for i, line in enumerate(lines):
        if line[:3] in ('END', 'CON'):
            insert_at = i
            break

    return ''.join(lines[:insert_at] + water_lines + lines[insert_at:])


# ----------------------------------------------------------------
# Full pipeline — WaterKit energy-scored placement
# ----------------------------------------------------------------

def run_waterkit_pipeline(input_data: str, input_format: str):
    """Original WaterKit pipeline (renamed from run_pipeline)."""
    workdir = tempfile.mkdtemp(prefix='boltzstar_full_')
    try:
        return _run_waterkit(input_data, input_format, workdir)
    finally:
        shutil.rmtree(workdir, ignore_errors=True)


# ----------------------------------------------------------------
# Experimental — Biotite-edition WaterKit
# ----------------------------------------------------------------

def run_biotite_pipeline(input_data: str, input_format: str):
    """Experimental hydration via the Biotite-edition WaterKit rewrite.

    Same shape as the Full pipeline, but the energy-grid + water-sampling
    engine is the pip-installable Biotite port (no OpenBabel, no patched
    AutoGrid binary). It runs in its own vendored venv via subprocess.

    Receptor atom typing still needs Amber ff14SB types, so we prepare the
    receptor with AmberTools (wk_prepare_receptor --amber_pdbqt) — the engine's
    own prepare-receptor emits incompatible AutoDock types. The SW-map
    transpose bug (non-cubic boxes) is patched in the vendored source.
    """
    if not os.path.exists(_BIOTITE_PY):
        raise RuntimeError(
            'Biotite WaterKit is not set up. Expected a venv at '
            f'{_BIOTITE_PY}. Build it with:\n'
            f'  cd {_BIOTITE_DIR} && uv venv --python 3.13 .venv && '
            f'uv pip install --python .venv/bin/python -e .'
        )

    workdir = tempfile.mkdtemp(prefix='boltzstar_biotite_')
    try:
        return _run_biotite(input_data, input_format, workdir)
    finally:
        shutil.rmtree(workdir, ignore_errors=True)


def _run_biotite(input_data, input_format, workdir):
    import json
    from pdbfixer import PDBFixer
    from openmm.app import PDBFile

    # Stage 1: PDBFixer (same prep as Full)
    suffix = '.cif' if input_format == 'mmcif' else '.pdb'
    input_path = os.path.join(workdir, 'input' + suffix)
    with open(input_path, 'w', encoding='utf-8') as fh:
        fh.write(input_data)

    fixer = PDBFixer(filename=input_path)
    fixer.findMissingResidues()
    fixer.findNonstandardResidues()
    fixer.replaceNonstandardResidues()
    fixer.findMissingAtoms()
    fixer.addMissingAtoms()
    fixer.addMissingHydrogens(7.4)

    fixed_pdb = os.path.join(workdir, 'fixed.pdb')
    with open(fixed_pdb, 'w') as fh:
        PDBFile.writeFile(fixer.topology, fixer.positions, fh)

    # Stage 2: split protein / ligand, derive grid box from the ligand
    protein_pdb, ligand_pdb, lig_centroid, lig_extent = _split_protein_ligand(
        fixed_pdb, workdir
    )
    box_size = lig_extent + _BIOTITE_PAD

    # Stage 3: Amber-typed receptor via AmberTools (works around the engine's
    # incompatible prepare-receptor). tleap lives in this env's bin.
    receptor_pdbqt = _amber_receptor(protein_pdb, workdir)

    # Stage 4: run the Biotite engine in its own venv
    out_dir = os.path.join(workdir, 'waters')
    os.makedirs(out_dir, exist_ok=True)
    config = {
        'receptor':        receptor_pdbqt,
        'center':          lig_centroid.tolist(),
        'size':            box_size.tolist(),
        'out_dir':         out_dir,
        'water_model':     'tip3p',
        'temperature':     300.0,
        'n_layers':        3,
        'sw_orientations': 2000,
        'n_jobs':          -1,
        'cache_dir':       _BIOTITE_CACHE,
    }
    config_path = os.path.join(workdir, 'config.json')
    with open(config_path, 'w') as fh:
        json.dump(config, fh)

    result = subprocess.run(
        [_BIOTITE_PY, _BIOTITE_RUNNER, config_path],
        capture_output=True, text=True, cwd=_BIOTITE_DIR,
    )
    water_frame = os.path.join(out_dir, 'water_000001.pdb')
    if result.returncode != 0 or not os.path.exists(water_frame):
        tail = (result.stderr or result.stdout or '')[-1500:]
        raise RuntimeError(f'Biotite WaterKit failed:\n{tail}')

    # Stage 5: merge — clash-filter vs ligand, strip TIP3P hydrogens, append.
    # (Same post-processing as the Full pipeline so the viewer behaves alike.)
    pdb_buf = io.StringIO()
    PDBFile.writeFile(fixer.topology, fixer.positions, pdb_buf)
    fixed_str = pdb_buf.getvalue()

    water_str = open(water_frame).read()
    water_lines = _filter_water_clashes(water_str, ligand_pdb, clash_dist=2.0)
    water_lines = [l for l in water_lines
                   if not ('HOH' in l[17:20]
                           and l[12:16].strip().startswith('H'))]

    n_waters = sum(1 for l in water_lines if '  O   HOH' in l)

    fixed_lines = [l for l in fixed_str.splitlines(keepends=True)
                   if not l.startswith('END')]
    output = ''.join(fixed_lines) + ''.join(water_lines) + 'END\n'
    return output, n_waters


def _amber_receptor(protein_pdb, workdir):
    """Amber ff14SB-typed receptor PDBQT via wk_prepare_receptor (AmberTools)."""
    if not os.path.exists(_WK_PREPARE):
        raise RuntimeError(
            f'wk_prepare_receptor.py not found at {_WK_PREPARE}. '
            'The Biotite engine needs an Amber-typed receptor.'
        )

    prefix = os.path.join(workdir, 'amber_receptor')
    env = dict(os.environ)
    env['PATH']      = os.path.join(_ENV, 'bin') + os.pathsep + env.get('PATH', '')
    env['AMBERHOME'] = _ENV

    result = subprocess.run(
        [_sys.executable, _WK_PREPARE, '-i', protein_pdb, '-o', prefix,
         '--amber_pdbqt', '--ignore_gaps'],
        capture_output=True, text=True, env=env,
    )
    receptor_pdbqt = prefix + '_amber.pdbqt'
    if result.returncode != 0 or not os.path.exists(receptor_pdbqt):
        tail = (result.stderr or result.stdout or '')[-1500:]
        raise RuntimeError(f'AmberTools receptor prep failed:\n{tail}')
    return receptor_pdbqt


# ----------------------------------------------------------------
# WaterKit pipeline stages
# ----------------------------------------------------------------

def _run_waterkit(input_data, input_format, workdir):
    from pdbfixer import PDBFixer
    from openmm.app import PDBFile
    from openmm.unit import angstrom

    # ----------------------------------------------------------
    # Stage 1: PDBFixer
    # ----------------------------------------------------------
    suffix = '.cif' if input_format == 'mmcif' else '.pdb'
    input_path = os.path.join(workdir, 'input' + suffix)
    with open(input_path, 'w', encoding='utf-8') as fh:
        fh.write(input_data)

    fixer = PDBFixer(filename=input_path)
    fixer.findMissingResidues()
    fixer.findNonstandardResidues()
    fixer.replaceNonstandardResidues()
    fixer.findMissingAtoms()
    fixer.addMissingAtoms()
    fixer.addMissingHydrogens(7.4)

    fixed_pdb = os.path.join(workdir, 'fixed.pdb')
    with open(fixed_pdb, 'w') as fh:
        PDBFile.writeFile(fixer.topology, fixer.positions, fh)

    # ----------------------------------------------------------
    # Stage 2: Split protein / ligand and compute ligand centroid
    # ----------------------------------------------------------
    protein_pdb, ligand_pdb, lig_centroid, lig_extent = _split_protein_ligand(
        fixed_pdb, workdir
    )

    # ----------------------------------------------------------
    # Stage 3: Convert to PDBQT
    # ----------------------------------------------------------
    receptor_pdbqt = _protein_to_pdbqt(protein_pdb, workdir)
    ligand_pdbqt   = _ligand_to_pdbqt(ligand_pdb, workdir)

    # ----------------------------------------------------------
    # Stage 4: AutoGrid4
    # ----------------------------------------------------------
    import waterkit
    box_size = lig_extent + 20.0   # 10 Å padding each side

    ag = waterkit.AutoGrid(exec_path=_AUTOGRID4, param_file=_PARAM_FILE)
    ad_map = ag.run(
        receptor_file=receptor_pdbqt,
        atom_types=['OA', 'HD', 'SW', 'OW', 'HW'],
        box_center=lig_centroid.tolist(),
        box_size=box_size.tolist(),
        spacing=0.375,
    )
    if ad_map is None:
        raise RuntimeError('AutoGrid4 failed — check receptor PDBQT and logs')

    # ----------------------------------------------------------
    # Stage 5: WaterKit
    # ----------------------------------------------------------
    from waterkit import WaterKit, Molecule

    receptor_mol = Molecule.from_file(receptor_pdbqt)
    wk = WaterKit(temperature=300.0, water_model='tip3p',
                  n_layer=3, n_frames=1, n_jobs=1)

    water_dir = os.path.join(workdir, 'waters')
    os.makedirs(water_dir, exist_ok=True)
    wk.hydrate(receptor_mol, ad_map, output_dir=water_dir)

    water_frame = os.path.join(water_dir, 'water_000001.pdb')
    if not os.path.exists(water_frame):
        raise RuntimeError('WaterKit produced no output — hydration failed')

    # ----------------------------------------------------------
    # Stage 6: Merge fixed structure + waters into one PDB
    # ----------------------------------------------------------
    pdb_buf = io.StringIO()
    PDBFile.writeFile(fixer.topology, fixer.positions, pdb_buf)
    fixed_str = pdb_buf.getvalue()

    water_str = open(water_frame).read()

    # ----------------------------------------------------------
    # Clash-filter: remove waters that overlap the ligand.
    # WaterKit places waters on the receptor surface without
    # knowing about the ligand, so a handful end up inside it.
    # ----------------------------------------------------------
    water_lines = _filter_water_clashes(
        water_str, ligand_pdb, clash_dist=2.0
    )

    # ----------------------------------------------------------
    # Strip TIP3P hydrogens from water records.
    # Molstar switches to strict D-H···A angle checking when
    # explicit H atoms are present — TIP3P H positions (optimised
    # for force-field energy) fail that check and suppress the
    # dashed H-bond lines.  The O positions are the chemically
    # meaningful WaterKit result; H atoms are modelling artefacts.
    # ----------------------------------------------------------
    water_lines = [l for l in water_lines
                   if not ('HOH' in l[17:20]
                           and l[12:16].strip().startswith('H'))]

    n_waters = sum(1 for l in water_lines if '  O   HOH' in l)

    # Strip END from fixed, append filtered water records, re-add END
    fixed_lines = [l for l in fixed_str.splitlines(keepends=True)
                   if not l.startswith('END')]

    output = ''.join(fixed_lines) + ''.join(water_lines) + 'END\n'
    return output, n_waters


# ----------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------

def _split_protein_ligand(fixed_pdb, workdir):
    """Split fixed.pdb into protein and ligand PDB files.
    Returns (protein_pdb_path, ligand_pdb_path, lig_centroid, lig_extent).
    """
    protein_lines, ligand_lines = [], []
    lig_coords = []

    with open(fixed_pdb) as fh:
        for line in fh:
            rec = line[:6]
            if rec in ('ATOM  ', 'HETATM', 'TER   '):
                resname = line[17:20].strip()
                if rec == 'ATOM  ' or resname in _POLYMER:
                    protein_lines.append(line)
                elif resname not in ('HOH', 'WAT', 'SOL'):
                    ligand_lines.append(line)
                    try:
                        x = float(line[30:38])
                        y = float(line[38:46])
                        z = float(line[46:54])
                        lig_coords.append([x, y, z])
                    except ValueError:
                        pass
            else:
                protein_lines.append(line)

    if not lig_coords:
        raise RuntimeError(
            'No ligand found in structure. '
            'WaterKit requires a protein-ligand complex.'
        )

    protein_pdb = os.path.join(workdir, 'protein.pdb')
    ligand_pdb  = os.path.join(workdir, 'ligand.pdb')

    with open(protein_pdb, 'w') as fh:
        fh.writelines(protein_lines)
    with open(ligand_pdb, 'w') as fh:
        fh.writelines(ligand_lines)
        fh.write('END\n')

    coords   = np.array(lig_coords)
    centroid = coords.mean(axis=0)
    extent   = coords.max(axis=0) - coords.min(axis=0)

    return protein_pdb, ligand_pdb, centroid, extent


def _protein_to_pdbqt(protein_pdb, workdir):
    """Convert protein PDB → PDBQT with Gasteiger charges via OpenBabel."""
    receptor_pdbqt = os.path.join(workdir, 'receptor.pdbqt')
    result = subprocess.run(
        [_OBABEL, protein_pdb, '-O', receptor_pdbqt, '-xr',
         '--partialcharge', 'gasteiger'],
        capture_output=True, text=True
    )
    if result.returncode != 0 or not os.path.exists(receptor_pdbqt):
        raise RuntimeError(f'OpenBabel receptor conversion failed:\n{result.stderr}')
    return receptor_pdbqt


def _ligand_to_pdbqt(ligand_pdb, workdir):
    """Convert ligand PDB → PDBQT via meeko + RDKit."""
    from rdkit import Chem
    from rdkit.Chem import AllChem
    from meeko import MoleculePreparation, PDBQTWriterLegacy

    ligand_pdbqt = os.path.join(workdir, 'ligand.pdbqt')

    mol = Chem.MolFromPDBFile(ligand_pdb, removeHs=False, sanitize=True)
    if mol is None:
        raise RuntimeError('RDKit could not parse the ligand PDB file')

    mol = Chem.AddHs(mol, addCoords=True)

    preparator  = MoleculePreparation()
    mol_setups  = preparator.prepare(mol)
    pdbqt_str, is_ok, err = PDBQTWriterLegacy.write_string(mol_setups[0])

    if not is_ok:
        raise RuntimeError(f'meeko ligand preparation failed: {err}')

    with open(ligand_pdbqt, 'w') as fh:
        fh.write(pdbqt_str)

    return ligand_pdbqt


def _filter_water_clashes(water_pdb_str: str, ligand_pdb: str,
                          clash_dist: float = 2.0) -> list:
    """
    Remove WaterKit water records whose oxygen atom is within
    clash_dist Å of any ligand heavy atom.

    WaterKit places waters on the receptor surface without knowledge
    of the ligand pose, so a small number always end up overlapping it.

    Returns a list of PDB line strings (keepends=True) for the survivors.
    """
    # Parse ligand heavy-atom coordinates
    lig_coords = []
    with open(ligand_pdb) as fh:
        for line in fh:
            if line[:6] in ('ATOM  ', 'HETATM'):
                try:
                    x = float(line[30:38])
                    y = float(line[38:46])
                    z = float(line[46:54])
                    lig_coords.append([x, y, z])
                except ValueError:
                    pass

    if not lig_coords:
        # No ligand coords found — keep all waters
        return [l for l in water_pdb_str.splitlines(keepends=True)
                if l.startswith('ATOM') or l.startswith('HETATM')]

    lig_arr = np.array(lig_coords)

    # Group water records by residue (each HOH has O + H1 + H2)
    # Index water molecules by their O-atom line
    water_groups = []   # list of (o_xyz, [lines])
    current_group = []
    current_o = None

    for line in water_pdb_str.splitlines(keepends=True):
        if not (line.startswith('ATOM') or line.startswith('HETATM')):
            continue
        resname = line[17:20].strip()
        if resname != 'HOH':
            continue
        atom_name = line[12:16].strip()
        if atom_name == 'O':
            if current_group:
                water_groups.append((current_o, current_group))
            current_o = np.array([float(line[30:38]),
                                   float(line[38:46]),
                                   float(line[46:54])])
            current_group = [line]
        else:
            current_group.append(line)

    if current_group:
        water_groups.append((current_o, current_group))

    # Keep waters whose O is > clash_dist from every ligand atom
    kept_lines = []
    n_clashing = 0
    for o_xyz, lines in water_groups:
        dists = np.linalg.norm(lig_arr - o_xyz, axis=1)
        if dists.min() < clash_dist:
            n_clashing += 1
            continue
        kept_lines.extend(lines)

    if n_clashing:
        print(f'[hydrate] removed {n_clashing} waters clashing with ligand '
              f'(< {clash_dist} Å)', flush=True)

    return kept_lines
