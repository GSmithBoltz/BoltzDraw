"""
BoltzStar PLIP analysis — protein-ligand interaction profiling.

Returns rich, web-server-style interaction tables for every binding site.
Field names match the canonical PLIP BindingSiteReport fields exactly.
"""

import os
import shutil
import tempfile


def run_plip(input_data: str, input_format: str) -> dict:
    """
    Run PLIP on the given structure.

    Returns
    -------
    {
        'sites': [{
            'id': str,
            'total': int,
            'interactions': {
                'hbonds':       [...],
                'hydrophobic':  [...],
                'pistacking':   [...],
                'pication':     [...],
                'saltbridges':  [...],
                'waterbridges': [...],
                'halogenbonds': [...],
                'metalcomplexes': [...],
            }
        }],
        'error': str | None
    }
    """
    workdir = tempfile.mkdtemp(prefix='boltzstar_plip_')
    try:
        return _run(input_data, input_format, workdir)
    finally:
        shutil.rmtree(workdir, ignore_errors=True)


# ----------------------------------------------------------------
# Pipeline
# ----------------------------------------------------------------

def _run(input_data, input_format, workdir):
    from pdbfixer import PDBFixer
    from openmm.app import PDBFile
    from plip.structure.preparation import PDBComplex

    # Step 1 — PDBFixer (PLIP needs complete, protonated structure)
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

    pdb_path = os.path.join(workdir, 'fixed.pdb')
    with open(pdb_path, 'w') as fh:
        PDBFile.writeFile(fixer.topology, fixer.positions, fh)

    # Step 2 — PLIP
    mol = PDBComplex()
    mol.load_pdb(pdb_path)
    mol.analyze()

    if not mol.interaction_sets:
        return {
            'sites': [],
            'error': 'No ligand binding sites found. '
                     'Ensure the structure contains a small-molecule ligand.'
        }

    sites = []
    for site_id, site in sorted(mol.interaction_sets.items()):
        ixt = _serialize_site(site)
        total = sum(len(v) for v in ixt.values())
        sites.append({'id': site_id, 'total': total, 'interactions': ixt})

    return {'sites': sites, 'error': None}


# ----------------------------------------------------------------
# Serialiser — mirrors BindingSiteReport field order exactly
# ----------------------------------------------------------------

def _f(val, decimals=2):
    """Format a float or return the value as-is."""
    try:
        return round(float(val), decimals)
    except (TypeError, ValueError):
        return val


def _bool(val):
    return bool(val)


def _serialize_site(site) -> dict:

    def safe(items, fn):
        try:
            return [fn(x) for x in (items or [])]
        except Exception:
            return []

    # ---- Hydrogen Bonds ----
    hbonds = safe(
        list(site.hbonds_pdon) + list(site.hbonds_ldon),
        lambda h: {
            'resnr':      h.resnr,
            'restype':    h.restype,
            'reschain':   h.reschain,
            'sidechain':  _bool(h.sidechain),
            'dist_h_a':   _f(h.distance_ah),
            'dist_d_a':   _f(h.distance_ad),
            'don_angle':  _f(h.angle),
            'protisdon':  _bool(h.protisdon),
            'donor_idx':  h.d_orig_idx,
            'donortype':  h.dtype,
            'acc_idx':    h.a_orig_idx,
            'acctype':    h.atype,
        }
    )

    # ---- Hydrophobic Contacts ----
    hydrophobic = safe(
        site.hydrophobic_contacts,
        lambda h: {
            'resnr':       h.resnr,
            'restype':     h.restype,
            'reschain':    h.reschain,
            'dist':        _f(h.distance),
            'lig_atom':    h.ligatom_orig_idx,
            'prot_atom':   h.bsatom_orig_idx,
        }
    )

    # ---- π-Stacking ----
    pistacking = safe(
        site.pistacking,
        lambda p: {
            'resnr':     p.resnr,
            'restype':   p.restype,
            'reschain':  p.reschain,
            'cent_dist': _f(p.distance),
            'angle':     _f(p.angle),
            'offset':    _f(p.offset),
            'type':      'Parallel' if p.type == 'P' else 'T-shaped',
        }
    )

    # ---- π-Cation ----
    pication = safe(
        list(site.pication_laro) + list(site.pication_paro),
        lambda p: {
            'resnr':        p.resnr,
            'restype':      p.restype,
            'reschain':     p.reschain,
            'dist':         _f(p.distance),
            'offset':       _f(p.offset),
            'protcharged':  _bool(p.protcharged),
        }
    )

    # ---- Salt Bridges ----
    def _saltbridge(sb):
        return {
            'resnr':      sb.resnr,
            'restype':    sb.restype,
            'reschain':   sb.reschain,
            'dist':       _f(sb.distance),
            'protispos':  _bool(sb.protispos),
            'lig_group':  (sb.negative.fgroup if sb.protispos else sb.positive.fgroup).capitalize(),
        }
    saltbridges = safe(list(site.saltbridge_lneg) + list(site.saltbridge_pneg), _saltbridge)

    # ---- Water Bridges ----
    waterbridges = safe(
        site.water_bridges,
        lambda w: {
            'resnr':       w.resnr,
            'restype':     w.restype,
            'reschain':    w.reschain,
            'dist_a_w':    _f(w.distance_aw),
            'dist_d_w':    _f(w.distance_dw),
            'don_angle':   _f(w.d_angle),
            'water_angle': _f(w.w_angle),
            'protisdon':   _bool(w.protisdon),
            'donor_idx':   w.d_orig_idx,
            'donortype':   w.dtype,
            'acc_idx':     w.a_orig_idx,
            'acctype':     w.atype,
            'water_idx':   w.water_orig_idx,
        }
    )

    # ---- Halogen Bonds ----
    halogenbonds = safe(
        site.halogen_bonds,
        lambda h: {
            'resnr':      h.resnr,
            'restype':    h.restype,
            'reschain':   h.reschain,
            'sidechain':  _bool(h.sidechain),
            'dist':       _f(h.distance),
            'don_angle':  _f(h.don_angle),
            'acc_angle':  _f(h.acc_angle),
            'don_idx':    h.don_orig_idx,
            'donortype':  h.donortype,
            'acc_idx':    h.acc_orig_idx,
            'acctype':    h.acctype,
        }
    )

    # ---- Metal Complexes ----
    metalcomplexes = safe(
        getattr(site, 'metal_complexes', []),
        lambda m: {
            'resnr':        m.resnr,
            'restype':      m.restype,
            'reschain':     m.reschain,
            'metal_idx':    m.metal_orig_idx,
            'metal_type':   m.metal_type,
            'target_idx':   m.target_orig_idx,
            'target_type':  m.targettype,
            'coordination': m.coordination,
            'dist':         _f(m.dist),
            'geometry':     m.geometry,
        }
    )

    return {
        'hbonds':         hbonds,
        'hydrophobic':    hydrophobic,
        'pistacking':     pistacking,
        'pication':       pication,
        'saltbridges':    saltbridges,
        'waterbridges':   waterbridges,
        'halogenbonds':   halogenbonds,
        'metalcomplexes': metalcomplexes,
    }
