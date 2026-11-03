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
            'id': str,                  # PLIP binding-site id, 'RES:CHAIN:NUM'
            'ligand': {'resname', 'chain', 'resnr'},
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

    Chains and residue numbers are the author ones in the input, with the
    insertion code in 'resicode' ('' when there is none). Every
    interaction also carries 'ligcoo' and 'protcoo', lists of [x, y, z] for
    the ligand-side and protein-side point PLIP measured (atoms, or group
    centres for rings and charges, whose member atoms are in 'ligatoms' and
    'protatoms'); water bridges add 'watercoo', metal complexes 'metalcoo'
    and 'targetcoo'.
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
    from plip.structure.preparation import PDBComplex

    # Step 1 — PDBFixer (PLIP needs complete, protonated structure)
    pdb_path = protonate(author_pdb(input_data, input_format), workdir)

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

    atoms = residue_index(pdb_path)
    sites = []
    for site_id, site in sorted(mol.interaction_sets.items()):
        ixt = _serialize_site(site)
        for items in ixt.values():
            for item in items:
                _set_author_residue(item, atoms, 'reschain', 'resnr', 'restype')
        total = sum(len(v) for v in ixt.values())
        resname, chain, resnr = site_id.split(':')
        sites.append({
            'id': site_id,
            'ligand': {'resname': resname, 'chain': chain, 'resnr': _int(resnr)},
            'total': total,
            'interactions': ixt,
        })

    return {'sites': sites, 'error': None}


def protonate(pdb_text, workdir):
    """
    Write a PDBFixer-completed, pH 7.4 protonated PDB and return its path.

    Takes PDB text from author_pdb(), so chain ids and residue numbers stay
    the author ones a viewer shows (writeFile keeps them only with keepIds).

    Missing residues are not modelled. A loop built with no knowledge of the
    ligand can invent contacts, and keeping the residue list unchanged keeps
    the numbering one to one with the input.
    """
    from pdbfixer import PDBFixer
    from openmm.app import PDBFile

    input_path = os.path.join(workdir, 'input.pdb')
    with open(input_path, 'w', encoding='utf-8') as fh:
        fh.write(pdb_text)

    fixer = PDBFixer(filename=input_path)
    fixer.missingResidues = {}
    fixer.findNonstandardResidues()
    fixer.replaceNonstandardResidues()
    fixer.findMissingAtoms()
    fixer.addMissingAtoms()
    fixer.addMissingHydrogens(7.4)

    pdb_path = os.path.join(workdir, 'fixed.pdb')
    with open(pdb_path, 'w') as fh:
        PDBFile.writeFile(fixer.topology, fixer.positions, fh, keepIds=True)
    return pdb_path


def residue_index(pdb_path):
    """Atom coordinates (3 dp, as written) -> (chain, resnr, icode, resname)."""
    index = {}
    with open(pdb_path) as fh:
        for line in fh:
            if line.startswith(('ATOM', 'HETATM')):
                key = (round(float(line[30:38]), 3), round(float(line[38:46]), 3),
                       round(float(line[46:54]), 3))
                index[key] = (line[21], int(line[22:26]), line[26].strip(), line[17:20].strip())
    return index


def _set_author_residue(item, index, chain_key, num_key, name_key):
    """
    Take the residue number and insertion code from the protein-side atom.

    OpenBabel, and so PLIP, reads a residue with an insertion code (thrombin's
    Trp60D) as residue 0, and ProLIF drops the code. The atom PLIP or ProLIF
    measured to is in the protonated PDB, which keeps both, so look it up
    there. Group centres are not atoms, hence the member atoms first.
    """
    points = item.get('protatoms') or item.get('protcoo') or []
    for point in points:
        hit = index.get(tuple(round(float(c), 3) for c in point))
        if hit and hit[0] == item[chain_key] and hit[3] == item[name_key]:
            item[num_key] = hit[1]
            item['resicode'] = hit[2]
            return


def author_pdb(input_data, input_format):
    """
    The structure as PDB text with author chain ids and residue numbers.

    PDBFixer reads mmCIF chains from label_asym_id whenever that column has
    more distinct ids than auth_asym_id, which is true of most RCSB entries,
    so mmCIF goes through PDB first.
    """
    import gemmi
    return read_structure(input_data, input_format).make_pdb_string(
        gemmi.PdbWriteOptions(minimal=True))


def read_structure(input_data, input_format):
    """First model as a gemmi Structure, alternate locations collapsed."""
    import gemmi

    if input_format == 'mmcif':
        st = gemmi.make_structure_from_block(gemmi.cif.read_string(input_data).sole_block())
    else:
        st = gemmi.read_pdb_string(input_data)
    for i in range(len(st) - 1, 0, -1):
        del st[i]
    st.remove_alternative_conformations()
    st.setup_entities()
    return st


def _int(val):
    try:
        return int(val)
    except (TypeError, ValueError):
        return val


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


def _xyz(coords):
    """One point as [x, y, z], 3 dp."""
    return [round(float(c), 3) for c in coords]


def _pts(*points):
    """Points as a list of [x, y, z]."""
    return [_xyz(p) for p in points]


def _group(group):
    """Coordinates of the atoms that make up a ring or charged group."""
    return _pts(*[a.coords for a in getattr(group, 'atoms', [])])


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
            'ligcoo':     _pts((h.a if h.protisdon else h.d).coords),
            'protcoo':    _pts((h.d if h.protisdon else h.a).coords),
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
            'ligcoo':      _pts(h.ligatom.coords),
            'protcoo':     _pts(h.bsatom.coords),
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
            'ligcoo':    _pts(p.ligandring.center),
            'protcoo':   _pts(p.proteinring.center),
            'ligatoms':  _group(p.ligandring),
            'protatoms': _group(p.proteinring),
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
            'ligcoo':       _pts((p.ring if p.protcharged else p.charge).center),
            'protcoo':      _pts((p.charge if p.protcharged else p.ring).center),
            'ligatoms':     _group(p.ring if p.protcharged else p.charge),
            'protatoms':    _group(p.charge if p.protcharged else p.ring),
        }
    )

    # ---- Salt Bridges ----
    def _saltbridge(sb):
        lig, prot = (sb.negative, sb.positive) if sb.protispos else (sb.positive, sb.negative)
        return {
            'resnr':      sb.resnr,
            'restype':    sb.restype,
            'reschain':   sb.reschain,
            'dist':       _f(sb.distance),
            'protispos':  _bool(sb.protispos),
            'lig_group':  lig.fgroup.capitalize(),
            'ligcoo':     _pts(lig.center),
            'protcoo':    _pts(prot.center),
            'ligatoms':   _group(lig),
            'protatoms':  _group(prot),
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
            'ligcoo':      _pts((w.a if w.protisdon else w.d).coords),
            'protcoo':     _pts((w.d if w.protisdon else w.a).coords),
            'watercoo':    _pts(w.water.coords),
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
            'ligcoo':     _pts(h.don.x.coords),
            'protcoo':    _pts(h.acc.o.coords),
        }
    )

    # ---- Metal Complexes ----
    # The target sits on the ligand when location is 'ligand'; otherwise the
    # ligand side is the metal and the target is protein or water.
    def _metal(m):
        on_lig = m.location == 'ligand'
        return {
            'resnr':        m.resnr,
            'restype':      m.restype,
            'reschain':     m.reschain,
            'metal_idx':    m.metal_orig_idx,
            'metal_type':   m.metal_type,
            'target_idx':   m.target_orig_idx,
            'target_type':  m.target_type,
            'coordination': m.coordination_num,
            'dist':         _f(m.distance),
            'location':     m.location,
            'geometry':     m.geometry,
            'metalcoo':     _pts(m.metal.coords),
            'targetcoo':    _pts(m.target.atom.coords),
            'ligcoo':       _pts((m.target.atom if on_lig else m.metal).coords),
            'protcoo':      _pts((m.metal if on_lig else m.target.atom).coords),
        }
    metalcomplexes = safe(getattr(site, 'metal_complexes', []), _metal)

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
