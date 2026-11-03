"""
BoltzStar ProLIF analysis: interaction fingerprint for one ligand.

The protein is completed and protonated exactly as for PLIP (PDBFixer,
pH 7.4, author numbering), then read through MDAnalysis with OpenMM's
template bonds; its RDKit converter infers bond orders and charges from the
explicit hydrogens. The
ligand's heavy atoms go through OpenBabel, which perceives bond orders from
the geometry and adds hydrogens at pH 7.4, since predicted poses carry none.
"""

import os
import shutil
import tempfile
import warnings

# ProLIF's default set, plus halogen bonds
INTERACTIONS = [
    'Hydrophobic', 'HBDonor', 'HBAcceptor', 'PiStacking', 'Anionic',
    'Cationic', 'CationPi', 'PiCation', 'VdWContact', 'XBAcceptor', 'XBDonor',
]
# Run alongside PiStacking only to report which geometry fired
_STACK_SUBTYPES = ['FaceToFace', 'EdgeToFace']

# Het groups that are almost never the ligand of interest: crystallisation
# additives, buffers, lipids, detergents, sugars and cofactor hemes.
ADDITIVES = {
    'SO4', 'PO4', 'GOL', 'EDO', 'PEG', 'PGE', 'PG4', '1PE', '12P', 'P6G',
    'ACT', 'ACY', 'ACM', 'BU1', 'DMS', 'MPD', 'TRS', 'EPE', 'MES', 'BME',
    'FMT', 'IMD', 'SIN', 'CIT', 'FLC', 'TAR', 'IPA', 'EOH', 'MOH',
    'CLR', 'PLM', 'OLC', 'OLA', 'OLB', 'MYR', 'STE', 'LDA', 'BOG', 'LMT',
    'DMU', 'UNL', 'NAG', 'NDG', 'MAN', 'BMA', 'GAL', 'GLC', 'FUC', 'SIA',
    'HEM', 'HEC', 'LPD',
}

# Distance (A) from the ligand within which a whole chain is kept
_CHAIN_CUTOFF = 12.0


def run_prolif(input_data: str, input_format: str, ligand: dict | None = None,
               smiles: str | None = None) -> dict:
    """
    Run a ProLIF fingerprint for one ligand of the given structure.

    ligand is {'resname', 'chain', 'resnr'}; None picks the largest het group
    that is not water, an ion or a common additive. smiles, when given, sets
    the ligand's bond orders; without it they are perceived from the geometry.

    Returns
    -------
    {
        'ligand':   {'resname', 'chain', 'resnr'},
        'ligands':  [{'resname', 'chain', 'resnr', 'n_heavy'}, ...],
        'bond_orders': 'smiles' | 'perceived',
        'warnings': [str, ...],
        'interactions': [{
            'chain', 'resnr', 'resicode', 'restype',
            'itype',     # ProLIF class name, e.g. 'HBDonor', 'PiStacking'
            'subtype',   # PiStacking only: 'FaceToFace' or 'EdgeToFace'
            'dist',
            'ligcoo':  [[x, y, z], ...],   # heavy atoms only; an H counts
            'protcoo': [[x, y, z], ...],   # as the atom it is bonded to
            ...          # any angles ProLIF reports, e.g. 'DHA_angle'
        }],
        'error': str | None
    }

    Chains and residue numbers are the author ones in the input.
    """
    workdir = tempfile.mkdtemp(prefix='boltzstar_prolif_')
    try:
        return _run(input_data, input_format, ligand, smiles, workdir)
    finally:
        shutil.rmtree(workdir, ignore_errors=True)


# ----------------------------------------------------------------
# Pipeline
# ----------------------------------------------------------------

def _run(input_data, input_format, ligand, smiles, workdir):
    import gemmi
    import numpy as np
    from plip_analysis import read_structure, protonate, residue_index, _set_author_residue

    st = read_structure(input_data, input_format)
    candidates = list_ligands(st)
    want = _match(candidates, ligand) if ligand else pick_ligand(candidates)
    out = {'ligand': want, 'ligands': candidates, 'bond_orders': None,
           'warnings': [], 'interactions': [], 'error': None}
    if want is None:
        out['error'] = ('Ligand not found.' if ligand else
                        'No ligand found. Ensure the structure contains a small-molecule ligand.')
        return out

    lig_res = _find_residue(st, want)
    lig_atoms = [a for a in lig_res if a.element.name not in ('H', 'D')]
    try:
        lig_mol, out['bond_orders'], out['warnings'] = _ligand_mol(lig_atoms, want, smiles)
    except ValueError as e:
        out['error'] = str(e)
        return out

    # Protein: polymer chains with any atom near the ligand. Whole chains, so
    # PDBFixer and the bond-order inference never see an artificial gap.
    lig_xyz = np.array([a.pos.tolist() for a in lig_atoms])
    prot = st.clone()
    prot.remove_ligands_and_waters()
    prot.remove_hydrogens()
    for i in range(len(prot[0]) - 1, -1, -1):
        ch = prot[0][i]
        xyz = np.array([a.pos.tolist() for r in ch for a in r])
        if not len(xyz) or _min_dist(xyz, lig_xyz) > _CHAIN_CUTOFF:
            del prot[0][i]
    if not len(prot[0]):
        out['error'] = 'No protein chain within %.0f A of the ligand.' % _CHAIN_CUTOFF
        return out
    prot_pdb = protonate(prot.make_pdb_string(gemmi.PdbWriteOptions(minimal=True)), workdir)

    prot_mol = _protein_mol(prot_pdb)

    out['interactions'] = _fingerprint(lig_mol, prot_mol)
    atoms = residue_index(prot_pdb)
    for item in out['interactions']:
        _set_author_residue(item, atoms, 'chain', 'resnr', 'restype')
    return out


def _protein_mol(pdb_path):
    """Bonds come from OpenMM's residue templates, not from distances: guessed
    bonds join clashing atoms (pentavalent carbons) and miss stretched ones
    (fragmented residues), and either stops the bond-order inference."""
    import MDAnalysis as mda
    import prolif as plf
    from openmm.app import PDBFile

    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        pdb = PDBFile(pdb_path)
        u = mda.Universe(pdb)
        # MDAnalysis numbers residues from OpenMM sequentially; put the
        # author numbers (kept by protonate) back
        author = [int(r.id) for r in pdb.topology.residues()]
        u.residues.resids = author
        u.residues.resnums = author
        return plf.Molecule.from_mda(u.atoms)


def _ligand_mol(atoms, want, smiles=None):
    """
    Heavy atoms -> bond orders -> pH 7.4 hydrogens (OpenBabel) -> RDKit.

    Bond orders come from the SMILES when one is given. Otherwise OpenBabel
    perceives them from the geometry, which works for most ligands but can
    fail on fused heteroaromatics: it then misassigns a bond (a lactam C=N
    beside its C=O, say) and the hydrogen count with it.

    Returns (mol, how, warnings) where how says which route set the bond
    orders.
    """
    import prolif as plf
    from openbabel import pybel
    from rdkit import Chem

    lines = []
    for i, a in enumerate(atoms, 1):
        lines.append('HETATM%5d %-4s %3s %1s%4d    %8.3f%8.3f%8.3f  1.00  0.00          %2s' % (
            i, a.name[:4], want['resname'][:3], (want['chain'] or 'A')[:1], int(want['resnr']),
            a.pos.x, a.pos.y, a.pos.z, a.element.name.upper()))
    block = '\n'.join(lines) + '\nEND\n'

    rdmol, how, warnings_ = None, None, []
    if smiles:
        templated = _from_template(block, smiles)
        if templated is not None:
            rdmol = _protonated(lambda: pybel.readstring('mol', Chem.MolToMolBlock(templated)))
            how = 'smiles'
        else:
            warnings_.append('The SMILES does not match the ligand atoms, so bond orders '
                             'were perceived from the geometry instead.')
    if rdmol is None:
        rdmol = _protonated(lambda: pybel.readstring('pdb', block))
        how = 'perceived'
    if rdmol is None:
        raise ValueError(' '.join(warnings_ + [
            'Could not perceive the ligand chemistry from its coordinates.',
            'Pass its SMILES to set the bond orders.']))

    mol = plf.Molecule.from_rdkit(rdmol, resname=want['resname'],
                                  resnumber=int(want['resnr']), chain=want['chain'])
    return mol, how, warnings_


def _protonated(read):
    """Add hydrogens at pH 7.4, falling back to a neutral protonation when
    RDKit rejects the charged form OpenBabel wrote. read() makes a fresh
    pybel molecule each time, since adding hydrogens changes it."""
    from rdkit import Chem

    for correct_ph in (True, False):
        ob = read()
        ob.OBMol.AddHydrogens(False, correct_ph, 7.4)
        rdmol = Chem.MolFromMolBlock(ob.write('mol'), removeHs=False)
        if rdmol is not None:
            return rdmol
    return None


def _from_template(block, smiles):
    """Heavy atoms with bond orders taken from the SMILES, or None if the
    SMILES is not this ligand. RDKit will happily match a smaller template
    as a substructure, so the heavy-atom formulas must agree first."""
    from collections import Counter
    from rdkit import Chem
    from rdkit.Chem import AllChem

    heavy = Chem.MolFromPDBBlock(block, removeHs=False, sanitize=False)
    template = Chem.MolFromSmiles(smiles)
    if heavy is None or template is None:
        return None
    formula = lambda m: Counter(a.GetSymbol() for a in m.GetAtoms() if a.GetAtomicNum() > 1)
    if formula(heavy) != formula(template):
        return None
    try:
        return AllChem.AssignBondOrdersFromTemplate(template, heavy)
    except Exception:
        return None


def _fingerprint(lig_mol, prot_mol):
    import prolif as plf

    fp = plf.Fingerprint(INTERACTIONS + _STACK_SUBTYPES, count=False)
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        fp.run_from_iterable([lig_mol], prot_mol, progress=False, n_jobs=1)

    rows = []
    for (lres, pres), hits in fp.ifp[0].items():
        for itype, metas in hits.items():
            if itype in _STACK_SUBTYPES:
                continue
            for meta in metas:
                row = {
                    'chain':   pres.chain,
                    'resnr':   pres.number,
                    'restype': pres.name,
                    'itype':   itype,
                    'dist':    _f(meta.get('distance')),
                    'ligcoo':  _heavy_xyz(lig_mol, meta['parent_indices']['ligand']),
                    'protcoo': _heavy_xyz(prot_mol, meta['parent_indices']['protein']),
                }
                if itype == 'PiStacking':
                    row['subtype'] = next((s for s in _STACK_SUBTYPES if s in hits), None)
                for k, v in meta.items():
                    if k not in ('indices', 'parent_indices', 'distance') and isinstance(v, (int, float)):
                        row[k] = _f(v)
                rows.append(row)
    rows.sort(key=lambda r: (r['chain'], r['resnr'], r['itype']))
    return rows


# ----------------------------------------------------------------
# Ligand candidates
# ----------------------------------------------------------------

def list_ligands(st):
    """Every non-polymer, non-water het group with more than one heavy atom."""
    import gemmi

    out = []
    for ch in st[0]:
        for r in ch:
            if r.het_flag != 'H' or r.is_water() or r.entity_type == gemmi.EntityType.Polymer:
                continue
            n = sum(1 for a in r if a.element.name not in ('H', 'D'))
            if n > 1:
                out.append({'resname': r.name, 'chain': ch.name, 'resnr': r.seqid.num, 'n_heavy': n})
    return out


def pick_ligand(candidates):
    """Largest candidate that is not a common additive (largest of all if none)."""
    pool = [c for c in candidates if c['resname'] not in ADDITIVES] or candidates
    if not pool:
        return None
    best = max(pool, key=lambda c: c['n_heavy'])
    return {k: best[k] for k in ('resname', 'chain', 'resnr')}


def _match(candidates, ligand):
    for c in candidates:
        if (c['resname'] == ligand.get('resname') and c['chain'] == ligand.get('chain')
                and str(c['resnr']) == str(ligand.get('resnr'))):
            return {k: c[k] for k in ('resname', 'chain', 'resnr')}
    return None


def _find_residue(st, want):
    for ch in st[0]:
        if ch.name != want['chain']:
            continue
        for r in ch:
            if r.name == want['resname'] and r.seqid.num == int(want['resnr']):
                return r
    raise ValueError('Ligand %(resname)s:%(chain)s:%(resnr)s not found.' % want)


# ----------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------

def _min_dist(a, b):
    import numpy as np
    d = np.sqrt(((a[:, None, :] - b[None, :, :]) ** 2).sum(-1))
    return float(d.min())


def _heavy_xyz(mol, indices):
    """Coordinates of the heavy atoms behind these indices. A hydrogen (an
    H-bond donor H, or either end of a VdW contact) stands for the heavy atom
    it is bonded to."""
    xyz = mol.xyz
    heavy = []
    for i in indices:
        atom = mol.GetAtomWithIdx(int(i))
        if atom.GetAtomicNum() == 1:
            nbrs = atom.GetNeighbors()
            if not nbrs:
                continue
            atom = nbrs[0]
        if atom.GetIdx() not in heavy:
            heavy.append(atom.GetIdx())
    return [[round(float(c), 3) for c in xyz[i]] for i in heavy]


def _f(val, decimals=2):
    try:
        return round(float(val), decimals)
    except (TypeError, ValueError):
        return val
