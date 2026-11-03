/**
 * contactengine.js
 *
 * Protein–ligand interaction profiling that runs in the browser, so BoltzContacts
 * works on a static host. Two rule sets read one typed structure:
 *
 *   plip    PLIP 3.0's detection and refinement (plip/structure/detection.py and
 *           preparation.py), thresholds from plip/basic/config.py. PLIP takes its
 *           atom typing from OpenBabel 3.2; that typing is reproduced here from
 *           geometry, quirks included, because the aim is to agree with PLIP, not
 *           to improve on it.
 *   prolif  ProLIF 2.2's default fingerprint plus halogen bonds. Its SMARTS are
 *           translated onto the same typing, with ligand charges at pH 7.4 as
 *           OpenBabel's pH model assigns them (that is how the exact pipeline
 *           protonates the ligand).
 *
 * Neither library runs in a browser (both need OpenBabel or RDKit's Python
 * bindings, and Pyodide ships neither), so this is a port of their rules, not
 * the libraries. With the local server running, BoltzContacts can call the real
 * tools instead.
 *
 * Hydrogens. Boltz poses carry none. When a donor has no hydrogen in the file,
 * its hydrogen is placed from the heavy-atom geometry: fixed for planar donors
 * (backbone NH, aromatic NH, amide NH2), and for rotatable ones (OH, NH3+) the
 * rotamer that best serves each acceptor is used. Results computed that way are
 * flagged `approx`.
 *
 * Pure module: no DOM, so Node can import it for testing.
 */

// ================================================================
// Thresholds
// ================================================================

export const PLIP_CONFIG = {
    BS_DIST: 7.5, MIN_DIST: 0.5,
    HYDROPH_DIST_MAX: 4.0,
    HBOND_DIST_MAX: 4.1, HBOND_DON_ANGLE_MIN: 100,
    PISTACK_DIST_MAX: 5.5, PISTACK_ANG_DEV: 30, PISTACK_OFFSET_MAX: 2.0,
    PICATION_DIST_MAX: 6.0,
    SALTBRIDGE_DIST_MAX: 5.5,
    HALOGEN_DIST_MAX: 4.0, HALOGEN_ACC_ANGLE: 120, HALOGEN_DON_ANGLE: 165, HALOGEN_ANGLE_DEV: 30,
    WATER_BRIDGE_MINDIST: 2.5, WATER_BRIDGE_MAXDIST: 4.1,
    WATER_BRIDGE_OMEGA_MIN: 71, WATER_BRIDGE_OMEGA_MAX: 140, WATER_BRIDGE_THETA_MIN: 100,
    AROMATIC_PLANARITY: 5.0,
};

export const PROLIF_CONFIG = {
    hydrophobic: 4.5,
    hbond: { distance: 3.5, angle: [130, 180] },
    ionic: 4.5,
    cationpi: { distance: 4.5, angle: [0, 30] },
    facetoface: { distance: 5.5, plane: [0, 35], ncc: [0, 33] },
    edgetoface: { distance: 6.5, plane: [50, 90], ncc: [0, 30], intersect: 1.5 },
    halogen: { distance: 3.5, axd: [130, 180], xar: [80, 140] },
    vdwTolerance: 0.0,
    // ProLIF's exact pipeline protonates both partners, and most vdW contacts it
    // finds are made by hydrogens. Without them, heavy-atom pairs get this much
    // extra reach (fitted against ProLIF on the benchmark set).
    vdwHeavyMargin: 0.4,
};

/** Interaction types per rule set, in display order. */
export const PLIP_TYPES = ['hbond', 'saltbridge', 'pistacking', 'pication', 'halogen', 'waterbridge', 'hydrophobic'];
export const PROLIF_TYPES = ['HBDonor', 'HBAcceptor', 'Cationic', 'Anionic', 'PiStacking', 'CationPi', 'PiCation',
                             'XBDonor', 'Hydrophobic', 'VdWContact'];

// ================================================================
// Elements, residues
// ================================================================

const COV_RADII = {
    H: 0.31, C: 0.76, N: 0.71, O: 0.66, F: 0.57, P: 1.07, S: 1.05, CL: 1.02, BR: 1.20, I: 1.39,
    B: 0.84, SI: 1.11, SE: 1.20, NA: 1.66, MG: 1.41, K: 2.03, CA: 1.76, MN: 1.39, FE: 1.32,
    CO: 1.26, NI: 1.24, CU: 1.32, ZN: 1.22,
};
// MDAnalysis vdW radii, ProLIF's default preset
const VDW_RADII = {
    H: 1.1, C: 1.7, N: 1.55, O: 1.52, F: 1.47, P: 1.8, S: 1.8, CL: 1.75, BR: 1.85, I: 1.98,
    SE: 1.9, B: 1.92, SI: 2.1,
};
const MAX_VALENCE = { C: 4, N: 4, O: 2, F: 1, CL: 1, BR: 1, I: 1, S: 6, P: 5, B: 4, SE: 2, SI: 4 };
const HALOGENS = new Set(['F', 'CL', 'BR', 'I']);

const AA = new Set(['ALA', 'ARG', 'ASN', 'ASP', 'CYS', 'GLN', 'GLU', 'GLY', 'HIS', 'ILE',
                    'LEU', 'LYS', 'MET', 'PHE', 'PRO', 'SER', 'THR', 'TRP', 'TYR', 'VAL']);
const AA_ALIAS = {
    MSE: 'MET', SEC: 'CYS', HID: 'HIS', HIE: 'HIS', HIP: 'HIS', HSD: 'HIS', HSE: 'HIS', HSP: 'HIS',
    CYX: 'CYS', CYM: 'CYS', ASH: 'ASP', GLH: 'GLU', LYN: 'LYS',
};
const NUCLEIC = new Set(['A', 'C', 'G', 'U', 'I', 'DA', 'DC', 'DG', 'DT', 'DI', 'DU']);
const WATER = new Set(['HOH', 'WAT', 'H2O', 'DOD', 'TIP', 'TIP3', 'SOL', 'SPC']);
const METALS = new Set(['CA', 'CO', 'MG', 'MN', 'FE', 'CU', 'ZN', 'FE2', 'FE3', 'FE4', 'LI', 'NA', 'K',
    'RB', 'SR', 'CS', 'BA', 'CR', 'NI', 'FE1', 'RU', 'RU1', 'RH', 'RH1', 'PD', 'AG', 'CD', 'LA',
    'W', 'W1', 'OS', 'IR', 'PT', 'PT1', 'AU', 'HG', 'CE', 'PR', 'SM', 'EU', 'GD', 'TB', 'YB',
    'LU', 'AL', 'GA', 'IN', 'SB', 'TL', 'PB']);
// Het groups that are almost never the ligand of interest. Still offered, listed last.
const ADDITIVES = new Set(['SO4', 'PO4', 'GOL', 'EDO', 'EOH', 'PEG', 'PG4', 'PGE', '1PE', 'P6G', 'MPD',
    'DMS', 'ACT', 'FMT', 'ACE', 'NH2', 'BME', 'TRS', 'MES', 'EPE', 'HED', 'IMD', 'CL', 'BR', 'IOD',
    'NO3', 'NH4', 'SCN', 'AZI', 'IPA', 'BU3', 'PE4', 'MRD', 'CIT', 'FLC', 'TAR', 'MLI', 'BOG', 'LDA',
    'UNX', 'UNL', 'CXS', 'CAC', 'SIN', 'OXY', 'PEO', 'HEZ', 'BCT', 'CO3', 'EGL']);

export const canonicalResname = (rn) => (AA.has(rn) ? rn : (AA_ALIAS[rn] || rn));
const isAminoAcid = (rn) => AA.has(rn) || rn in AA_ALIAS;

// ================================================================
// Vector helpers
// ================================================================

const sub = (a, b) => [a[0] - b[0], a[1] - b[1], a[2] - b[2]];
const add = (a, b) => [a[0] + b[0], a[1] + b[1], a[2] + b[2]];
const scale = (a, s) => [a[0] * s, a[1] * s, a[2] * s];
const dot = (a, b) => a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
const cross = (a, b) => [a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0]];
const norm = (a) => Math.sqrt(dot(a, a));
const unit = (a) => { const n = norm(a); return n > 1e-9 ? scale(a, 1 / n) : [0, 0, 0]; };
const dist = (a, b) => norm(sub(a, b));
const DEG = 180 / Math.PI;

function angleDeg(u, v) {
    const nu = norm(u), nv = norm(v);
    if (nu < 1e-9 || nv < 1e-9) return 0;
    return Math.acos(Math.max(-1, Math.min(1, dot(u, v) / (nu * nv)))) * DEG;
}

/** Angle between two plane normals, folded to 0–90. */
const planeAngle = (n1, n2) => { const a = angleDeg(n1, n2); return a > 90 ? 180 - a : a; };

function centroid(pts) {
    const c = [0, 0, 0];
    for (const p of pts) { c[0] += p[0]; c[1] += p[1]; c[2] += p[2]; }
    return scale(c, 1 / pts.length);
}

/** Ring normal by Newell's method, robust to slight non-planarity. */
function ringNormal(pts) {
    const n = [0, 0, 0];
    for (let i = 0; i < pts.length; i++) {
        const a = pts[i], b = pts[(i + 1) % pts.length];
        n[0] += (a[1] - b[1]) * (a[2] + b[2]);
        n[1] += (a[2] - b[2]) * (a[0] + b[0]);
        n[2] += (a[0] - b[0]) * (a[1] + b[1]);
    }
    return unit(n);
}

/** Projection of point p onto the plane through c with normal n. */
const projectOnPlane = (n, c, p) => sub(p, scale(n, dot(sub(p, c), n)));

// ================================================================
// Parsing
// ================================================================

/**
 * Parse a PDB or mmCIF file (first model, first alternate location).
 * @returns {Structure} {name, atoms, residues, hasHydrogens}
 */
export function parseStructure(text, filename = '') {
    const head = text.slice(0, 8192);
    const isCif = /\.(cif|mmcif)$/i.test(filename) ||
        (!/\.(pdb|ent)$/i.test(filename) && (/^data_/m.test(head) || text.includes('_atom_site.')));
    const atoms = _dropAltlocs(isCif ? _parseCif(text) : _parsePdb(text));
    if (!atoms.length) throw new Error('no atom records found');
    return _buildStructure(atoms, filename);
}

function _parsePdb(text) {
    const atoms = [];
    for (const line of text.split(/\r?\n/)) {
        const rec = line.slice(0, 6);
        if (rec === 'ENDMDL' && atoms.length) break;
        if (rec !== 'ATOM  ' && rec !== 'HETATM') continue;
        const rawName = line.slice(12, 16);
        const resname = line.slice(17, 20).trim().toUpperCase();
        let el = line.slice(76, 78).trim().toUpperCase();
        if (!el || /\d/.test(el)) el = _elementFromName(rawName, resname, rec === 'HETATM');
        atoms.push({
            serial: parseInt(line.slice(6, 11), 10),
            name: rawName.trim(),
            alt: (line[16] || ' ').trim(),
            resname,
            chain: (line[21] || ' ').trim(),
            resnr: parseInt(line.slice(22, 26), 10),
            icode: (line[26] || ' ').trim(),
            x: parseFloat(line.slice(30, 38)),
            y: parseFloat(line.slice(38, 46)),
            z: parseFloat(line.slice(46, 54)),
            el,
            het: rec === 'HETATM',
        });
    }
    return atoms;
}

function _elementFromName(rawName, resname, het) {
    const name = rawName.trim().toUpperCase();
    // A two-letter element sits in columns 13-14; a one-letter element in column 14
    if (het && rawName[0] !== ' ' && /^[A-Z]{2}/.test(rawName)) {
        const two = rawName.slice(0, 2).toUpperCase();
        if (two in COV_RADII || METALS.has(two)) return two;
    }
    if (het && (name === resname) && (name in COV_RADII || METALS.has(name))) return name;
    const m = name.match(/[A-Z]/);
    return m ? m[0] : 'C';
}

function _cifTokens(line) {
    const out = [];
    const n = line.length;
    let i = 0;
    while (i < n) {
        while (i < n && (line[i] === ' ' || line[i] === '\t')) i++;
        if (i >= n) break;
        const q = line[i];
        if (q === "'" || q === '"') {
            let j = i + 1;
            while (j < n && !(line[j] === q && (j + 1 === n || line[j + 1] === ' ' || line[j + 1] === '\t'))) j++;
            out.push(line.slice(i + 1, j));
            i = j + 1;
        } else {
            let j = i;
            while (j < n && line[j] !== ' ' && line[j] !== '\t') j++;
            out.push(line.slice(i, j));
            i = j;
        }
    }
    return out;
}

function _parseCif(text) {
    const lines = text.split(/\r?\n/);
    let i = 0;
    while (i < lines.length && !lines[i].startsWith('_atom_site.')) i++;
    const fields = [];
    while (i < lines.length && lines[i].startsWith('_atom_site.')) {
        fields.push(lines[i].trim().slice(11).split(/\s/)[0]);
        i++;
    }
    if (!fields.length) return [];
    const col = (...names) => { for (const n of names) { const k = fields.indexOf(n); if (k >= 0) return k; } return -1; };
    const C = {
        group: col('group_PDB'), id: col('id'), el: col('type_symbol'),
        name: col('auth_atom_id', 'label_atom_id'), alt: col('label_alt_id'),
        resname: col('auth_comp_id', 'label_comp_id'), chain: col('auth_asym_id', 'label_asym_id'),
        resnr: col('auth_seq_id', 'label_seq_id'), label_seq: col('label_seq_id'),
        icode: col('pdbx_PDB_ins_code'), x: col('Cartn_x'), y: col('Cartn_y'), z: col('Cartn_z'),
        model: col('pdbx_PDB_model_num'),
    };
    const val = (row, k) => (k < 0 || row[k] === '?' || row[k] === '.' ? '' : row[k]);
    const atoms = [];
    let buf = [];
    let firstModel = null;
    for (; i < lines.length; i++) {
        const line = lines[i];
        if (!line.trim()) continue;
        if (line[0] === '#' || line[0] === '_' || line.startsWith('loop_') || line.startsWith('data_')) break;
        for (const t of _cifTokens(line)) buf.push(t);
        while (buf.length >= fields.length) {
            const row = buf.slice(0, fields.length);
            buf = buf.slice(fields.length);
            const model = val(row, C.model);
            if (firstModel === null) firstModel = model;
            if (model !== firstModel) return atoms;
            let resnr = parseInt(val(row, C.resnr), 10);
            if (!Number.isFinite(resnr)) resnr = parseInt(val(row, C.label_seq), 10);
            const resname = val(row, C.resname).toUpperCase();
            let el = val(row, C.el).toUpperCase();
            const name = val(row, C.name);
            if (!el) el = _elementFromName(' ' + name.padEnd(3), resname, true);
            atoms.push({
                serial: parseInt(val(row, C.id), 10),
                name, alt: val(row, C.alt), resname,
                chain: val(row, C.chain),
                resnr: Number.isFinite(resnr) ? resnr : 0,
                icode: val(row, C.icode),
                x: parseFloat(row[C.x]), y: parseFloat(row[C.y]), z: parseFloat(row[C.z]),
                el,
                het: val(row, C.group) === 'HETATM',
            });
        }
    }
    return atoms;
}

/** Keep blank altlocs, and for each atom only the first alternate location met. */
function _dropAltlocs(atoms) {
    const firstAlt = new Map();
    return atoms.filter(a => {
        if (!a.alt) return true;
        const key = `${a.chain}|${a.resnr}|${a.icode}|${a.resname}|${a.name}`;
        if (!firstAlt.has(key)) { firstAlt.set(key, a.alt); return true; }
        return false;
    });
}

function _buildStructure(atoms, name) {
    const residues = [];
    let cur = null;
    atoms.forEach((a, i) => {
        a.i = i;
        a.xyz = [a.x, a.y, a.z];
        const key = `${a.chain}|${a.resnr}|${a.icode}|${a.resname}`;
        if (!cur || cur.key !== key) {
            cur = { key, idx: residues.length, chain: a.chain, resnr: a.resnr, icode: a.icode,
                    resname: a.resname, atoms: [], kind: _residueKind(a.resname) };
            residues.push(cur);
        }
        a.res = cur.idx;
        cur.atoms.push(i);
    });
    const seenChain = new Set();
    for (const r of residues) {
        if (r.kind === 'het' && r.atoms.length === 1 && METALS.has(atoms[r.atoms[0]].el)) r.kind = 'metal';
        if (r.kind === 'aa' && !seenChain.has(r.chain)) { seenChain.add(r.chain); r.nterm = true; }
    }
    return { name, atoms, residues, hasHydrogens: atoms.some(a => a.el === 'H' || a.el === 'D') };
}

function _residueKind(rn) {
    if (WATER.has(rn)) return 'water';
    if (isAminoAcid(rn)) return 'aa';
    if (NUCLEIC.has(rn)) return 'na';
    return 'het';
}

const isHeavy = (a) => a.el !== 'H' && a.el !== 'D';

// ================================================================
// Ligand discovery
// ================================================================

/**
 * Candidate ligands: het groups (covalently linked ones merged) and short
 * peptide chains. Ordered so the first is the sensible default: the largest
 * drug-like het group, then peptides, then additives.
 */
export function findLigands(st) {
    const { atoms, residues } = st;
    const hets = residues.filter(r => r.kind === 'het' && r.atoms.some(i => isHeavy(atoms[i])));

    // Merge het residues joined by a covalent bond (glycans, split ligands)
    const parent = hets.map((_, k) => k);
    const find = (k) => (parent[k] === k ? k : (parent[k] = find(parent[k])));
    for (let p = 0; p < hets.length; p++) {
        for (let q = p + 1; q < hets.length; q++) {
            if (_residuesBonded(atoms, hets[p], hets[q])) parent[find(p)] = find(q);
        }
    }
    const groups = new Map();
    hets.forEach((r, k) => {
        const g = find(k);
        if (!groups.has(g)) groups.set(g, []);
        groups.get(g).push(r);
    });

    const out = [];
    for (const rs of groups.values()) {
        const first = rs[0];
        const resname = rs.map(r => r.resname).join('-');
        const atomIdx = rs.flatMap(r => r.atoms);
        const nHeavy = atomIdx.filter(i => isHeavy(atoms[i])).length;
        out.push({
            id: `${resname}:${first.chain}:${first.resnr}`,
            kind: 'small', resname, chain: first.chain, resnr: first.resnr,
            residues: rs.map(r => r.idx), atoms: atomIdx, nHeavy,
            additive: rs.every(r => ADDITIVES.has(r.resname)) || nHeavy < 4,
            label: `${resname} ${first.chain}${first.resnr} · ${nHeavy} heavy atoms`,
        });
    }

    // Short protein chains are peptide ligands when there is a longer receptor
    const chains = new Map();
    for (const r of residues) {
        if (r.kind !== 'aa') continue;
        if (!chains.has(r.chain)) chains.set(r.chain, []);
        chains.get(r.chain).push(r);
    }
    const longest = Math.max(0, ...[...chains.values()].map(rs => rs.length));
    for (const [chain, rs] of chains) {
        if (rs.length > 30 || rs.length === longest || longest < 40) continue;
        const atomIdx = rs.flatMap(r => r.atoms);
        out.push({
            id: `peptide:${chain}`, kind: 'peptide', resname: 'peptide', chain, resnr: rs[0].resnr,
            residues: rs.map(r => r.idx), atoms: atomIdx,
            nHeavy: atomIdx.filter(i => isHeavy(atoms[i])).length,
            additive: false,
            label: `Peptide chain ${chain} · ${rs.length} residues`,
        });
    }

    const rank = (l) => (l.additive ? 2 : (l.kind === 'peptide' ? 1 : 0));
    out.sort((a, b) => rank(a) - rank(b) || b.nHeavy - a.nHeavy);
    return out;
}

function _residuesBonded(atoms, r1, r2) {
    for (const i of r1.atoms) {
        const a = atoms[i];
        if (!isHeavy(a)) continue;
        for (const j of r2.atoms) {
            const b = atoms[j];
            if (!isHeavy(b)) continue;
            if (Math.abs(a.x - b.x) > 2 || Math.abs(a.y - b.y) > 2) continue;
            if (dist(a.xyz, b.xyz) < 1.95) return true;
        }
    }
    return false;
}

// ================================================================
// Bonds and spatial lookup
// ================================================================

function _grid(atoms, idxs, cell) {
    const map = new Map();
    for (const i of idxs) {
        const a = atoms[i];
        const k = `${Math.floor(a.x / cell)},${Math.floor(a.y / cell)},${Math.floor(a.z / cell)}`;
        let b = map.get(k);
        if (!b) map.set(k, (b = []));
        b.push(i);
    }
    return {
        near(p, r) {
            const out = [];
            const span = Math.ceil(r / cell);
            const cx = Math.floor(p[0] / cell), cy = Math.floor(p[1] / cell), cz = Math.floor(p[2] / cell);
            for (let dx = -span; dx <= span; dx++)
                for (let dy = -span; dy <= span; dy++)
                    for (let dz = -span; dz <= span; dz++) {
                        const b = map.get(`${cx + dx},${cy + dy},${cz + dz}`);
                        if (b) for (const j of b) out.push(j);
                    }
            return out;
        },
    };
}

/**
 * Distance bonds among `idxs` (OpenBabel's rule: sum of covalent radii + 0.45 Å),
 * then trimmed to each element's maximum valence, longest bonds dropped first.
 * `allow(i, j)` can veto a pair (used to keep receptor bonds within a residue
 * plus peptide and disulfide links).
 */
function _bonds(atoms, idxs, allow = null) {
    const grid = _grid(atoms, idxs, 2.5);
    const set = new Set(idxs);
    const adj = new Map(idxs.map(i => [i, []]));
    for (const i of idxs) {
        const a = atoms[i];
        const ri = COV_RADII[_elH(a.el)] ?? 0.77;
        for (const j of grid.near(a.xyz, 2.5)) {
            if (j <= i || !set.has(j)) continue;
            const b = atoms[j];
            if (!isHeavy(a) && !isHeavy(b)) continue;
            const d = dist(a.xyz, b.xyz);
            const lim = ri + (COV_RADII[_elH(b.el)] ?? 0.77) + 0.45;
            if (d > 0.4 && d <= lim && (!allow || allow(a, b))) {
                adj.get(i).push(j);
                adj.get(j).push(i);
            }
        }
    }
    // Valence trimming
    for (const i of idxs) {
        const a = atoms[i];
        const nb = adj.get(i);
        const max = isHeavy(a) ? (MAX_VALENCE[a.el] ?? 6) : 1;
        if (nb.length <= max) continue;
        nb.sort((p, q) => dist(a.xyz, atoms[p].xyz) - dist(a.xyz, atoms[q].xyz));
        for (const j of nb.splice(max)) {
            const back = adj.get(j);
            back.splice(back.indexOf(i), 1);
        }
    }
    return adj;
}

const _elH = (el) => (el === 'D' ? 'H' : el);

// ================================================================
// Ligand perception: rings, bond orders, hydrogens, charges
// ================================================================

const DOUBLE_MAX = { 'C-O': 1.30, 'C-N': 1.32, 'C-C': 1.40, 'C-S': 1.72, 'N-N': 1.30, 'N-O': 1.30,
                     'O-S': 1.55, 'O-P': 1.56, 'C-SE': 1.86 };
const TRIPLE_MAX = { 'C-N': 1.20, 'C-C': 1.25 };
const pairKey = (e1, e2) => (e1 < e2 ? `${e1}-${e2}` : `${e2}-${e1}`);
const bondKey = (i, j) => (i < j ? `${i}-${j}` : `${j}-${i}`);

/**
 * Type a small molecule from its coordinates. Returns Map(atomIdx -> info):
 *   el, heavy (neighbour idx), hx (explicit H idx), nH, arom, inRing, hyb,
 *   conj (N bonded to an sp2/aromatic centre, so OpenBabel calls it sp2),
 *   carbonylC, charge (OpenBabel pH 7.4 model), plus mol.rings / mol.bo.
 */
function _perceiveSmallMolecule(st, idxs) {
    const { atoms } = st;
    const adj = _bonds(atoms, idxs);
    const heavyIdx = idxs.filter(i => isHeavy(atoms[i]));
    const info = new Map();
    for (const i of heavyIdx) {
        const nb = adj.get(i);
        info.set(i, {
            i, el: atoms[i].el,
            heavy: nb.filter(j => isHeavy(atoms[j])),
            hx: nb.filter(j => !isHeavy(atoms[j])),
            nH: 0, arom: false, inRing: false, hyb: 3, conj: false, carbonylC: false, charge: 0,
        });
    }
    const X = (i) => atoms[i].xyz;
    const len = (i, j) => dist(X(i), X(j));

    // 1. Ring membership (any size): an atom is in a ring if one of its bonds is not a bridge
    const bridges = _bridges(heavyIdx, (i) => info.get(i).heavy);
    for (const [i, a] of info) {
        a.inRing = a.heavy.some(j => !bridges.has(bondKey(i, j)));
    }

    // 2. 5/6-membered cycles; aromatic if planar with short bonds (OpenBabel's
    //    aromaticity proxy), PLIP ring if aromatic or planar by PLIP's own 5° test
    const cycles = _cycles56(heavyIdx, (i) => info.get(i).heavy);
    const rings = [];
    for (const r of cycles) {
        const pts = r.map(X);
        const c = centroid(pts);
        const n = ringNormal(pts);
        const planeDev = Math.max(...pts.map(p => Math.abs(dot(sub(p, c), n))));
        let lenSum = 0;
        for (let k = 0; k < r.length; k++) lenSum += len(r[k], r[(k + 1) % r.length]);
        const meanLen = lenSum / r.length;
        const localNormals = r.map((a, k) => {
            const prev = r[(k + r.length - 1) % r.length], next = r[(k + 1) % r.length];
            return cross(sub(X(prev), X(a)), sub(X(next), X(a)));
        });
        let maxDev = 0;
        for (let p = 0; p < localNormals.length; p++)
            for (let q = p + 1; q < localNormals.length; q++)
                maxDev = Math.max(maxDev, planeAngle(localNormals[p], localNormals[q]));
        const heteroOk = r.every(a => ['C', 'N', 'O', 'S', 'SE', 'B', 'P'].includes(info.get(a).el));
        // Aromatic: flat, and no ring atom is plainly sp3 (four heavy neighbours,
        // a pyramidal centre, or nothing but long single bonds in the ring).
        // Lenient on bond lengths: crystal ligands at modest resolution and
        // predicted poses both bend aromatic rings more than textbook geometry.
        const sp3 = r.some((a, k) => {
            const A = info.get(a);
            if (A.heavy.length >= 4) return true;
            const prev = r[(k + r.length - 1) % r.length], next = r[(k + 1) % r.length];
            if (A.heavy.length === 3) {
                let sum = 0;
                for (let p = 0; p < 3; p++) for (let q = p + 1; q < 3; q++)
                    sum += angleDeg(sub(X(A.heavy[p]), X(a)), sub(X(A.heavy[q]), X(a)));
                if (sum < 345) return true;
            }
            return A.el === 'C' && len(a, prev) >= 1.47 && len(a, next) >= 1.47;
        });
        const aromatic = heteroOk && !sp3 && planeDev <= 0.25 && meanLen <= 1.47;
        const planar = maxDev <= PLIP_CONFIG.AROMATIC_PLANARITY;
        rings.push({ atoms: r, center: c, normal: n, aromatic, plipRing: aromatic || planar });
    }
    const aromBond = new Set();
    for (const r of rings) {
        if (!r.aromatic) continue;
        for (let k = 0; k < r.atoms.length; k++) {
            const a = r.atoms[k], b = r.atoms[(k + 1) % r.atoms.length];
            info.get(a).arom = true;
            aromBond.add(bondKey(a, b));
        }
    }

    // 3. Hybridisation from heavy-atom geometry
    for (const [i, a] of info) {
        const k = a.heavy.length + a.hx.length;
        if (a.arom) { a.hyb = 2; continue; }
        const nb = a.heavy.concat(a.hx);
        if (nb.length >= 4) { a.hyb = 3; continue; }
        if (nb.length >= 2) {
            const angs = [];
            for (let p = 0; p < nb.length; p++)
                for (let q = p + 1; q < nb.length; q++)
                    angs.push(angleDeg(sub(X(nb[p]), X(i)), sub(X(nb[q]), X(i))));
            if (nb.length === 3) a.hyb = angs.reduce((s, v) => s + v, 0) >= 350 ? 2 : 3;
            else a.hyb = angs[0] >= 160 ? 1 : (angs[0] >= 115 ? 2 : 3);
            continue;
        }
        if (k === 1) {
            const j = a.heavy[0] ?? a.hx[0];
            const d = len(i, j);
            const key = pairKey(a.el, atoms[j].el);
            if (TRIPLE_MAX[key] && d <= TRIPLE_MAX[key]) a.hyb = 1;
            else if (DOUBLE_MAX[key] && d <= DOUBLE_MAX[key]) a.hyb = 2;
            else a.hyb = 3;
        }
    }

    // 4. Double and triple bonds outside aromatic rings, most double-like first,
    //    within each atom's capacity
    const bo = new Map();
    for (const [i, a] of info) for (const j of a.heavy) if (i < j) bo.set(bondKey(i, j), aromBond.has(bondKey(i, j)) ? 1.5 : 1);
    const terminalO = (i) => info.get(i).heavy.filter(j => info.get(j).el === 'O' && info.get(j).heavy.length === 1 && !info.get(j).hx.length);
    const capacity = new Map();
    for (const [i, a] of info) {
        const h = a.heavy.length;
        let cap = 0;
        if (a.el === 'C') cap = a.arom ? 1 : (h >= 4 ? 0 : (a.hyb === 1 ? 2 : 1));
        else if (a.el === 'N') cap = a.arom ? 0 : (h <= 2 ? (a.hyb === 1 ? 2 : 1) : (terminalO(i).length ? 1 : 0));
        else if (a.el === 'O') cap = h === 1 && !a.hx.length ? 1 : 0;
        else if (a.el === 'S') cap = a.arom ? 0 : Math.min(2, Math.max(0, 6 - h - 2));
        else if (a.el === 'P') cap = 1;
        else if (a.el === 'SE') cap = 1;
        capacity.set(i, cap);
    }
    const cands = [];
    for (const [k, order] of bo) {
        if (order !== 1) continue;
        const [i, j] = k.split('-').map(Number);
        const ai = info.get(i), aj = info.get(j);
        if (ai.arom && aj.arom) continue;                       // both in rings: aromatic bonds only
        if ((ai.arom || aj.arom) && !['O', 'S', 'N'].includes(ai.arom ? aj.el : ai.el)) continue;
        const key = pairKey(ai.el, aj.el);
        const d = len(i, j);
        if (TRIPLE_MAX[key] && d <= TRIPLE_MAX[key]) cands.push({ k, i, j, order: 3, score: d - TRIPLE_MAX[key] - 1 });
        else if (DOUBLE_MAX[key] && d <= DOUBLE_MAX[key]) cands.push({ k, i, j, order: 2, score: d - DOUBLE_MAX[key] });
    }
    cands.sort((p, q) => p.score - q.score);
    for (const c of cands) {
        const need = c.order - 1;
        if (capacity.get(c.i) >= need && capacity.get(c.j) >= need) {
            bo.set(c.k, c.order);
            capacity.set(c.i, capacity.get(c.i) - need);
            capacity.set(c.j, capacity.get(c.j) - need);
        }
    }
    const order = (i, j) => bo.get(bondKey(i, j)) || 1;
    const hasMultiple = (i, except = -1) => info.get(i).heavy.some(j => j !== except && order(i, j) >= 2);

    for (const [i, a] of info) {
        if (a.el === 'C') a.carbonylC = a.heavy.some(j => ['O', 'S'].includes(info.get(j).el) && order(i, j) === 2);
    }

    // 5. Hydrogen counts (OpenBabel AddPolarHydrogens: neutral valence)
    for (const [i, a] of info) {
        if (a.hx.length) { a.nH = a.hx.length; continue; }
        const sumBo = a.heavy.reduce((s, j) => s + order(i, j), 0);
        if (a.el === 'N') {
            if (a.arom) a.nH = 0;                               // settled per ring below
            else if (terminalO(i).length >= 2 || a.heavy.length >= 4) a.nH = 0;
            else a.nH = Math.max(0, Math.round(3 - sumBo));
        } else if (a.el === 'O') {
            a.nH = a.arom ? 0 : Math.max(0, Math.round(2 - sumBo));
        } else if (a.el === 'S' || a.el === 'SE') {
            a.nH = a.arom || a.heavy.length >= 3 ? 0 : Math.max(0, Math.round(2 - sumBo));
        } else if (a.el === 'C') {
            a.nH = Math.max(0, Math.round(4 - (a.arom ? a.heavy.length + 1 : sumBo)));
        }
    }
    _aromaticNH(info, rings, X, order);
    for (const a of info.values()) a.nH0 = a.nH;          // neutral valence, what PLIP sees

    // 6. Conjugation: a 3-connected N next to an aromatic or multiply bonded C/N
    //    is sp2 to OpenBabel (amide, aniline, enamine, urea...); next to S/P it is not
    for (const [i, a] of info) {
        if (a.el !== 'N') continue;
        a.conj = a.arom || hasMultiple(i) || a.heavy.some(j => {
            const b = info.get(j);
            return b.arom || (['C', 'N'].includes(b.el) && hasMultiple(j, i));
        });
    }

    // 7. Charges at pH 7.4, following OpenBabel's phmodel.txt
    _phModel(info, rings, order, terminalO);

    return { info, rings, bo, order, adj };
}

/** Tarjan bridge finding over the heavy-atom graph. */
function _bridges(nodes, nbrs) {
    const disc = new Map(), low = new Map(), out = new Set();
    let t = 0;
    const visit = (u, parent) => {
        disc.set(u, t); low.set(u, t); t++;
        for (const v of nbrs(u)) {
            if (v === parent) continue;
            if (!disc.has(v)) {
                visit(v, u);
                low.set(u, Math.min(low.get(u), low.get(v)));
                if (low.get(v) > disc.get(u)) out.add(bondKey(u, v));
            } else {
                low.set(u, Math.min(low.get(u), disc.get(v)));
            }
        }
    };
    for (const n of nodes) if (!disc.has(n)) visit(n, -1);
    return out;
}

/**
 * All simple 5- and 6-membered cycles, deduplicated by atom set. Bounded:
 * a clashing pose can bond everything to everything, and then cycle
 * enumeration explodes; carbon cannot exceed four neighbours, so a denser
 * graph is not chemistry and ring detection is skipped.
 */
function _cycles56(nodes, nbrs) {
    if (nodes.some(n => nbrs(n).length > 6)) return [];
    const seen = new Set(), out = [];
    let steps = 0;
    const walk = (start, path, visited) => {
        if (out.length >= 80 || ++steps > 300000) return;
        const last = path[path.length - 1];
        for (const nb of nbrs(last)) {
            if (nb === start && path.length >= 5) {
                const key = [...path].sort((a, b) => a - b).join(',');
                if (!seen.has(key)) { seen.add(key); out.push([...path]); }
            } else if (!visited.has(nb) && nb > start && path.length < 6) {
                visited.add(nb); path.push(nb);
                walk(start, path, visited);
                path.pop(); visited.delete(nb);
            }
        }
    };
    for (const s of nodes) if (nbrs(s).length >= 2) walk(s, [s], new Set([s]));
    return out;
}

/**
 * Which aromatic N carry a hydrogen. 6-rings: pyridine-type unless next to an
 * exocyclic C=O (lactam). 5-rings: one atom donates the lone pair; if no O, S
 * or substituted N does, the deg-2 N with the longest ring bonds is the NH.
 */
function _aromaticNH(info, rings, X, order) {
    for (const r of rings) {
        if (!r.aromatic) continue;
        const ns = r.atoms.filter(a => info.get(a).el === 'N' && info.get(a).heavy.length === 2 && !info.get(a).hx.length);
        if (!ns.length) continue;
        if (r.atoms.length === 6) {
            for (const n of ns) {
                const lactam = info.get(n).heavy.some(c => info.get(c).heavy.some(o =>
                    !r.atoms.includes(o) && ['O', 'S'].includes(info.get(o).el) && order(c, o) === 2));
                if (lactam) info.get(n).nH = 1;
            }
            continue;
        }
        const donor = r.atoms.some(a => {
            const b = info.get(a);
            return ['O', 'S', 'SE'].includes(b.el) || (b.el === 'N' && (b.heavy.length === 3 || b.hx.length || b.nH > 0));
        });
        if (donor) continue;
        let best = null, bestLen = -1;
        for (const n of ns) {
            const ringNb = info.get(n).heavy.filter(j => r.atoms.includes(j));
            const m = ringNb.reduce((s, j) => s + dist(X(n), X(j)), 0) / ringNb.length;
            if (m > bestLen) { bestLen = m; best = n; }
        }
        if (best !== null) info.get(best).nH = 1;
    }
}

/** OpenBabel's pH 7.4 transforms that matter for ligands. */
function _phModel(info, rings, order, terminalO) {
    for (const [i, a] of info) {
        // carboxylic, sulfonic, sulfinic, phosphonic acids lose the proton
        if (['C', 'S', 'P'].includes(a.el)) {
            const tO = terminalO(i);
            const hasDouble = tO.some(o => order(i, o) === 2);
            if (hasDouble) {
                for (const o of tO) {
                    if (order(i, o) === 1 && info.get(o).nH > 0) { info.get(o).nH = 0; info.get(o).charge = -1; }
                }
            }
        }
        // aliphatic sp3 amine bonded only to carbon: protonated (pKa 10)
        if (a.el === 'N' && !a.arom && !a.conj && a.heavy.length <= 3 &&
            a.heavy.every(j => info.get(j).el === 'C') && a.heavy.length + a.nH <= 3) {
            a.charge = 1; a.nH += 1;
        }
        if (a.el === 'N' && a.heavy.length === 4) a.charge = 1;
        // amidine / guanidine: C(=N)(N) protonated on the imine N (pKa 12.5)
        if (a.el === 'C' && !a.arom) {
            const ns = a.heavy.filter(j => info.get(j).el === 'N' && !info.get(j).arom);
            const imine = ns.find(j => order(i, j) === 2);
            if (imine !== undefined && ns.some(j => j !== imine && order(i, j) === 1 && !terminalO(j).length) &&
                info.get(imine).charge === 0) {
                info.get(imine).charge = 1; info.get(imine).nH += 1;
            }
        }
        // nitro: charge-separated N+ / O-
        if (a.el === 'N' && terminalO(i).length >= 2) {
            a.charge = 1;
            const o = terminalO(i).find(o => order(i, o) === 1);
            if (o !== undefined) info.get(o).charge = -1;
        }
    }
    // tetrazole: the NH becomes the anion
    for (const r of rings) {
        if (r.atoms.length !== 5) continue;
        const ns = r.atoms.filter(a => info.get(a).el === 'N');
        if (ns.length !== 4) continue;
        const nh = ns.find(n => info.get(n).nH > 0 && info.get(n).heavy.length === 2);
        if (nh !== undefined) { info.get(nh).nH = 0; info.get(nh).charge = -1; }
    }
}

// ================================================================
// Receptor typing (residue templates)
// ================================================================

// PLIP: carbons whose neighbours are only C/H
const PLIP_PHOB = {
    ALA: ['CB'], VAL: ['CB', 'CG1', 'CG2'], LEU: ['CB', 'CG', 'CD1', 'CD2'], ILE: ['CB', 'CG1', 'CG2', 'CD1'],
    MET: ['CB'], PHE: ['CB', 'CG', 'CD1', 'CD2', 'CE1', 'CE2', 'CZ'],
    TRP: ['CB', 'CG', 'CD2', 'CE3', 'CZ2', 'CZ3', 'CH2'], TYR: ['CB', 'CG', 'CD1', 'CD2', 'CE1', 'CE2'],
    PRO: ['CB', 'CG'], LYS: ['CB', 'CG', 'CD'], ARG: ['CB', 'CG'], THR: ['CG2'], CYS: [],
    GLU: ['CB', 'CG'], GLN: ['CB', 'CG'], ASP: ['CB'], ASN: ['CB'], HIS: ['CB'], SER: [], GLY: [],
};
// ProLIF's hydrophobic SMARTS on a protonated residue
const PROLIF_PHOB = {
    ALA: [], VAL: ['CB'], LEU: ['CB', 'CG'], ILE: ['CB', 'CG1'], MET: ['CB', 'CG', 'SD'],
    PHE: ['CB', 'CG', 'CD1', 'CD2', 'CE1', 'CE2', 'CZ'], TRP: ['CB', 'CG', 'CD2', 'CE3', 'CZ2', 'CZ3', 'CH2'],
    TYR: ['CB', 'CG', 'CD1', 'CD2', 'CE1', 'CE2'], PRO: [], LYS: ['CB', 'CG', 'CD'], ARG: ['CB', 'CG'],
    THR: [], CYS: ['CB'], GLU: ['CB', 'CG'], GLN: ['CB', 'CG'], ASP: ['CB'], ASN: ['CB'], HIS: ['CB'],
    SER: [], GLY: [],
};
// Side-chain polar atoms: [donor H count at pH 7.4, acceptor?]
const SIDECHAIN_POLAR = {
    ARG: { NE: [1, 0], NH1: [2, 0], NH2: [2, 0] }, LYS: { NZ: [3, 0] },
    ASN: { OD1: [0, 1], ND2: [2, 0] }, GLN: { OE1: [0, 1], NE2: [2, 0] },
    ASP: { OD1: [0, 1], OD2: [0, 1] }, GLU: { OE1: [0, 1], OE2: [0, 1] },
    SER: { OG: [1, 1] }, THR: { OG1: [1, 1] }, TYR: { OH: [1, 1] }, TRP: { NE1: [1, 0] },
    HIS: { ND1: [1, 1], NE2: [1, 1] },          // tautomer unknown without H: both ways
    CYS: { SG: [1, 0] }, MET: { SD: [0, 0] },
};
const RING_ATOMS = {
    PHE: [['CG', 'CD1', 'CE1', 'CZ', 'CE2', 'CD2']], TYR: [['CG', 'CD1', 'CE1', 'CZ', 'CE2', 'CD2']],
    HIS: [['CG', 'ND1', 'CE1', 'NE2', 'CD2']],
    TRP: [['CD2', 'CE2', 'CZ2', 'CH2', 'CZ3', 'CE3'], ['CG', 'CD1', 'NE1', 'CE2', 'CD2']],
};
const PLIP_POS = { ARG: ['NE', 'NH1', 'NH2'], LYS: ['NZ'], HIS: ['ND1', 'NE2'] };
const PLIP_NEG = { ASP: ['OD1', 'OD2'], GLU: ['OE1', 'OE2'] };
const BACKBONE = new Set(['N', 'CA', 'C', 'O', 'OXT']);
const MET_SE = (rn, name) => (rn === 'MSE' && name === 'SE' ? 'SD' : name);

/**
 * Per-atom info for amino-acid residues from templates, in the same shape the
 * small-molecule perception produces, so the rule sets can treat both alike.
 */
function _typeResidues(st, resIdxs, adj) {
    const { atoms, residues } = st;
    const info = new Map();
    const resAtoms = new Map();
    for (const ri of resIdxs) {
        const r = residues[ri];
        const rn = canonicalResname(r.resname);
        const byName = {};
        for (const i of r.atoms) if (isHeavy(atoms[i])) byName[MET_SE(r.resname, atoms[i].name)] = i;
        resAtoms.set(ri, { rn, byName });
        const hisH = rn === 'HIS' ? _hisTautomer(atoms, byName, adj) : null;
        for (const [name, i] of Object.entries(byName)) {
            const nb = adj.get(i) || [];
            const a = {
                i, el: atoms[i].el, name, rn, res: ri,
                heavy: nb.filter(j => isHeavy(atoms[j])),
                hx: nb.filter(j => !isHeavy(atoms[j])),
                nH: 0, don: false, acc: false, arom: false, charge: 0,
                backbone: BACKBONE.has(name),
            };
            if (name === 'N') {
                // a chain break in a cropped pose is not a terminus
                const nterm = residues[ri].nterm && !a.heavy.some(j => atoms[j].name === 'C' && atoms[j].res !== ri);
                a.nH = nterm ? 3 : (rn === 'PRO' ? 0 : 1);
                a.don = a.nH > 0;
                if (nterm) a.charge = 1;
            } else if (name === 'O') {
                a.acc = true;
            } else if (name === 'OXT') {
                a.acc = true; a.charge = -1;
            }
            const sc = SIDECHAIN_POLAR[rn]?.[name];
            if (sc) {
                a.nH = sc[0]; a.acc = !!sc[1]; a.don = sc[0] > 0;
                if (hisH && (name === 'ND1' || name === 'NE2')) {
                    a.nH = hisH[name] ? 1 : 0; a.don = !!hisH[name]; a.acc = !hisH[name] || hisH.both;
                }
                if (rn === 'CYS' && a.heavy.some(j => atoms[j].el === 'S' && atoms[j].res !== ri)) { a.nH = 0; a.don = false; }
            }
            if (a.hx.length) { a.nH = a.hx.length; a.don = a.nH > 0 && ['N', 'O', 'S'].includes(a.el); }
            if (rn === 'LYS' && name === 'NZ') a.charge = 1;
            if (rn === 'ARG' && ['NE', 'NH1', 'NH2'].includes(name)) a.charge = 1;
            if (rn === 'ASP' && ['OD1', 'OD2'].includes(name)) a.charge = -1;
            if (rn === 'GLU' && ['OE1', 'OE2'].includes(name)) a.charge = -1;
            if (rn === 'HIS' && hisH?.both && ['ND1', 'NE2'].includes(name)) a.charge = 1;
            if (RING_ATOMS[rn]?.some(ring => ring.includes(name))) a.arom = true;
            info.set(i, a);
        }
        // the C-terminal carbonyl O shares the carboxylate charge
        if (byName.OXT !== undefined && byName.O !== undefined) info.get(byName.O).charge = -1;
    }
    return { info, resAtoms };
}

/** His protonation from explicit H; null when the file has none. */
function _hisTautomer(atoms, byName, adj) {
    const hOn = (n) => byName[n] !== undefined && (adj.get(byName[n]) || []).some(j => !isHeavy(atoms[j]));
    if (byName.ND1 === undefined || byName.NE2 === undefined) return null;
    const d = hOn('ND1'), e = hOn('NE2');
    if (!d && !e) return null;
    return { ND1: d, NE2: e, both: d && e };
}

// ================================================================
// Donor hydrogens: explicit, or placed from heavy-atom geometry
// ================================================================

const XH_LEN = { N: 1.01, O: 0.96, S: 1.34 };

/**
 * Hydrogen geometry for a donor atom: {fixed: [[x,y,z]...]} or a rotor
 * {axisFrom, d, len, cone} whose best hydrogen is chosen per acceptor.
 */
function _donorGeometry(st, a, hyb) {
    const { atoms } = st;
    const D = atoms[a.i].xyz;
    if (a.hx.length) return { fixed: a.hx.map(j => atoms[j].xyz), explicit: true };
    const L = XH_LEN[a.el] || 1.0;
    const nb = a.heavy.map(j => atoms[j].xyz);
    const u = nb.map(p => unit(sub(p, D)));
    if (nb.length === 0) return { free: true, len: L };
    if (nb.length >= 3) return { fixed: [add(D, scale(unit(scale(u.reduce(add), -1)), L))] };
    if (nb.length === 2) {
        const bis = unit(scale(add(u[0], u[1]), -1));
        if (hyb !== 3) return { fixed: [add(D, scale(bis, L))] };
        const n = unit(cross(u[0], u[1]));
        const c = Math.cos(54.75 / DEG), s = Math.sin(54.75 / DEG);
        const hs = [add(D, scale(unit(add(scale(bis, c), scale(n, s))), L)),
                    add(D, scale(unit(sub(scale(bis, c), scale(n, s))), L))];
        return { fixed: hs };
    }
    // one heavy neighbour X
    const Xi = a.heavy[0];
    if (hyb === 2) {
        // in the plane of X and one of X's other neighbours, 120° from D→X
        const Y = (st._adj.get(Xi) || []).find(j => j !== a.i && isHeavy(atoms[j]));
        if (Y !== undefined) {
            const ux = u[0];
            const w0 = sub(sub(atoms[Y].xyz, atoms[Xi].xyz), scale(ux, dot(sub(atoms[Y].xyz, atoms[Xi].xyz), ux)));
            const w = unit(w0);
            const ang = (a.el === 'O' ? 109.5 : 120) / DEG;
            const dir1 = add(scale(ux, Math.cos(ang)), scale(w, Math.sin(ang)));
            const dir2 = sub(scale(ux, Math.cos(ang)), scale(w, Math.sin(ang)));
            return { fixed: [add(D, scale(unit(dir1), L)), add(D, scale(unit(dir2), L))] };
        }
    }
    return { rotor: true, axis: scale(u[0], -1), len: L, cone: (180 - 109.5) / DEG };
}

/**
 * Hydrogens where OpenBabel's AddPolarHydrogens puts them (OBBuilder::
 * GetNewBondVector). This is what PLIP works with on a pose that has no
 * hydrogens and no CCD template for PDBFixer to use, i.e. any Boltz ligand.
 * OpenBabel lists an atom's neighbours in ascending z (ConnectTheDots sorts
 * by z), and each new hydrogen is placed relative to the neighbours so far.
 */
function _openBabelHydrogens(st, info, a, hyb, nH) {
    const { atoms } = st;
    const D = atoms[a.i].xyz;
    const L = XH_LEN[a.el] || 1.0;
    const byZ = (p, q) => atoms[p].z - atoms[q].z;
    const nbrs = a.heavy.slice().sort(byZ).map(j => atoms[j].xyz);
    const placed = [];
    for (let h = 0; h < nH; h++) {
        const cur = nbrs.concat(placed);
        let dir;
        if (cur.length === 0) {
            dir = [0, 0, 1];
        } else if (cur.length === 1) {
            const b1 = unit(sub(D, cur[0]));
            const Xi = a.heavy[0];
            const xNbrs = Xi === undefined ? [] : (info.get(Xi)?.heavy || []).filter(j => j !== a.i).sort(byZ);
            let b2 = xNbrs.length ? unit(sub(atoms[Xi].xyz, atoms[xNbrs[xNbrs.length - 1]].xyz)) : null;
            if (!b2 || norm(cross(b1, b2)) < 1e-3) b2 = unit(cross(b1, Math.abs(b1[1]) < 0.9 ? [0, 1, 0] : [1, 0, 0]));
            const v2 = unit(cross(b1, cross(b1, b2)));
            dir = hyb === 1 ? b1 : sub(b1, scale(v2, Math.tan((hyb === 2 ? 60 : 70.5) / DEG)));
        } else if (cur.length === 2) {
            const b1 = unit(sub(D, cur[0])), b2 = unit(sub(D, cur[1]));
            const v1 = unit(add(b1, b2));
            dir = hyb === 2 ? v1 : add(unit(cross(b1, b2)), scale(v1, Math.tan(35.25 / DEG)));
        } else {
            dir = cur.reduce((s0, p) => add(s0, unit(sub(D, p))), [0, 0, 0]);
        }
        placed.push(add(D, scale(unit(dir), L)));
    }
    return { fixed: placed };
}

/** OpenBabel's hybridisation of a hydroxyl O: sp2 next to an aromatic or multiply bonded atom. */
function _obHydroxylHyb(a, info, mol) {
    const x = a.heavy[0];
    if (x === undefined) return 3;
    const X = info.get(x);
    return X.arom || X.heavy.some(k => mol.order(x, k) >= 2) ? 2 : 3;
}

/** The donor hydrogen that best serves an acceptor at A: {h, angle} (angle D-H...A at H). */
function _bestH(D, geo, A) {
    if (geo.free) {
        const h = add(D, scale(unit(sub(A, D)), geo.len));
        return { h, angle: 180 };
    }
    let cands;
    if (geo.fixed) cands = geo.fixed;
    else {
        const v = sub(A, D);
        let p = sub(v, scale(geo.axis, dot(v, geo.axis)));
        if (norm(p) < 1e-6) p = cross(geo.axis, Math.abs(geo.axis[0]) < 0.9 ? [1, 0, 0] : [0, 1, 0]);
        const dir = add(scale(geo.axis, Math.cos(geo.cone)), scale(unit(p), Math.sin(geo.cone)));
        cands = [add(D, scale(dir, geo.len))];
    }
    let best = null;
    for (const h of cands) {
        const ang = angleDeg(sub(D, h), sub(A, h));
        if (!best || ang > best.angle) best = { h, angle: ang };
    }
    return best;
}

// ================================================================
// Preparation: binding site, typing for both sides
// ================================================================

function _prepare(st, lig, opts) {
    const { atoms, residues } = st;
    const ligSet = new Set(lig.atoms);
    const ligHeavy = lig.atoms.filter(i => isHeavy(atoms[i]));
    const ligPts = ligHeavy.map(i => atoms[i].xyz);
    const ligCenter = centroid(ligPts);
    const maxToCenter = Math.max(...ligPts.map(p => dist(p, ligCenter)));

    // Receptor: amino-acid residues (not the ligand's) with an atom within BS_DIST
    const ligGrid = _grid(atoms, ligHeavy, PLIP_CONFIG.BS_DIST);
    const ligRes = new Set(lig.residues);
    const bs = [];
    for (const r of residues) {
        if (r.kind !== 'aa' || ligRes.has(r.idx)) continue;
        let hit = false;
        for (const i of r.atoms) {
            const a = atoms[i];
            if (!isHeavy(a)) continue;
            for (const j of ligGrid.near(a.xyz, PLIP_CONFIG.BS_DIST)) {
                if (dist(a.xyz, atoms[j].xyz) <= PLIP_CONFIG.BS_DIST) { hit = true; break; }
            }
            if (hit) break;
        }
        if (hit) bs.push(r.idx);
    }
    // Bond context: binding-site residues plus their sequence neighbours
    const ctxRes = new Set();
    for (const ri of bs.concat(lig.kind === 'peptide' ? lig.residues : [])) {
        ctxRes.add(ri);
        for (const k of [ri - 1, ri + 1]) if (residues[k] && residues[k].chain === residues[ri].chain && residues[k].kind === 'aa') ctxRes.add(k);
    }
    const ctxAtoms = [...ctxRes].flatMap(ri => residues[ri].atoms);
    const recAdj = _bonds(atoms, ctxAtoms, (a, b) => {
        if (a.res === b.res) return true;
        const names = [a.name, b.name].sort().join('-');
        return names === 'C-N' || (a.name === 'SG' && b.name === 'SG');
    });
    st._adj = recAdj;
    const rec = _typeResidues(st, bs, recAdj);

    let ligTyped;
    if (lig.kind === 'peptide') {
        ligTyped = _typeResidues(st, lig.residues, recAdj);
        ligTyped.peptide = true;
    } else {
        const mol = _perceiveSmallMolecule(st, lig.atoms);
        for (const [k, v] of mol.adj) recAdj.set(k, v);
        ligTyped = { info: mol.info, mol, peptide: false };
    }

    // Waters near the ligand (PLIP: within max_dist_to_center + BS_DIST of its centroid)
    const waters = [];
    for (const r of residues) {
        if (r.kind !== 'water') continue;
        const o = r.atoms.find(i => atoms[i].el === 'O');
        if (o !== undefined && dist(atoms[o].xyz, ligCenter) < maxToCenter + PLIP_CONFIG.BS_DIST) waters.push(o);
    }
    return { st, lig, ligSet, ligHeavy, bs, rec, ligTyped, waters, ligCenter, ligandH: opts.ligandH || 'openbabel' };
}

// ================================================================
// Rule set: PLIP
// ================================================================

function _plipLigandFeatures(ctx) {
    const { st, ligTyped } = ctx;
    const { atoms } = st;
    if (ligTyped.peptide) return _plipResidueFeatures(ctx, ligTyped);
    const { info, mol } = ligTyped;
    const el = (i) => info.get(i).el;
    const F = { hydrophobic: [], donors: [], acceptors: [], pos: [], neg: [], rings: [], halDon: [] };

    for (const [i, a] of info) {
        const nH = a.nH0;
        if (a.el === 'C' && a.heavy.every(j => el(j) === 'C')) F.hydrophobic.push(i);
        if (['N', 'O'].includes(a.el) && nH > 0) {
            const hyb = a.el === 'O' ? _obHydroxylHyb(a, info, mol) : _obHyb({ ...a, nH });
            const geo = a.hx.length || ctx.ligandH !== 'openbabel'
                ? _donorGeometry(st, { ...a, nH }, hyb)
                : _openBabelHydrogens(st, info, a, hyb, nH);
            F.donors.push({ d: i, geo });
        }
        if (_obAcceptor(a, nH, info)) F.acceptors.push(i);
        if (HALOGENS.has(a.el)) {
            const cs = a.heavy.filter(j => el(j) === 'C');
            if (cs.length === 1) F.halDon.push({ x: i, c: cs[0] });
        }
        // PLIP's functional groups, on the H-free ligand
        const n = a.heavy.map(el);
        const count = (e) => n.filter(x => x === e).length;
        if (a.el === 'N') {
            if (a.heavy.length === 4) F.pos.push({ atoms: [i], center: atoms[i].xyz, fgroup: 'quartamine' });
            else if (!a.arom && !a.conj && a.heavy.length >= 3) F.pos.push({ atoms: [i], center: atoms[i].xyz, fgroup: 'tertamine', amine: i });
        }
        if (a.el === 'S') {
            if (a.heavy.length === 3) F.pos.push({ atoms: [i], center: atoms[i].xyz, fgroup: 'sulfonium' });
            else if (count('O') === 3) F.neg.push({ atoms: [i, ...a.heavy.filter(j => el(j) === 'O')], center: atoms[i].xyz, fgroup: 'sulfonicacid' });
            else if (count('O') === 4) F.neg.push({ atoms: [i, ...a.heavy], center: atoms[i].xyz, fgroup: 'sulfate' });
        }
        if (a.el === 'P' && n.length && n.every(x => x === 'O')) {
            F.neg.push({ atoms: [i, ...a.heavy], center: atoms[i].xyz, fgroup: 'phosphate' });
        }
        if (a.el === 'C') {
            if (count('O') === 2 && count('C') === 1) {
                const os = a.heavy.filter(j => el(j) === 'O');
                F.neg.push({ atoms: os, center: centroid(os.map(j => atoms[j].xyz)), fgroup: 'carboxylate' });
            } else if (count('N') === 3 && n.length === 3 &&
                       Math.min(...a.heavy.map(j => info.get(j).heavy.length)) === 1) {
                F.pos.push({ atoms: a.heavy.slice(), center: atoms[i].xyz, fgroup: 'guanidine' });
            }
        }
    }
    for (const r of mol.rings) {
        if (!r.plipRing) continue;
        F.rings.push({ atoms: r.atoms, center: r.center, normal: r.normal, type: `${r.atoms.length}-membered` });
    }
    return F;
}

/** OpenBabel's hybridisation for a ligand atom. */
const _obHyb = (a) => (a.arom ? 2 : (a.el === 'N' ? (a.conj ? 2 : (a.hyb === 1 ? 1 : (a.heavy.length + a.nH <= 2 ? a.hyb : 3))) : a.hyb));

/** OpenBabel 3.2 OBAtom::IsHbondAcceptor, as probed on representative groups. */
function _obAcceptor(a, nH, info) {
    if (a.el === 'O') {
        if (nH > 0) return true;
        if (a.heavy.length === 0) return true;
        if (a.heavy.length === 1) {
            const p = info.get(a.heavy[0]);
            const pO = p.heavy.filter(j => info.get(j).el === 'O' && info.get(j).heavy.length === 1);
            if (p.el === 'N' && pO.length >= 2) return false;                       // nitro
            if (p.el === 'S' && pO.length === 2 && p.heavy.length === 4 &&
                p.heavy.filter(j => info.get(j).el === 'C').length === 2) return false;   // sulfone
            return true;
        }
        if (a.arom) return false;                                                  // furan, oxazole
        return !a.heavy.some(j => info.get(j).carbonylC);                           // ester, carbamate
    }
    if (a.el === 'N') {
        const conns = a.heavy.length + nH;
        if (conns >= 4) return false;
        if (conns === 3 && (a.arom || a.conj)) return false;
        return true;
    }
    return false;
}

/** PLIP features for amino-acid residues (receptor, or a peptide ligand). */
function _plipResidueFeatures(ctx, typed) {
    const { st } = ctx;
    const { atoms } = st;
    const F = { hydrophobic: [], donors: [], acceptors: [], pos: [], neg: [], rings: [], halAcc: [] };
    for (const [ri, { rn, byName }] of typed.resAtoms) {
        for (const name of PLIP_PHOB[rn] || []) if (byName[name] !== undefined) F.hydrophobic.push(byName[name]);
        for (const [grp, list] of [[PLIP_POS, F.pos], [PLIP_NEG, F.neg]]) {
            const names = grp[rn];
            if (!names) continue;
            const ids = names.map(n => byName[n]).filter(i => i !== undefined);
            if (ids.length) list.push({ atoms: ids, center: centroid(ids.map(i => atoms[i].xyz)), res: ri, fgroup: rn });
        }
        for (const ring of RING_ATOMS[rn] || []) {
            const ids = ring.map(n => byName[n]).filter(i => i !== undefined);
            if (ids.length < 5) continue;
            const pts = ids.map(i => atoms[i].xyz);
            F.rings.push({ atoms: ids, center: centroid(pts), normal: ringNormal(pts), res: ri, type: `${ids.length}-membered` });
        }
    }
    for (const [i, a] of typed.info) {
        // PLIP's donors are N/O with hydrogens (OpenBabel never counts S-H)
        if (a.don && a.el !== 'S') F.donors.push({ d: i, geo: _donorGeometry(st, a, _residueHyb(a)) });
        if (a.acc) F.acceptors.push(i);
        if (['O', 'N', 'S'].includes(a.el)) {
            const ys = a.heavy.filter(j => ['C', 'N', 'P', 'S'].includes(atoms[j].el));
            if (ys.length === 1) F.halAcc.push({ o: i, y: ys[0] });
        }
    }
    return F;
}

/** Hybridisation of a residue donor, for hydrogen placement. */
function _residueHyb(a) {
    if (a.name === 'N') return a.nH >= 3 ? 3 : 2;
    if (['LYS', 'SER', 'THR', 'CYS'].includes(a.rn)) return 3;
    return 2;      // Arg, Asn, Gln, His, Trp, Tyr OH (in the ring plane)
}

const _resOf = (st, i) => st.residues[st.atoms[i].res];

function _record(st, type, protAtoms, ligAtoms, d, detail, extra = {}) {
    const r = _resOf(st, protAtoms[0]);
    const pts = (ids) => centroid(ids.map(i => st.atoms[i].xyz));
    return {
        type, chain: r.chain, resnr: r.resnr, icode: r.icode, restype: r.resname,
        dist: Math.round(d * 100) / 100,
        prot: protAtoms, lig: ligAtoms,
        protPt: extra.protPt || pts(protAtoms), ligPt: extra.ligPt || pts(ligAtoms),
        detail, ...extra.more,
    };
}

function _plipDetect(ctx, opts) {
    const { st } = ctx;
    const { atoms } = st;
    const C = PLIP_CONFIG;
    const L = _plipLigandFeatures(ctx);
    const R = _plipResidueFeatures(ctx, ctx.rec);
    const X = (i) => atoms[i].xyz;
    const heavyHB = opts.hbondModel === 'heavy';
    const out = [];

    // Salt bridges
    const sbLneg = [], sbPneg = [];
    for (const pc of R.pos) for (const nc of L.neg) {
        const d = dist(pc.center, nc.center);
        if (d > C.MIN_DIST && d < C.SALTBRIDGE_DIST_MAX) sbLneg.push({ prot: pc, lig: nc, d, protispos: true });
    }
    for (const pc of L.pos) for (const nc of R.neg) {
        const d = dist(pc.center, nc.center);
        if (d > C.MIN_DIST && d < C.SALTBRIDGE_DIST_MAX) sbPneg.push({ prot: nc, lig: pc, d, protispos: false });
    }
    for (const sb of sbLneg.concat(sbPneg)) {
        out.push(_record(st, 'saltbridge', sb.prot.atoms, sb.lig.atoms, sb.d,
            { protispos: sb.protispos, lig_group: sb.lig.fgroup },
            { protPt: sb.prot.center, ligPt: sb.lig.center }));
    }

    // Hydrogen bonds: all candidates, then PLIP's refinement
    const hbonds = (accs, dons, protisdon) => {
        const found = [];
        for (const a of accs) for (const dn of dons) {
            const D = X(dn.d), A = X(a);
            const dad = dist(D, A);
            if (heavyHB && !dn.geo.explicit) {
                if (dad > C.MIN_DIST && dad <= 3.5) found.push({ a, d: dn.d, h: null, dad, dah: null, angle: null, protisdon, approx: true });
                continue;
            }
            if (!(dad > C.MIN_DIST && dad < C.HBOND_DIST_MAX)) continue;
            const best = _bestH(D, dn.geo, A);
            if (!(best.angle > C.HBOND_DON_ANGLE_MIN)) continue;
            found.push({ a, d: dn.d, h: best.h, dad, dah: dist(best.h, A), angle: best.angle, protisdon, approx: !dn.geo.explicit });
        }
        return found;
    };
    const inSalt = (hb) => sbPneg.concat(sbLneg).some(sb => {
        const pAtoms = sb.prot.atoms, lAtoms = sb.lig.atoms;
        return hb.protisdon ? (pAtoms.includes(hb.d) && lAtoms.includes(hb.a)) : (lAtoms.includes(hb.d) && pAtoms.includes(hb.a));
    });
    const refine = (list) => {
        const byDonor = new Map();
        for (const hb of list) {
            if (inSalt(hb)) continue;
            const score = hb.angle ?? -hb.dad;
            const cur = byDonor.get(hb.d);
            if (!cur || cur.score < score) byDonor.set(hb.d, { score, hb });
        }
        return [...byDonor.values()].map(v => v.hb);
    };
    const hbLdon = refine(hbonds(R.acceptors, L.donors, false));
    const hbPdon = refine(hbonds(L.acceptors, R.donors, true));
    for (const hb of hbPdon.concat(hbLdon)) {
        const protAtom = hb.protisdon ? hb.d : hb.a;
        const ligAtom = hb.protisdon ? hb.a : hb.d;
        out.push(_record(st, 'hbond', [protAtom], [ligAtom], hb.dad, {
            dist_h_a: hb.dah == null ? null : +hb.dah.toFixed(2),
            dist_d_a: +hb.dad.toFixed(2),
            don_angle: hb.angle == null ? null : +hb.angle.toFixed(1),
            protisdon: hb.protisdon,
            sidechain: !BACKBONE.has(atoms[protAtom].name),
            donortype: _atomType(ctx, hb.d), acctype: _atomType(ctx, hb.a),
        }, { more: { approx: hb.approx, hPt: hb.h } }));
    }

    // π-stacking
    const stacks = [];
    for (const r of R.rings) for (const l of L.rings) {
        const d = dist(r.center, l.center);
        if (!(d > C.MIN_DIST && d < C.PISTACK_DIST_MAX)) continue;
        const a = planeAngle(r.normal, l.normal);
        const offset = Math.min(dist(projectOnPlane(l.normal, l.center, r.center), l.center),
                                dist(projectOnPlane(r.normal, r.center, l.center), r.center));
        let ptype = null;
        if (a > 0 && a < C.PISTACK_ANG_DEV && offset < C.PISTACK_OFFSET_MAX) ptype = 'P';
        if (a > 90 - C.PISTACK_ANG_DEV && a < 90 + C.PISTACK_ANG_DEV && offset < C.PISTACK_OFFSET_MAX) ptype = 'T';
        if (!ptype) continue;
        stacks.push({ r, l });
        out.push(_record(st, 'pistacking', r.atoms, l.atoms, d,
            { cent_dist: +d.toFixed(2), angle: +a.toFixed(1), offset: +offset.toFixed(2), type: ptype === 'P' ? 'Parallel' : 'T-shaped' },
            { protPt: r.center, ligPt: l.center }));
    }

    // π-cation, with PLIP's His refinement
    const pication = (rings, charges, protcharged) => {
        const emit = (ring, p, d, offset) => {
            const protAtoms = protcharged ? p.atoms : ring.atoms;
            const ligAtoms = protcharged ? ring.atoms : p.atoms;
            out.push(_record(st, 'pication', protAtoms, ligAtoms, d,
                { dist: +d.toFixed(2), offset: +offset.toFixed(2), protcharged, lig_group: protcharged ? 'aromatic' : p.fgroup },
                { protPt: protcharged ? p.center : ring.center, ligPt: protcharged ? ring.center : p.center }));
        };
        for (const ring of rings) for (const p of charges) {
            const d = dist(ring.center, p.center);
            const offset = dist(projectOnPlane(ring.normal, ring.center, p.center), ring.center);
            if (!(d > C.MIN_DIST && d < C.PICATION_DIST_MAX) || !(offset < C.PISTACK_OFFSET_MAX)) continue;
            if (p.fgroup === 'tertamine') {
                // the amine's plane must face the ring; PLIP then stops looking at
                // further charges for this ring (a `break` in detection.py)
                const nb = ctx.ligTyped.info.get(p.amine).heavy.map(j => atoms[j].xyz);
                const amineNormal = cross(sub(nb[1], nb[0]), sub(nb[0], nb[2]));
                if (planeAngle(ring.normal, amineNormal) <= 30) emit(ring, p, d, offset);
                break;
            }
            if (protcharged && p.fgroup === 'HIS' && stacks.some(s => _resOf(st, s.r.atoms[0]).idx === p.res && s.l === ring)) continue;
            emit(ring, p, d, offset);
        }
    };
    pication(L.rings, R.pos, true);
    pication(R.rings, L.pos, false);

    // Hydrophobic contacts, then PLIP's three refinement rules
    let hyd = [];
    for (const b of R.hydrophobic) for (const l of L.hydrophobic) {
        const d = dist(X(b), X(l));
        if (d > C.MIN_DIST && d < C.HYDROPH_DIST_MAX) hyd.push({ b, l, d, res: atoms[b].res });
    }
    hyd = hyd.filter(h => !stacks.some(s => s.r.atoms.includes(h.b) && s.l.atoms.includes(h.l)));     // rule 1
    const perLigRes = new Map();                                                                       // rule 2
    for (const h of hyd) {
        const k = `${h.l}|${h.res}`;
        if (!perLigRes.has(k) || perLigRes.get(k).d > h.d) perLigRes.set(k, h);
    }
    hyd = [...perLigRes.values()];
    const final = [];
    if (ctx.ligTyped.peptide) {
        const perProt = new Map();
        for (const h of hyd) {
            const k = `${h.b}|${atoms[h.l].res}`;
            if (!perProt.has(k) || perProt.get(k).d > h.d) perProt.set(k, h);
        }
        final.push(...perProt.values());
    } else {
        const byProt = new Map();                                                                      // rule 3
        for (const h of hyd) { if (!byProt.has(h.b)) byProt.set(h.b, []); byProt.get(h.b).push(h); }
        const ligAdj = ctx.ligTyped.mol.adj;
        for (const list of byProt.values()) {
            if (list.length === 1) { final.push(list[0]); continue; }
            // patches of bonded ligand atoms; only the closest contact per patch
            const ids = new Set(list.map(h => h.l));
            const seen = new Set();
            for (const h of list) {
                if (seen.has(h.l)) continue;
                const patch = [], stack = [h.l];
                seen.add(h.l);
                while (stack.length) {
                    const u = stack.pop();
                    patch.push(u);
                    for (const v of ligAdj.get(u) || []) if (ids.has(v) && !seen.has(v)) { seen.add(v); stack.push(v); }
                }
                if (patch.length === 1) continue;     // PLIP's tuple clustering drops an isolated atom
                const members = list.filter(o => patch.includes(o.l));
                final.push(members.reduce((m, o) => (o.d < m.d ? o : m)));
            }
        }
    }
    for (const h of final) {
        out.push(_record(st, 'hydrophobic', [h.b], [h.l], h.d, { dist: +h.d.toFixed(2) }));
    }

    // Halogen bonds Y-O...X-C
    if (L.halDon) for (const acc of R.halAcc) for (const don of L.halDon) {
        const O = X(acc.o), Xh = X(don.x);
        const d = dist(O, Xh);
        if (!(d > C.MIN_DIST && d < C.HALOGEN_DIST_MAX)) continue;
        const accAngle = angleDeg(sub(X(acc.y), O), sub(Xh, O));
        const donAngle = angleDeg(sub(O, Xh), sub(X(don.c), Xh));
        if (!(accAngle > C.HALOGEN_ACC_ANGLE - C.HALOGEN_ANGLE_DEV && accAngle < C.HALOGEN_ACC_ANGLE + C.HALOGEN_ANGLE_DEV)) continue;
        if (!(donAngle > C.HALOGEN_DON_ANGLE - C.HALOGEN_ANGLE_DEV && donAngle < C.HALOGEN_DON_ANGLE + C.HALOGEN_ANGLE_DEV)) continue;
        out.push(_record(st, 'halogen', [acc.o], [don.x], d, {
            dist: +d.toFixed(2), don_angle: +donAngle.toFixed(1), acc_angle: +accAngle.toFixed(1),
            sidechain: !BACKBONE.has(atoms[acc.o].name), donortype: atoms[don.x].el, acctype: atoms[acc.o].el,
        }));
    }

    // Water bridges (first degree), with PLIP's refinement
    const wb = [];
    for (const w of ctx.waters) {
        const W = X(w);
        const inRange = (p) => { const d = dist(p, W); return d >= C.WATER_BRIDGE_MINDIST && d <= C.WATER_BRIDGE_MAXDIST ? d : null; };
        const ligAcc = L.acceptors.map(a => ({ a, d: inRange(X(a)) })).filter(x => x.d != null);
        const protAcc = R.acceptors.map(a => ({ a, d: inRange(X(a)) })).filter(x => x.d != null);
        const donW = (dons) => dons.map(dn => {
            const d = inRange(X(dn.d));
            if (d == null) return null;
            const best = _bestH(X(dn.d), dn.geo, W);
            return best.angle > C.WATER_BRIDGE_THETA_MIN ? { dn, d, h: best.h, angle: best.angle } : null;
        }).filter(Boolean);
        const ligDon = donW(L.donors), protDon = donW(R.donors);
        for (const [accs, dons, protisdon] of [[ligAcc, protDon, true], [protAcc, ligDon, false]]) {
            for (const acc of accs) for (const don of dons) {
                const wAngle = angleDeg(sub(X(acc.a), W), sub(don.h, W));
                if (!(wAngle > C.WATER_BRIDGE_OMEGA_MIN && wAngle < C.WATER_BRIDGE_OMEGA_MAX)) continue;
                wb.push({ w, acc, don, wAngle, protisdon });
            }
        }
    }
    const hbDonors = new Set(hbLdon.concat(hbPdon).map(h => h.d));
    const perWA = new Map();
    for (const b of wb) {
        if (hbDonors.has(b.don.dn.d)) continue;
        const k = `${b.w}|${b.acc.a}`;
        if (!perWA.has(k) || Math.abs(110 - perWA.get(k).wAngle) > Math.abs(110 - b.wAngle)) perWA.set(k, b);
    }
    const perW = new Map();
    for (const b of perWA.values()) {
        if (!perW.has(b.w)) perW.set(b.w, []);
        perW.get(b.w).push(b);
    }
    for (const list of perW.values()) {
        list.sort((p, q) => Math.abs(110 - p.wAngle) - Math.abs(110 - q.wAngle));
        for (const b of list.slice(0, 2)) {
            const protAtom = b.protisdon ? b.don.dn.d : b.acc.a;
            const ligAtom = b.protisdon ? b.acc.a : b.don.dn.d;
            const r = _record(st, 'waterbridge', [protAtom], [ligAtom], b.acc.d, {
                dist_a_w: +b.acc.d.toFixed(2), dist_d_w: +b.don.d.toFixed(2),
                don_angle: +b.don.angle.toFixed(1), water_angle: +b.wAngle.toFixed(1),
                protisdon: b.protisdon, donortype: _atomType(ctx, b.don.dn.d), acctype: _atomType(ctx, b.acc.a),
            }, { more: { water: [b.w], approx: !b.don.dn.geo.explicit } });
            out.push(r);
        }
    }

    return out;
}

/** A short atom-type label in the spirit of OpenBabel's (N3, Nam, Nar, O2, O3...). */
function _atomType(ctx, i) {
    const a = ctx.st.atoms[i];
    const t = ctx.ligTyped.info.get(i) || ctx.rec.info.get(i);
    if (!t) return a.el;
    const el = a.el[0] + a.el.slice(1).toLowerCase();
    if (t.arom) return `${el}ar`;
    if (a.el === 'N' && t.conj) return 'Nam';
    const hyb = t.hyb ?? (t.backbone && a.name === 'O' ? 2 : 3);
    return `${el}${hyb}`;
}

// ================================================================
// Rule set: ProLIF
// ================================================================

function _prolifLigandFeatures(ctx) {
    const { st, ligTyped } = ctx;
    const { atoms } = st;
    if (ligTyped.peptide) return _prolifResidueFeatures(ctx, ligTyped);
    const { info, mol } = ligTyped;
    const el = (i) => info.get(i).el;
    const F = { hydrophobic: [], donors: [], acceptors: [], cations: [], anions: [], rings: [], halDon: [], all: [] };

    for (const [i, a] of info) {
        F.all.push(i, ...a.hx);
        // Hydrophobic: [c,s,Br,I,S&H0&v2, C with >=3 heavy or chain C with 2; +0; not on N/O/F]
        if (a.charge === 0) {
            if (a.el === 'BR' || a.el === 'I') F.hydrophobic.push(i);
            else if (a.el === 'S' && (a.arom || (a.nH === 0 && a.heavy.length === 2))) F.hydrophobic.push(i);
            else if (a.el === 'C' && !a.heavy.some(j => ['N', 'O', 'F'].includes(el(j))) &&
                     (a.arom || a.heavy.length >= 3 || (a.heavy.length === 2 && !a.inRing))) F.hydrophobic.push(i);
        }
        // Donors: neutral O/S/N-H, ammonium N-H
        // Best rotamer per acceptor: ProLIF's exact pipeline takes its hydrogens
        // from RDKit or OpenBabel, and neither placement tracks the other well
        if (['N', 'O', 'S'].includes(a.el) && a.nH > 0) {
            const hyb = a.el === 'O' ? _obHydroxylHyb(a, info, mol) : _obHyb(a);
            F.donors.push({ d: i, geo: _donorGeometry(st, a, hyb) });
        }
        if (_prolifAcceptor(a, info, mol)) F.acceptors.push(i);
        // Charges: charged atoms not bonded to an opposite charge, plus amidinium /
        // carboxylate partners
        if (a.charge > 0 && !a.heavy.some(j => info.get(j).charge < 0)) F.cations.push(i);
        if (a.charge < 0 && !a.heavy.some(j => info.get(j).charge > 0)) F.anions.push(i);
        if (a.el === 'N' && a.charge === 0 && a.heavy.length + a.nH === 3 && !a.heavy.some(j => el(j) === 'O') &&
            a.heavy.some(c => el(c) === 'C' && info.get(c).heavy.some(n => n !== i && info.get(n).charge > 0 && mol.order(c, n) === 2))) {
            F.cations.push(i);
        }
        if (a.el === 'O' && a.charge === 0 && a.heavy.length === 1 && ['C', 'S', 'P'].includes(el(a.heavy[0])) &&
            mol.order(i, a.heavy[0]) === 2 && info.get(a.heavy[0]).heavy.some(o => info.get(o).charge < 0 && el(o) === 'O')) {
            F.anions.push(i);
        }
        if (['CL', 'BR', 'I'].includes(a.el) && a.heavy.length === 1 &&
            ['C', 'N', 'SI', 'F', 'CL', 'BR', 'I'].includes(el(a.heavy[0]))) F.halDon.push({ x: i, d: a.heavy[0] });
    }
    for (const r of mol.rings) {
        if (!r.aromatic) continue;
        F.rings.push({ atoms: r.atoms, center: r.center, normal: r.normal });
    }
    return F;
}

/** ProLIF's HBond acceptor SMARTS on the perceived ligand. */
function _prolifAcceptor(a, info, mol) {
    const el = (i) => info.get(i).el;
    const conns = a.heavy.length + a.nH;
    if (a.el === 'N') {
        if (a.charge !== 0) return false;
        if (a.arom) return conns !== 3;
        if (conns === 3) {
            // [NX3]-*=[O,N,P,S] (amide, sulfonamide, urea...) and [NX3]-[a] (aniline)
            for (const j of a.heavy) {
                if (info.get(j).arom) return false;
                if (mol.order(a.i, j) === 1 && info.get(j).heavy.some(k => k !== a.i && mol.order(j, k) === 2 && ['O', 'N', 'P', 'S'].includes(el(k)))) return false;
            }
        }
        // N=C(-[C,N])-N: amidine / guanidine imine
        for (const c of a.heavy) {
            if (el(c) !== 'C' || mol.order(a.i, c) !== 2) continue;
            const others = info.get(c).heavy.filter(k => k !== a.i);
            if (others.some(k => el(k) === 'N') && others.some(k => ['C', 'N'].includes(el(k)))) return false;
        }
        return true;
    }
    if (a.el === 'O') {
        if (a.arom) return a.charge === 0;
        if (a.heavy.length === 1 && el(a.heavy[0]) === 'N' &&
            info.get(a.heavy[0]).heavy.filter(k => el(k) === 'O').length >= 2) return false;   // nitro
        if (a.heavy.length === 2 && a.nH === 0) {
            const [p, q] = a.heavy;
            const aliphC = (k) => el(k) === 'C' && !info.get(k).arom;
            if ((aliphC(p) && info.get(q).carbonylC && aliphC(q)) || (aliphC(q) && info.get(p).carbonylC && aliphC(p))) return false;
            if (info.get(p).arom && info.get(q).arom) return false;
        }
        return true;
    }
    if (a.el === 'F') {
        const c = a.heavy[0];
        return c !== undefined && el(c) === 'C' && !info.get(c).heavy.some(k => k !== a.i && HALOGENS.has(el(k)));
    }
    return false;
}

function _prolifResidueFeatures(ctx, typed) {
    const { st } = ctx;
    const { atoms } = st;
    const F = { hydrophobic: [], donors: [], acceptors: [], cations: [], anions: [], rings: [], xbAcc: [], all: [] };
    for (const [ri, { rn, byName }] of typed.resAtoms) {
        for (const name of PROLIF_PHOB[rn] || []) {
            if (byName[name] === undefined) continue;
            if (name === 'SD' && rn !== 'MET') continue;
            F.hydrophobic.push(byName[name]);
        }
        // a disulfide sulfur is S&H0&v2
        if (rn === 'CYS' && byName.SG !== undefined && typed.info.get(byName.SG)?.nH === 0) F.hydrophobic.push(byName.SG);
        for (const ring of RING_ATOMS[rn] || []) {
            const ids = ring.map(n => byName[n]).filter(i => i !== undefined);
            if (ids.length < 5) continue;
            const pts = ids.map(i => atoms[i].xyz);
            F.rings.push({ atoms: ids, center: centroid(pts), normal: ringNormal(pts), res: ri });
        }
        for (const i of st.residues[ri].atoms) F.all.push(i);
    }
    for (const [i, a] of typed.info) {
        if (a.don) F.donors.push({ d: i, geo: _donorGeometry(st, a, _residueHyb(a)) });
        // ProLIF acceptors on a protonated residue: no charged N, no amide or aromatic N-H
        if (a.acc && !(a.el === 'N' && a.charge > 0)) F.acceptors.push(i);
        if (a.charge > 0) F.cations.push(i);
        if (a.charge < 0) F.anions.push(i);
        const xbOk = ['N', 'O', 'P', 'S', 'SE'].includes(a.el) || a.arom;
        if (xbOk && a.charge <= 0) F.xbAcc.push({ a: i, r: a.heavy.slice() });
    }
    return F;
}

function _prolifDetect(ctx) {
    const { st } = ctx;
    const { atoms } = st;
    const P = PROLIF_CONFIG;
    const L = _prolifLigandFeatures(ctx);
    const R = _prolifResidueFeatures(ctx, ctx.rec);
    const X = (i) => atoms[i].xyz;
    const fold = (a) => (a > 90 ? 180 - a : a);
    const within = (v, [lo, hi]) => v >= lo && v <= hi;

    // keep the closest hit per (residue, interaction), as ProLIF's count=False does
    const best = new Map();
    const hit = (type, protAtoms, ligAtoms, d, detail, extra = {}) => {
        const k = `${atoms[protAtoms[0]].res}|${type}`;
        if (best.has(k) && best.get(k).dist <= d) return;
        best.set(k, _record(st, type, protAtoms, ligAtoms, d, detail, extra));
    };

    for (const p of R.hydrophobic) for (const l of L.hydrophobic) {
        const d = dist(X(p), X(l));
        if (d <= P.hydrophobic) hit('Hydrophobic', [p], [l], d, { distance: +d.toFixed(2) });
    }
    const hb = (accs, dons, type, protIsDonor) => {
        for (const a of accs) for (const dn of dons) {
            const d = dist(X(a), X(dn.d));
            if (d > P.hbond.distance) continue;
            const b = _bestH(X(dn.d), dn.geo, X(a));
            if (!within(b.angle, P.hbond.angle)) continue;
            const prot = protIsDonor ? dn.d : a, lig = protIsDonor ? a : dn.d;
            hit(type, [prot], [lig], d, { distance: +d.toFixed(2), DHA_angle: +b.angle.toFixed(1) },
                { more: { approx: !dn.geo.explicit, hPt: b.h } });
        }
    };
    hb(L.acceptors, R.donors, 'HBAcceptor', true);
    hb(R.acceptors, L.donors, 'HBDonor', false);

    for (const l of L.cations) for (const p of R.anions) {
        const d = dist(X(l), X(p));
        if (d <= P.ionic) hit('Cationic', [p], [l], d, { distance: +d.toFixed(2) });
    }
    for (const l of L.anions) for (const p of R.cations) {
        const d = dist(X(l), X(p));
        if (d <= P.ionic) hit('Anionic', [p], [l], d, { distance: +d.toFixed(2) });
    }
    const cationPi = (rings, cations, type, ringIsLig) => {
        for (const ring of rings) for (const c of cations) {
            const d = dist(ring.center, X(c));
            if (d > P.cationpi.distance) continue;
            const ang = fold(angleDeg(ring.normal, sub(X(c), ring.center)));
            if (!within(ang, P.cationpi.angle)) continue;
            const prot = ringIsLig ? [c] : ring.atoms, lig = ringIsLig ? ring.atoms : [c];
            hit(type, prot, lig, d, { distance: +d.toFixed(2), angle: +ang.toFixed(1) },
                { protPt: ringIsLig ? X(c) : ring.center, ligPt: ringIsLig ? ring.center : X(c) });
        }
    };
    cationPi(R.rings, L.cations, 'CationPi', false);
    cationPi(L.rings, R.cations, 'PiCation', true);

    for (const r of R.rings) for (const l of L.rings) {
        const d = dist(l.center, r.center);
        if (d > P.edgetoface.distance) continue;
        const plane = planeAngle(l.normal, r.normal);
        const n1 = fold(angleDeg(l.normal, sub(r.center, l.center)));
        const n2 = fold(angleDeg(r.normal, sub(l.center, r.center)));
        let subtype = null, ncc = null, inter = null;
        if (d <= P.facetoface.distance && within(plane, P.facetoface.plane)) {
            ncc = within(n1, P.facetoface.ncc) ? n1 : (within(n2, P.facetoface.ncc) ? n2 : null);
            if (ncc != null) subtype = 'FaceToFace';
        }
        if (!subtype && within(plane, P.edgetoface.plane)) {
            ncc = within(n1, P.edgetoface.ncc) ? n1 : (within(n2, P.edgetoface.ncc) ? n2 : null);
            if (ncc != null) {
                const pt = _planeIntersectPoint(l.normal, l.center, r.normal, r.center);
                if (pt) {
                    inter = Math.min(dist(l.center, pt), dist(r.center, pt));
                    if (inter <= P.edgetoface.intersect) subtype = 'EdgeToFace';
                }
            }
        }
        if (!subtype) continue;
        hit('PiStacking', r.atoms, l.atoms, d,
            { distance: +d.toFixed(2), plane_angle: +plane.toFixed(1), normal_to_centroid_angle: +ncc.toFixed(1), subtype },
            { protPt: r.center, ligPt: l.center });
    }

    for (const xd of L.halDon) for (const acc of R.xbAcc) {
        const d = dist(X(acc.a), X(xd.x));
        if (d > P.halogen.distance) continue;
        const axd = angleDeg(sub(X(acc.a), X(xd.x)), sub(X(xd.d), X(xd.x)));
        if (!within(axd, P.halogen.axd)) continue;
        const xar = acc.r.map(r => angleDeg(sub(X(xd.x), X(acc.a)), sub(X(r), X(acc.a)))).find(v => within(v, P.halogen.xar));
        if (xar === undefined) continue;
        hit('XBDonor', [acc.a], [xd.x], d, { distance: +d.toFixed(2), AXD_angle: +axd.toFixed(1), XAR_angle: +xar.toFixed(1) });
    }

    // vdW contacts: explicit hydrogens count; heavy atoms get a margin standing in for them
    const ligAll = L.all, ligGrid = _grid(atoms, ligAll, 4.5);
    const ligHasH = ligAll.some(i => !isHeavy(atoms[i])), recHasH = R.all.some(i => !isHeavy(atoms[i]));
    const heavyMargin = P.vdwHeavyMargin * ((ligHasH ? 0 : 0.5) + (recHasH ? 0 : 0.5));
    for (const p of R.all) {
        const pa = atoms[p];
        for (const l of ligGrid.near(pa.xyz, 4.6)) {
            const la = atoms[l];
            const margin = isHeavy(pa) && isHeavy(la) ? heavyMargin : 0;
            const lim = (VDW_RADII[_elH(pa.el)] ?? 1.8) + (VDW_RADII[_elH(la.el)] ?? 1.8) + P.vdwTolerance + margin;
            const d = dist(pa.xyz, la.xyz);
            if (d <= lim) hit('VdWContact', [p], [l], d, { distance: +d.toFixed(2) });
        }
    }

    return [...best.values()];
}

/** ProLIF's EdgeToFace intersect point: the ligand centroid projected onto the planes' intersection line. */
function _planeIntersectPoint(n1, c1, n2, c2) {
    const dir = cross(n1, n2);
    const A = [n1, n2, dir];
    const det = dot(A[0], cross(A[1], A[2]));
    if (Math.abs(det) < 1e-9) return null;
    const d = [dot(n1, c1), dot(n2, c2), 0];
    // Cramer's rule
    const col = (k) => A.map((row, r) => row.map((v, c) => (c === k ? d[r] : v)));
    const det3 = (M) => dot(M[0], cross(M[1], M[2]));
    const p = [det3(col(0)) / det, det3(col(1)) / det, det3(col(2)) / det];
    const u = unit(dir);
    return add(p, scale(u, dot(sub(c1, p), u)));
}

// ================================================================
// Public entry point
// ================================================================

/**
 * Profile the interactions of one ligand.
 *
 * @param {Structure} st         from parseStructure()
 * @param {string}    ligandId   from findLigands(); defaults to the first
 * @param {{rules?: 'plip'|'prolif', hbondModel?: 'geometric'|'heavy'}} [opts]
 * @returns {{ligand, rules, interactions: object[], residues: object[], hydrogens: string, ms: number}}
 */
export function profile(st, ligandId, opts = {}) {
    const t0 = (typeof performance !== 'undefined' ? performance : Date).now();
    const ligands = findLigands(st);
    const lig = ligands.find(l => l.id === ligandId) || ligands[0];
    if (!lig) throw new Error('no ligand found in this structure');
    const rules = opts.rules === 'prolif' ? 'prolif' : 'plip';
    const ctx = _prepare(st, lig, opts);
    const interactions = rules === 'prolif' ? _prolifDetect(ctx) : _plipDetect(ctx, opts);
    const order = rules === 'prolif' ? PROLIF_TYPES : PLIP_TYPES;
    interactions.sort((a, b) => order.indexOf(a.type) - order.indexOf(b.type) ||
                                a.chain.localeCompare(b.chain) || a.resnr - b.resnr || a.dist - b.dist);
    const ligHasH = lig.atoms.some(i => !isHeavy(st.atoms[i]));
    return {
        ligand: { id: lig.id, kind: lig.kind, resname: lig.resname, chain: lig.chain, resnr: lig.resnr, nHeavy: lig.nHeavy, atoms: lig.atoms },
        rules,
        interactions,
        residues: summarizeResidues(interactions),
        hydrogens: st.hasHydrogens ? (ligHasH ? 'explicit' : 'receptor only') : 'none',
        bindingSite: ctx.bs.length,
        ms: Math.round((typeof performance !== 'undefined' ? performance : Date).now() - t0),
    };
}

/** One row per contacted residue, with the interaction types it makes. */
export function summarizeResidues(interactions) {
    const m = new Map();
    for (const x of interactions) {
        const k = `${x.chain}:${x.resnr}${x.icode || ''}`;
        if (!m.has(k)) m.set(k, { key: k, chain: x.chain, resnr: x.resnr, icode: x.icode, restype: x.restype, types: {}, minDist: Infinity });
        const r = m.get(k);
        r.types[x.type] = (r.types[x.type] || 0) + 1;
        r.minDist = Math.min(r.minDist, x.dist);
    }
    return [...m.values()].sort((a, b) => a.chain.localeCompare(b.chain) || a.resnr - b.resnr);
}

// ================================================================
// Export helpers
// ================================================================

/** BoltzTriage's JSONL pose record (PLIP vocabulary). */
export function toTriageRecord(id, result, extra = {}) {
    return {
        id,
        ...extra,
        contacts: result.interactions
            .filter(x => PLIP_TYPES.includes(x.type))
            .map(x => ({ chain: x.chain, resnr: x.resnr, restype: x.restype, itype: x.type, dist: x.dist })),
    };
}

/** Long-format CSV rows: one interaction per row. */
export function toCsvRows(id, result) {
    return result.interactions.map(x => ({
        id, rules: result.rules, ligand: result.ligand.id, chain: x.chain, resnr: x.resnr, restype: x.restype,
        itype: x.type, dist: x.dist, approx: x.approx ? 1 : 0,
    }));
}
