/**
 * triagedemo.js
 *
 * Demo screens for BoltzTriage.
 *
 * WHAT IS REAL AND WHAT IS NOT
 * ----------------------------
 * The *pocket definition* below is transcribed from the project's own CRBN
 * allosteric notebook (`Internal Research/CRBN_allosteric/CRBN_allosteric_jupyter.ipynb`,
 * which derives it live from 9SFM). Residue numbers, roles and the degron
 * off-site pocket are the real ones, abridged.
 *
 * The *poses* are synthetic. The screen is generated here from a latent mixture
 * model so the prototype can demonstrate the behaviours the notebooks found in
 * the real data:
 *
 *   - raw interaction scores track molecular size (the size confound),
 *   - anionic ligands pile onto cationic anchors (the salt-bridge artefact),
 *   - some poses sit in a second, off-site pocket rather than binding weakly,
 *   - some pose crops exclude the pocket, so their tier means nothing,
 *   - the library contains a handful of distinguishable binding modes.
 *
 * No demo number should ever be quoted as a result. The UI labels the demo
 * screen as synthetic for exactly that reason.
 */

// ----------------------------------------------------------------
// Deterministic PRNG — the demo must look the same on every reload
// ----------------------------------------------------------------

function mulberry32(seed) {
    return function () {
        seed |= 0; seed = (seed + 0x6D2B79F5) | 0;
        let t = Math.imul(seed ^ (seed >>> 15), 1 | seed);
        t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
        return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
    };
}

const _gauss = (rnd, mean, sd) => {
    const u = Math.max(rnd(), 1e-9), v = rnd();
    return mean + sd * Math.sqrt(-2 * Math.log(u)) * Math.cos(2 * Math.PI * v);
};
const _pick = (rnd, arr) => arr[Math.floor(rnd() * arr.length)];

// ----------------------------------------------------------------
// Ligand pool — plain drug-like SMILES in three size bins.
// MW is computed from the SMILES by RDKit at load, so structure and
// size always agree. `anionic` drives the salt-bridge artefact demo.
// ----------------------------------------------------------------

const LIGANDS = {
    small: [
        { smiles: 'COc1ccc(cc1)C(=O)Nc1ccccc1' },
        { smiles: 'Clc1ccc(cc1)C(=O)NCc1ccccc1' },
        { smiles: 'Fc1ccc(cc1)NC(=O)c1ccncc1' },
        { smiles: 'CN1CCN(CC1)c1ccc(F)cc1' },
        { smiles: 'Cc1ccc(cc1)S(=O)(=O)Nc1ccccc1' },
        { smiles: 'c1ccc2c(c1)[nH]c(n2)Cc1ccccc1' },
        { smiles: 'NC(=O)c1cccc(c1)Nc1ccccc1' },
        { smiles: 'CCOC(=O)c1ccc(cc1)NC(=O)C' },
        { smiles: 'OC(=O)c1ccc(cc1)Oc1ccccc1', anionic: 'carboxylate' },
        { smiles: 'CC(=O)Nc1ccc(cc1)S(=O)(=O)N' },
    ],
    mid: [
        { smiles: 'COc1ccc(cc1)C(=O)Nc1ccc(cc1)N1CCOCC1' },
        { smiles: 'Clc1ccc(cc1)C(=O)Nc1ccc(cc1)N1CCN(C)CC1' },
        { smiles: 'Fc1ccc(cc1)C(=O)NCc1ccc(cc1)S(=O)(=O)N' },
        { smiles: 'Cc1ccc(cc1)C(=O)Nc1ccc2ncccc2c1' },
        { smiles: 'c1ccc(cc1)C(=O)N1CCN(CC1)c1ccccn1' },
        { smiles: 'COc1cc(OC)cc(c1)C(=O)NCc1ccncc1' },
        { smiles: 'Brc1ccc(cc1)NC(=O)c1ccc(cc1)N1CCCC1' },
        { smiles: 'CN(C)CCNC(=O)c1ccc(cc1)Oc1ccccc1' },
        { smiles: 'OC(=O)Cc1ccc(cc1)C(=O)Nc1ccccc1', anionic: 'carboxylate' },
        { smiles: 'OS(=O)(=O)c1ccc(cc1)NC(=O)c1ccccc1', anionic: 'sulfonate' },
    ],
    large: [
        { smiles: 'COc1ccc(cc1)C(=O)Nc1ccc(cc1)N1CCN(CC1)Cc1ccccc1' },
        { smiles: 'Clc1ccc(cc1)C(=O)Nc1ccc(cc1)C(=O)N1CCN(C)CC1' },
        { smiles: 'Fc1ccc(cc1)C(=O)Nc1ccc(cc1)S(=O)(=O)N1CCOCC1' },
        { smiles: 'c1ccc(cc1)CN1CCN(CC1)C(=O)c1ccc(cc1)Oc1ccccc1' },
        { smiles: 'COc1ccc(cc1)S(=O)(=O)Nc1ccc(cc1)C(=O)N1CCCCC1' },
        { smiles: 'Clc1cccc(c1)C(=O)Nc1ccc(cc1)N1CCN(CC1)c1ccccn1' },
        { smiles: 'CCN(CC)CCNC(=O)c1ccc(cc1)Nc1ccc(F)cc1' },
        { smiles: 'c1ccc2c(c1)nc(n2Cc1ccccc1)C(=O)NCc1ccccc1' },
        { smiles: 'OC(=O)c1ccc(cc1)NC(=O)c1ccc(cc1)N1CCN(C)CC1', anionic: 'carboxylate' },
        { smiles: 'Cc1ccc(cc1)C(=O)Nc1ccc(cc1)Nc1ncccn1' },
    ],
};

// ----------------------------------------------------------------
// Interaction types PLIP would plausibly call for a given residue
// ----------------------------------------------------------------

const HYDROPHOBIC = new Set(['ALA', 'VAL', 'LEU', 'ILE', 'MET', 'PHE', 'TRP', 'PRO', 'CYS', 'TYR']);
const AROMATIC    = new Set(['PHE', 'TRP', 'TYR', 'HIS']);
const CATIONIC    = new Set(['ARG', 'LYS', 'HIS']);
const ANIONIC_RES = new Set(['ASP', 'GLU']);
const POLAR_RES   = new Set(['SER', 'THR', 'ASN', 'GLN', 'TYR', 'HIS', 'TRP', 'ARG', 'LYS', 'ASP', 'GLU', 'CYS']);

function _itype(rnd, restype, ligandAnionic) {
    const opts = [];
    if (HYDROPHOBIC.has(restype)) opts.push('hydrophobic', 'hydrophobic', 'hydrophobic');
    if (POLAR_RES.has(restype))   opts.push('H-bond(sidechain,prot-donor)', 'H-bond(sidechain,lig-donor)');
    if (AROMATIC.has(restype))    opts.push('pi-stack(P)');
    if (CATIONIC.has(restype) && ligandAnionic) opts.push('salt-bridge', 'salt-bridge', 'salt-bridge');
    if (ANIONIC_RES.has(restype)) opts.push('H-bond(sidechain,lig-donor)');
    opts.push('hydrophobic');
    return _pick(rnd, opts);
}

function _contact(rnd, chain, resnr, restype, ligandAnionic) {
    const itype = _itype(rnd, restype, ligandAnionic);
    if (itype.startsWith('H-bond')) {
        const ad = 2.6 + rnd() * 0.9;
        return { chain, resnr, restype, itype, dist_ad: ad, dist_ah: ad - 0.85 };
    }
    if (itype === 'salt-bridge') return { chain, resnr, restype, itype, dist: 3.2 + rnd() * 1.3 };
    if (itype.startsWith('pi'))  return { chain, resnr, restype, itype, dist: 3.8 + rnd() * 1.2 };
    return { chain, resnr, restype, itype, dist: 3.4 + rnd() * 0.7 };
}

// ----------------------------------------------------------------
// Screen definitions
//
// residues[]  one entry per residue the screen knows about
//   key    'chain:resnr'
//   role   key | shell | reach | decoy | report | background
//   p      base probability that an arbitrary pose contacts it
//   note   shown in the UI (Wootten number, zone, why it matters)
// modes[]     latent binding modes — what the clustering step should find
//   boost  per-residue probability override for members of this mode
// ----------------------------------------------------------------

function R(key, restype, role, p, note) {
    const [chain, resnr] = key.split(':');
    return { key, chain, resnr: +resnr, restype, role, p, note: note || '' };
}

const SCREENS = [
    {
        id: 'crbn',
        target: 'CRBN allosteric',
        label: 'CRBN allosteric pocket screen',
        arm: 'reference',
        n: 520,
        seed: 20261023,
        lede: 'A small molecule is the bound reference: GSK SB-405483 (A1CEG) in 9SFM chain B. ' +
              'Lenalidomide sits 20 Å away in the degron pocket — poses there are off-site, not weak.',
        numbering: 'canonical Q96SW2 — 9SFM chain B',
        reference: {
            kind: 'smallmol',
            label: 'SB-405483 (A1CEG) — 9SFM chain B',
            note: 'Ten of the twelve reference contacts are hydrophobic. The pocket is lipophilic, ' +
                  'so breadth of contact is the signal; the single Arg283 H-bond is reported, not gated.',
        },
        guards: {
            anionIsLiability: true,
            anionNote: 'Gating on Arg283 would promote anionic frequent hitters by charge ' +
                       'complementarity. anion → R283 is a column to check before trusting a Tier 1 call.',
        },
        residues: [
            R('B:264', 'TRP', 'key', 0.33, 'Lipophilic wall'),
            R('B:268', 'LEU', 'key', 0.38, 'Lipophilic wall'),
            R('B:273', 'LEU', 'key', 0.41, 'Lipophilic wall'),
            R('B:283', 'ARG', 'key', 0.36, 'Polar anchor — H-bond D-A 2.71 Å. Reported, NOT gated'),
            R('B:288', 'LEU', 'key', 0.35, 'Lipophilic wall'),
            R('B:289', 'PRO', 'key', 0.28, 'Lipophilic wall'),
            R('B:314', 'ILE', 'key', 0.30, 'Lipophilic floor'),
            R('B:321', 'LEU', 'key', 0.32, 'Lipophilic floor'),
            R('B:334', 'LYS', 'key', 0.25, 'Rim — cationic, watch salt bridges'),
            R('B:337', 'ILE', 'key', 0.27, 'Lipophilic floor'),
            R('B:339', 'SER', 'key', 0.22, 'Second polar anchor'),
            R('B:261', 'LEU', 'shell', 0.30, '4 Å shell, no interaction in the reference'),
            R('B:284', 'VAL', 'shell', 0.28, '4 Å shell'),
            R('B:287', 'CYS', 'shell', 0.21, '4 Å shell'),
            R('B:318', 'CYS', 'shell', 0.19, '4 Å shell'),
            R('B:342', 'LEU', 'shell', 0.24, '4 Å shell'),
            R('B:343', 'CYS', 'shell', 0.18, '4 Å shell'),
            R('B:345', 'PRO', 'shell', 0.20, '4 Å shell'),
            R('B:351', 'ASN', 'decoy', 0.09, 'Degron / IMiD pocket — 20 Å away'),
            R('B:353', 'HIS', 'decoy', 0.10, 'Degron / IMiD pocket'),
            R('B:378', 'HIS', 'decoy', 0.08, 'Degron / IMiD pocket'),
            R('B:380', 'TRP', 'decoy', 0.12, 'Tri-Trp degron pocket'),
            R('B:386', 'TRP', 'decoy', 0.11, 'Tri-Trp degron pocket'),
            R('B:400', 'TRP', 'decoy', 0.10, 'Tri-Trp degron pocket'),
            R('B:402', 'PHE', 'decoy', 0.09, 'Degron pocket floor'),
            R('B:276', 'GLU', 'background', 0.17, ''),
            R('B:302', 'TYR', 'background', 0.15, ''),
            R('B:325', 'ARG', 'background', 0.18, 'Surface arginine — salt-bridge magnet'),
            R('B:360', 'SER', 'background', 0.13, ''),
        ],
        modes: [
            { name: 'allosteric pocket, broad', w: 0.22, site: 'main', mw: [385, 55], prob: [0.58, 0.12],
              boost: { 'B:264': 0.72, 'B:268': 0.80, 'B:273': 0.84, 'B:288': 0.70, 'B:314': 0.62,
                       'B:321': 0.66, 'B:337': 0.58, 'B:283': 0.55, 'B:339': 0.40 } },
            { name: 'allosteric rim only', w: 0.33, site: 'main', mw: [330, 50], prob: [0.44, 0.11],
              boost: { 'B:273': 0.52, 'B:283': 0.48, 'B:334': 0.44, 'B:261': 0.50, 'B:342': 0.42 } },
            { name: 'degron / IMiD pocket', w: 0.18, site: 'off', mw: [350, 58], prob: [0.51, 0.13],
              boost: { 'B:380': 0.78, 'B:386': 0.72, 'B:400': 0.64, 'B:402': 0.55, 'B:353': 0.48, 'B:351': 0.42 } },
            { name: 'surface / unresolved', w: 0.27, site: 'main', mw: [320, 60], prob: [0.33, 0.10],
              boost: { 'B:325': 0.46, 'B:276': 0.40, 'B:302': 0.36 } },
        ],
        cationicAnchors: ['B:283', 'B:334', 'B:325'],
        siteSeparation: 0.04,   // 20 Angstrom apart: a pose cannot be in both
        cropMissRate: 0.11,
    }
];

export function listDemoScreens() {
    return SCREENS.map(s => ({
        id: s.id, target: s.target, label: s.label, arm: s.arm, n: s.n,
        lede: s.lede, hasReference: !!s.reference,
        referenceLabel: s.reference ? s.reference.label : null,
    }));
}

// ----------------------------------------------------------------
// Pose generation
// ----------------------------------------------------------------

const ID_CHARS = 'ABCDEFGHJKLMNPQRSTUVWXYZ0123456789';

function _makeId(rnd) {
    let s = 'SM-';
    for (let i = 0; i < 8; i++) s += ID_CHARS[Math.floor(rnd() * ID_CHARS.length)];
    return s;
}

/**
 * Build a demo screen. `mwOf(smiles)` resolves a real MW from RDKit; when it is
 * unavailable the ligand bin's nominal midpoint is used instead, so the demo
 * still runs with RDKit missing.
 */
export function buildDemoScreen(id, mwOf) {
    const def = SCREENS.find(s => s.id === id);
    if (!def) throw new Error(`unknown demo screen: ${id}`);
    const rnd = mulberry32(def.seed);

    // Resolve the ligand pool once, with real MW where we can get it, and keep it
    // sorted so a drawn size can be matched to a structure that really has it.
    const NOMINAL = { small: 260, mid: 350, large: 450 };
    const pool = [];
    for (const bin of ['small', 'mid', 'large']) {
        LIGANDS[bin].forEach((l, i) => {
            // The nominal spread keeps the pool usable without RDKit; with RDKit
            // every MW is the structure's own.
            pool.push({ ...l, mw: (mwOf && mwOf(l.smiles)) || (NOMINAL[bin] + i * 11) });
        });
    }
    pool.sort((a, b) => a.mw - b.mw);

    const byKey = new Map(def.residues.map(r => [r.key, r]));
    const modeCdf = [];
    let acc = 0;
    for (const m of def.modes) { acc += m.w; modeCdf.push(acc); }

    const poses = [];
    for (let i = 0; i < def.n; i++) {
        const u = rnd() * acc;
        const mi = modeCdf.findIndex(c => u <= c);
        const mode = def.modes[Math.max(mi, 0)];

        // Size first: the ligand decides how much of anything it can touch.
        const mwWant = _gauss(rnd, mode.mw[0], mode.mw[1]);
        const near = pool
            .map((l, j) => [Math.abs(l.mw - mwWant), j])
            .sort((a, b) => a[0] - b[0])
            .slice(0, 3);
        const lig = pool[near[Math.floor(rnd() * near.length)][1]];
        // Larger ligands touch more residues — this is the confound the UI has to expose.
        const sizeFactor = 0.85 + 0.0022 * (lig.mw - 300);

        // Boltz pocket crops do not all contain the whole pocket. A pose whose
        // crop excludes the site cannot score, and its tier says nothing about
        // the compound — the UI has to show that as "not assessable", not Tier 5.
        const cropMiss = rnd() < def.cropMissRate;
        const crop = new Set();
        for (const r of def.residues) {
            if (r.role === 'background') { crop.add(r.key); continue; }
            if (cropMiss) continue;
            if (rnd() < 0.06) continue;             // partial crop
            crop.add(r.key);
        }

        const contacts = [];
        for (const r of def.residues) {
            if (!crop.has(r.key)) continue;
            let p = (mode.boost && mode.boost[r.key] != null) ? mode.boost[r.key] : r.p;
            p *= sizeFactor;
            // A pose sits in one pocket. How much the two sites can bleed into
            // each other is a property of the structure, not of the scoring.
            const sep = def.siteSeparation != null ? def.siteSeparation : 1;
            if (mode.site === 'main' && r.role === 'decoy') p *= sep;
            if (mode.site === 'off' && ['key', 'shell', 'reach'].includes(r.role)) p *= sep;
            if (lig.anionic && def.cationicAnchors.includes(r.key)) p = Math.min(1, p * 2.4);
            if (rnd() < p) contacts.push(_contact(rnd, r.chain, r.resnr, r.restype, lig.anionic));
        }
        // A few poses pick up a second contact of a different type on the same residue.
        for (const c of contacts.slice()) {
            if (rnd() < 0.12) contacts.push(_contact(rnd, c.chain, c.resnr, c.restype, lig.anionic));
        }

        let prob = _gauss(rnd, mode.prob[0], mode.prob[1]) + 0.0004 * (contacts.length - 8);
        prob = Math.min(0.95, Math.max(0.05, prob));

        poses.push({
            id: _makeId(rnd),
            smiles: lig.smiles,
            mw: Math.round(lig.mw * 10) / 10,
            prob: Math.round(prob * 1000) / 1000,
            anion: lig.anionic || '',
            contacts,
            // What the sequence alignment would tell you: which of the screen's
            // residues this pose's crop actually contains.
            crop: [...crop],
            _mode: mode.name,          // ground truth — the demo's own check, never scored
        });
    }

    // The reference footprint. A peptide reaches far further than any small
    // molecule — that mismatch is the first thing the UI has to say out loud, so
    // the peptide footprint is generated at its real breadth.
    let refContacts = null;
    if (def.reference) {
        const keep = def.reference.kind === 'peptide'
            ? { key: 1, report: 1, shell: 0.9, reach: 0.8, decoy: 0.7, background: 0.75 }
            : { key: 1, report: 0.9, shell: 0.3, reach: 0.15, decoy: 0, background: 0.1 };
        refContacts = [];
        for (const r of def.residues) {
            if (rnd() < (keep[r.role] ?? 0)) {
                refContacts.push(_contact(rnd, r.chain, r.resnr, r.restype, false));
            }
        }
    }

    return {
        id: def.id,
        name: def.label,
        target: def.target,
        synthetic: true,
        lede: def.lede,
        numbering: def.numbering,
        suggestedArm: def.arm,
        reference: def.reference ? { ...def.reference, contacts: refContacts } : null,
        guards: def.guards,
        roles: Object.fromEntries(def.residues.map(r => [r.key, r.role])),
        notes: Object.fromEntries(def.residues.map(r => [r.key, r.note])),
        restypes: Object.fromEntries(def.residues.map(r => [r.key, r.restype])),
        cationicAnchors: def.cationicAnchors,
        poses,
    };
}
