/**
 * peptide.js
 *
 * Peptide sequence → SMILES conversion (browser-side, no API).
 * Supports all 20 canonical L-amino acids (uppercase) and
 * their D-forms (lowercase), plus common N/C terminal modifications.
 *
 * Algorithm: each residue contributes an N-Cα(sidechain)-C(=O) block;
 * blocks are concatenated and capped with terminal groups.
 */

// ----------------------------------------------------------------
// Amino acid SMILES blocks
// Each entry is the N-Cα(sidechain) fragment; C(=O) is appended per residue.
// L-amino acids = uppercase, D-amino acids = lowercase (flipped stereo).
// ----------------------------------------------------------------
const AA_BLOCKS = {
    // L-forms
    A: 'N[C@@H](C)',
    G: 'NC',
    V: 'N[C@@H](C(C)C)',
    L: 'N[C@@H](CC(C)C)',
    I: 'N[C@@H]([C@@H](CC)C)',
    P: 'N1CCC[C@@H]1',
    F: 'N[C@@H](Cc1ccccc1)',
    W: 'N[C@@H](Cc1c[nH]c2ccccc12)',
    M: 'N[C@@H](CCSC)',
    S: 'N[C@@H](CO)',
    T: 'N[C@@H]([C@@H](O)C)',
    C: 'N[C@@H](CS)',
    Y: 'N[C@@H](Cc1ccc(O)cc1)',
    H: 'N[C@@H](Cc1cnc[nH]1)',
    D: 'N[C@@H](CC(=O)O)',
    E: 'N[C@@H](CCC(=O)O)',
    N: 'N[C@@H](CC(N)=O)',
    Q: 'N[C@@H](CCC(N)=O)',
    K: 'N[C@@H](CCCCN)',
    R: 'N[C@@H](CCCNC(=N)N)',
    // D-forms (lowercase)
    a: 'N[C@H](C)',
    g: 'NC',
    v: 'N[C@H](C(C)C)',
    l: 'N[C@H](CC(C)C)',
    i: 'N[C@H]([C@H](CC)C)',
    p: 'N1CCC[C@H]1',
    f: 'N[C@H](Cc1ccccc1)',
    w: 'N[C@H](Cc1c[nH]c2ccccc12)',
    m: 'N[C@H](CCSC)',
    s: 'N[C@H](CO)',
    t: 'N[C@H]([C@H](O)C)',
    c: 'N[C@H](CS)',
    y: 'N[C@H](Cc1ccc(O)cc1)',
    h: 'N[C@H](Cc1cnc[nH]1)',
    d: 'N[C@H](CC(=O)O)',
    e: 'N[C@H](CCC(=O)O)',
    n: 'N[C@H](CC(N)=O)',
    q: 'N[C@H](CCC(N)=O)',
    k: 'N[C@H](CCCCN)',
    r: 'N[C@H](CCCNC(=N)N)',
};

// N-terminal prefix SMILES (prepended before first residue N)
const N_TERMINAL = {
    'free': '',
    'ACT':  'CC(=O)',                   // Acetyl
    'For':  'C(=O)',                    // Formyl
    'C16':  'CCCCCCCCCCCCCCCC(=O)',     // Palmitoyl
};

// C-terminal suffix SMILES (appended after last residue's C(=O))
const C_TERMINAL = {
    'free': 'O',    // free carboxylic acid → C(=O)O
    'AMD':  'N',    // amide              → C(=O)N
    'OMe':  'OC',   // methyl ester       → C(=O)OC
};

/**
 * Convert a one-letter amino acid sequence to SMILES.
 *
 * @param {string} sequence  — one-letter codes; uppercase = L, lowercase = D
 * @param {string} nTerminal — key from N_TERMINAL ('free', 'ACT', 'For', 'C16')
 * @param {string} cTerminal — key from C_TERMINAL ('free', 'AMD', 'OMe')
 * @returns {{ smiles: string, warnings: string[] }}
 * @throws {Error} if sequence contains unsupported residue codes
 */
export function convertPeptideToSmiles(sequence, nTerminal = 'free', cTerminal = 'free') {
    const cleaned = sequence.replace(/\s/g, '');
    if (!cleaned.length) throw new Error('Sequence is empty');

    const residues = cleaned.split('');
    const unknown  = residues.filter(r => !(r in AA_BLOCKS));
    if (unknown.length) {
        throw new Error(
            `Unsupported residue code${unknown.length > 1 ? 's' : ''}: ` +
            `${[...new Set(unknown)].join(', ')}. ` +
            `Use uppercase for L-amino acids, lowercase for D-amino acids.`
        );
    }

    const prefix = N_TERMINAL[nTerminal] ?? '';
    const suffix = C_TERMINAL[cTerminal] ?? 'O';

    // Build linear chain: prefix + [N-Cα(sc)-C(=O)]×n + suffix
    const chain  = residues.map(r => AA_BLOCKS[r] + 'C(=O)').join('');
    const smiles = prefix + chain + suffix;

    const warnings = [];
    if (residues.length > 30) {
        warnings.push(
            'Long peptides (>30 residues) may exceed Ketcher rendering limits. ' +
            'Consider breaking into fragments.'
        );
    }

    return { smiles, warnings };
}

/**
 * Return the set of valid one-letter codes this converter supports.
 * @returns {Set<string>}
 */
export function supportedCodes() {
    return new Set(Object.keys(AA_BLOCKS));
}
