/**
 * fileparse.js
 *
 * Browser-side parsers for PDB (.pdb), mmCIF (.cif/.mmcif) and SD (.sdf) files.
 *
 * parseSdf(text) — splits an SD file into records of { title, molblock, data }
 *   plus the data-item names in file order. It does no chemistry; turning the
 *   mol blocks into SMILES is left to the caller's RDKit instance.
 *
 * The PDB and mmCIF parsers return:
 *   { compIds: string[], smilesMap: Record<string,string> }
 *
 *   compIds  — unique HETATM residue codes (filtered via LIGAND_EXCLUDE)
 *   smilesMap — comp_id → SMILES, populated from inline _pdbx_chem_comp_descriptor
 *               data when the file contains it (Boltz outputs, component CIFs, etc.)
 *
 * buildMolFromPdbCoords(text, compId) — coordinate-based fallback for files
 *   like docking pocket PDBs where the ligand code is generic ('LIG', 'MOL', …)
 *   and there is no inline SMILES.  Returns a V2000 mol-block string for RDKit.
 */

import { LIGAND_EXCLUDE } from './pdb.js';

// ----------------------------------------------------------------
// Covalent radii (Å) for bond detection
// ----------------------------------------------------------------
const COV_RADII = {
    H: 0.31, C: 0.76, N: 0.71, O: 0.66, F: 0.57,
    S: 1.05, P: 1.07, Cl: 1.02, Br: 1.20, I: 1.39,
    B: 0.84, Se: 1.20, Si: 1.11,
};
const BOND_SLACK = 1.30; // allow 30 % beyond sum of covalent radii

// comp_ids that signal a generic/renamed ligand (no CCD entry)
export const GENERIC_LIGAND_CODES = new Set(['LIG', 'MOL', 'UNL', 'UNK', 'DRG', 'INH', 'LGN', 'X']);

// ----------------------------------------------------------------
// PDB format (.pdb)
// ----------------------------------------------------------------

export function parsePdb(text) {
    const seen    = new Set();
    const compIds = [];

    for (const line of text.split('\n')) {
        if (!line.startsWith('HETATM')) continue;
        const resName = line.slice(17, 20).trim().toUpperCase();
        if (resName && !LIGAND_EXCLUDE.has(resName) && !seen.has(resName)) {
            seen.add(resName);
            compIds.push(resName);
        }
    }
    return { compIds, smilesMap: {} };
}

// ----------------------------------------------------------------
// mmCIF format (.cif / .mmcif)
// ----------------------------------------------------------------

export function parseCif(text) {
    return {
        compIds:   _extractHetatmCodes(text),
        smilesMap: _extractInlineSmiles(text),
    };
}

// ----------------------------------------------------------------
// SD format (.sdf / .sd)
//
// Records end at a `$$$$` line. Each holds a mol block (V2000 or V3000, up to
// `M  END`) followed by data items:
//
//   > <pIC50>          header — anything may surround the <name>,
//   6.2                e.g. `>  <ID>  (1)` or `> 25 <ID>`
//                      a blank line ends the value; values may span lines
//
// Returns { records: [{ title, molblock, data }], fields: string[] }.
// ----------------------------------------------------------------

export function parseSdf(text) {
    const records = [];
    const fields  = [];
    const seen    = new Set();

    // Consume the `$$$$` line with its newline, so each chunk starts exactly
    // on its title line — which may itself be blank.
    for (const chunk of text.split(/^\$\$\$\$[^\r\n]*(?:\r?\n|$)/m)) {
        if (!chunk.trim()) continue;

        const lines = chunk.split(/\r?\n/);
        let end = lines.findIndex(l => l.startsWith('M  END'));
        if (end === -1) end = lines.length - 1;

        const data = {};
        let key = null, buf = [];
        const flush = () => {
            if (key && !(key in data)) {
                data[key] = buf.join('\n').trim();
                if (!seen.has(key)) { seen.add(key); fields.push(key); }
            }
            key = null; buf = [];
        };

        for (let i = end + 1; i < lines.length; i++) {
            const line = lines[i];
            const head = line.match(/^>[^<]*<([^>]*)>/);
            if (head)               { flush(); key = head[1].trim(); continue; }
            if (key === null)       continue;
            if (line.trim() === '') { flush(); continue; }
            buf.push(line);
        }
        flush();

        records.push({
            title:    lines[0].trim(),
            molblock: lines.slice(0, end + 1).join('\n'),
            data,
        });
    }
    return { records, fields };
}

// ----------------------------------------------------------------
// Coordinate-based mol-block builder
//
// Works for PDB files that have no CONECT records and no inline SMILES,
// e.g. docking pocket files where the ligand code is 'LIG'.
//
// Returns a V2000 MDL mol-block string ready for RDKit.get_mol(), or null.
// ----------------------------------------------------------------

export function buildMolFromPdbCoords(pdbText, targetCompId) {
    const atoms = _extractLigandAtoms(pdbText, targetCompId);
    if (atoms.length < 2) return null;

    // Step 1: detect bonds
    const bonds = _detectBonds(atoms);          // [{a,b,distRatio}]
    if (!bonds.length) return null;

    // Step 2: build adjacency list
    const adj = atoms.map(() => []);
    bonds.forEach((bond, idx) => {
        adj[bond.a].push({ to: bond.b, bi: idx });
        adj[bond.b].push({ to: bond.a, bi: idx });
    });

    // Step 3: assign bond types (aromatic / double / single)
    const bondTypes = bonds.map((bond, idx) => {
        const { a, b, distRatio } = bond;
        const ea = atoms[a].element;
        const eb = atoms[b].element;

        // Check if bond is in a 5- or 6-membered ring
        const rsize = _ringSize(adj, idx, a, b);
        const inSmallRing = rsize === 5 || rsize === 6;

        // Aromatic: small ring, both atoms C/N/O/S, not suspiciously long
        if (inSmallRing && 'CNOS'.includes(ea) && 'CNOS'.includes(eb) && distRatio < 0.97) {
            return 4; // aromatic (V2000 type 4)
        }

        // Double bond: clearly short, not already handled as aromatic
        if (distRatio < 0.88) return 2;

        return 1; // single
    });

    // Step 4: emit V2000 mol block
    const na = atoms.length;
    const nb = bonds.length;

    let mol = `\n  BoltzDraw\n\n`;
    mol += `${String(na).padStart(3)}${String(nb).padStart(3)}  0  0  0  0  0  0  0  0999 V2000\n`;

    for (const { x, y, z, element } of atoms) {
        mol += `${x.toFixed(4).padStart(10)}${y.toFixed(4).padStart(10)}${z.toFixed(4).padStart(10)} `;
        mol += `${element.padEnd(3)} 0  0  0  0  0  0  0  0  0  0  0  0\n`;
    }

    bonds.forEach(({ a, b }, idx) => {
        const t = bondTypes[idx];
        mol += `${String(a + 1).padStart(3)}${String(b + 1).padStart(3)}${String(t).padStart(3)}  0  0  0  0\n`;
    });

    mol += 'M  END\n';
    return mol;
}

// ----------------------------------------------------------------
// Helpers — coordinate reconstruction
// ----------------------------------------------------------------

function _extractLigandAtoms(pdbText, targetCompId) {
    const atoms = [];
    for (const line of pdbText.split('\n')) {
        if (!line.startsWith('HETATM') && !line.startsWith('ATOM')) continue;
        if (line.slice(17, 20).trim().toUpperCase() !== targetCompId.toUpperCase()) continue;

        const x = parseFloat(line.slice(30, 38));
        const y = parseFloat(line.slice(38, 46));
        const z = parseFloat(line.slice(46, 54));

        // Prefer the element column (77-78), fall back to atom-name heuristic
        let element = line.slice(76, 78).trim();
        if (!element) {
            element = line.slice(12, 16).trim().replace(/\d/g, '').slice(0, 2);
        }
        element = element[0].toUpperCase() + (element.slice(1).toLowerCase() || '');

        if (!isNaN(x) && element) atoms.push({ x, y, z, element });
    }
    return atoms;
}

function _detectBonds(atoms) {
    const bonds = [];
    for (let i = 0; i < atoms.length; i++) {
        for (let j = i + 1; j < atoms.length; j++) {
            const ri = COV_RADII[atoms[i].element] ?? 0.80;
            const rj = COV_RADII[atoms[j].element] ?? 0.80;
            const maxDist = (ri + rj) * BOND_SLACK;

            const dx = atoms[i].x - atoms[j].x;
            const dy = atoms[i].y - atoms[j].y;
            const dz = atoms[i].z - atoms[j].z;
            const dist = Math.sqrt(dx * dx + dy * dy + dz * dz);

            if (dist <= maxDist) {
                bonds.push({ a: i, b: j, distRatio: dist / (ri + rj) });
            }
        }
    }
    return bonds;
}

/**
 * BFS to find the shortest ring containing a given bond.
 * Returns the ring size (e.g. 6 for benzene), or Infinity if not in a ring.
 */
function _ringSize(adj, bondIdx, a, b) {
    // BFS from a to b, avoiding bondIdx
    const dist = new Map([[a, 0]]);
    const q    = [a];
    while (q.length) {
        const cur = q.shift();
        for (const { to, bi } of adj[cur]) {
            if (bi === bondIdx) continue;
            if (dist.has(to)) continue;
            const d = dist.get(cur) + 1;
            if (to === b) return d + 1; // +1 for the original bond
            dist.set(to, d);
            q.push(to);
        }
    }
    return Infinity;
}

// ----------------------------------------------------------------
// Helpers — HETATM code extraction from mmCIF atom_site loop
// ----------------------------------------------------------------

function _extractHetatmCodes(text) {
    const seen    = new Set();
    const codes   = [];
    const lines   = text.split('\n');

    let inAtomSiteLoop = false;
    let headers        = [];
    let groupPdbCol    = -1;
    let compIdCol      = -1;

    for (const raw of lines) {
        const line = raw.trim();

        if (line === 'loop_') {
            inAtomSiteLoop = false;
            headers = []; groupPdbCol = -1; compIdCol = -1;
            continue;
        }

        if (line.startsWith('_atom_site.')) {
            inAtomSiteLoop = true;
            const field = line.toLowerCase();
            headers.push(field);
            if (field === '_atom_site.group_pdb')      groupPdbCol = headers.length - 1;
            if (field === '_atom_site.auth_comp_id')   compIdCol   = headers.length - 1;
            else if (field === '_atom_site.label_comp_id' && compIdCol === -1)
                                                       compIdCol   = headers.length - 1;
            continue;
        }

        if (inAtomSiteLoop && groupPdbCol >= 0 && compIdCol >= 0) {
            if (line === '#' || line.startsWith('loop_') || line.startsWith('data_') ||
                (line.startsWith('_') && !line.startsWith('_atom_site.'))) {
                inAtomSiteLoop = false;
                continue;
            }
            if (!line.length) continue;

            const parts = _splitCifLine(line);
            const maxCol = Math.max(groupPdbCol, compIdCol);
            if (parts.length <= maxCol) continue;

            if (parts[groupPdbCol] === 'HETATM') {
                const code = parts[compIdCol].toUpperCase();
                if (code && code !== '.' && !LIGAND_EXCLUDE.has(code) && !seen.has(code)) {
                    seen.add(code);
                    codes.push(code);
                }
            }
        }
    }
    return codes;
}

// ----------------------------------------------------------------
// Helpers — inline SMILES from _pdbx_chem_comp_descriptor
// ----------------------------------------------------------------

function _extractInlineSmiles(text) {
    const smilesMap = {};
    const lines     = text.split('\n');
    let i = 0;

    while (i < lines.length) {
        const line = lines[i].trim();

        // ---- Loop block ----
        if (line === 'loop_') {
            let j = i + 1;
            while (j < lines.length && !lines[j].trim()) j++;

            if (lines[j]?.trim().startsWith('_pdbx_chem_comp_descriptor.')) {
                const headers = [];
                while (j < lines.length && lines[j].trim().startsWith('_')) {
                    headers.push(lines[j].trim()
                        .replace(/^_pdbx_chem_comp_descriptor\./, '')
                        .toLowerCase());
                    j++;
                }
                const compCol = headers.indexOf('comp_id');
                const typeCol = headers.indexOf('type');
                const descCol = headers.indexOf('descriptor');

                if (compCol >= 0 && typeCol >= 0 && descCol >= 0) {
                    while (j < lines.length) {
                        const dataLine = lines[j].trim();
                        if (!dataLine || dataLine === '#' || dataLine.startsWith('_') ||
                            dataLine === 'loop_' || dataLine.startsWith('data_')) break;
                        if (dataLine.startsWith(';')) {
                            j++;
                            while (j < lines.length && !lines[j].trim().startsWith(';')) j++;
                            j++;
                            continue;
                        }
                        const parts = _splitCifLine(dataLine);
                        if (parts.length > Math.max(compCol, typeCol, descCol)) {
                            const cid  = parts[compCol]?.toUpperCase();
                            const type = parts[typeCol];
                            const desc = parts[descCol];
                            if (cid && _isSmilesType(type) && desc && desc !== '?' && !smilesMap[cid]) {
                                smilesMap[cid] = desc;
                            }
                        }
                        j++;
                    }
                }
                i = j;
                continue;
            }
        }

        // ---- Single-value key-value block ----
        if (line.startsWith('_pdbx_chem_comp_descriptor.')) {
            const kv = {};
            while (i < lines.length) {
                const cur = lines[i].trim();
                if (!cur.startsWith('_pdbx_chem_comp_descriptor.')) break;

                const spIdx = cur.search(/\s/);
                const key   = spIdx > 0 ? cur.slice(0, spIdx) : cur;
                const field = key.replace('_pdbx_chem_comp_descriptor.', '').toLowerCase();
                let value   = spIdx > 0 ? cur.slice(spIdx).trim() : '';

                if (!value) {
                    i++;
                    const nextRaw = lines[i]?.trim() ?? '';
                    if (nextRaw.startsWith(';')) {
                        const parts = [];
                        i++;
                        while (i < lines.length && !lines[i].trim().startsWith(';')) {
                            parts.push(lines[i]);
                            i++;
                        }
                        value = parts.join('\n').trim();
                    } else {
                        value = _unquoteCif(nextRaw);
                    }
                } else {
                    value = _unquoteCif(value);
                }

                kv[field] = value;
                i++;
            }

            const cid  = kv['comp_id']?.toUpperCase();
            const type = kv['type'];
            const desc = kv['descriptor'];
            if (cid && _isSmilesType(type) && desc && !smilesMap[cid]) {
                smilesMap[cid] = desc;
            }
            continue;
        }

        i++;
    }
    return smilesMap;
}

// ----------------------------------------------------------------
// CIF tokeniser helpers
// ----------------------------------------------------------------

function _splitCifLine(line) {
    const tokens = [];
    let i = 0;
    while (i < line.length) {
        while (i < line.length && (line[i] === ' ' || line[i] === '\t')) i++;
        if (i >= line.length) break;
        if (line[i] === '"') {
            i++;
            const s = i;
            while (i < line.length && line[i] !== '"') i++;
            tokens.push(line.slice(s, i));
            i++;
        } else if (line[i] === "'") {
            i++;
            const s = i;
            while (i < line.length && line[i] !== "'") i++;
            tokens.push(line.slice(s, i));
            i++;
        } else {
            const s = i;
            while (i < line.length && line[i] !== ' ' && line[i] !== '\t') i++;
            tokens.push(line.slice(s, i));
        }
    }
    return tokens;
}

function _unquoteCif(s) {
    s = s.trim();
    if ((s.startsWith('"') && s.endsWith('"')) ||
        (s.startsWith("'") && s.endsWith("'"))) {
        return s.slice(1, -1);
    }
    return s;
}

function _isSmilesType(t) {
    return t === 'SMILES' || t === 'SMILES_CANONICAL';
}
