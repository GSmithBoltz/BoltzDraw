/**
 * pdb.js
 *
 * Fetches small-molecule ligands from an RCSB PDB entry.
 * Uses the public, CORS-enabled RCSB Data API.
 *
 * Chain (fetchPdbLigands):
 *   1. GET /entry/{id}                   → non_polymer_entity_ids
 *   2. GET /nonpolymer_entity/{id}/{eid} → comp_id per entity
 *   3. fetchLigandsByCompIds()           → SMILES, formula, MW, image
 *
 * fetchLigandsByCompIds() is also exported so the file-upload tab
 * can reuse it after parsing comp_ids directly from a PDB/CIF file.
 */

const RCSB_BASE = 'https://data.rcsb.org/rest/v1/core';
const IMAGE_CDN = 'https://cdn.rcsb.org/images/ccd/unlabeled';

// Exclude water, common ions, buffers, and cryoprotectants
export const LIGAND_EXCLUDE = new Set([
    'HOH', 'WAT', 'DOD',                                // water / deuterium
    'SO4', 'PO4', 'GOL', 'EDO', 'EOH', 'PEG',          // cryoprotectants
    'MES', 'EPE', 'HED', 'TRS', 'BME',                  // buffers
    'CL',  'NA',  'MG',  'ZN',  'CA',  'K',  'FE',     // common ions
    'CU',  'MN',  'CO',  'NI',  'CD',  'HG', 'SE',     // metals
    'ACT', 'ACE', 'FOR', 'DMS', 'MSE',                  // acetate, DMSO, etc.
]);

/**
 * Return all unique small-molecule ligands for a PDB entry.
 *
 * @param {string} pdbId  - 4-char PDB accession code
 * @returns {Promise<Array<{comp_id, name, formula, mw, smiles, imageUrl}>>}
 * @throws {Error}  if the entry is not found or the API returns an error
 */
export async function fetchPdbLigands(pdbId) {
    const id = pdbId.toUpperCase();

    // 1. Entry metadata → nonpolymer entity IDs
    const entry = await _get(`${RCSB_BASE}/entry/${id}`);
    const entityIds = entry?.rcsb_entry_container_identifiers?.non_polymer_entity_ids ?? [];
    if (!entityIds.length) return [];

    // 2. Fetch each entity in parallel → comp_ids
    const entities = await Promise.all(
        entityIds.map(eid =>
            _get(`${RCSB_BASE}/nonpolymer_entity/${id}/${eid}`).catch(() => null)
        )
    );

    const rawIds = entities
        .map(e => e?.pdbx_entity_nonpoly?.comp_id)
        .filter(Boolean);

    // 3. Resolve to full ligand objects
    return fetchLigandsByCompIds(rawIds);
}

/**
 * Given an array of CCD comp_ids (3-letter codes), deduplicate, filter
 * exclusions, fetch chemcomp data, and return ligand objects.
 * Exported so the file-upload tab can call it directly.
 *
 * @param {string[]} rawCompIds
 * @returns {Promise<Array<{comp_id, name, formula, mw, smiles, imageUrl}>>}
 */
export async function fetchLigandsByCompIds(rawCompIds) {
    // Deduplicate and filter
    const seen    = new Set();
    const compIds = [];
    for (const cid of rawCompIds) {
        const c = cid?.toUpperCase?.();
        if (!c || LIGAND_EXCLUDE.has(c) || seen.has(c)) continue;
        seen.add(c);
        compIds.push(c);
    }
    if (!compIds.length) return [];

    // Fetch chemical component data in parallel
    const chemComps = await Promise.all(
        compIds.map(cid => _get(`${RCSB_BASE}/chemcomp/${cid}`).catch(() => null))
    );

    return compIds
        .map((comp_id, i) => {
            const cc      = chemComps[i];
            const name    = cc?.chem_comp?.name ?? comp_id;
            const formula = cc?.chem_comp?.formula ?? '';
            const mw      = cc?.chem_comp?.formula_weight ?? null;

            // Prefer canonical SMILES, fall back to any SMILES descriptor
            const descs  = cc?.pdbx_chem_comp_descriptor ?? [];
            const smiDes =
                descs.find(d => d.type === 'SMILES_CANONICAL') ??
                descs.find(d => d.type === 'SMILES') ??
                null;
            const smiles = smiDes?.descriptor ?? '';

            const imageUrl = `${IMAGE_CDN}/${comp_id[0].toLowerCase()}/${comp_id}.svg`;
            return { comp_id, name, formula, mw, smiles, imageUrl };
        })
        .filter(lig => lig.smiles);
}

async function _get(url) {
    const res = await fetch(url);
    if (!res.ok) throw new Error(`RCSB API ${res.status} — ${url}`);
    return res.json();
}
