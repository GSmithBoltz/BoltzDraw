/**
 * boltzcontacts.js
 *
 * BoltzContacts: protein–ligand interaction profiling with PLIP's or ProLIF's
 * rules, drawn in 3D.
 *
 * Two engines behind one view:
 *   Browser  contactengine.js, a JavaScript port of the rules. Works anywhere,
 *            including the static GitHub Pages build.
 *   Exact    the real PLIP / ProLIF, run by the local server (server.py /plip and
 *            /prolif). Offered only when the backend answers /ping.
 *
 * Several structures can be loaded at once (a set of Boltz poses, say); each is
 * profiled on its own and the poses tab lines their fingerprints up.
 */

import { parseStructure, findLigands, profile, PLIP_TYPES, PROLIF_TYPES, toTriageRecord, toCsvRows }
    from './contactengine.js';
import { ensureMolstarLoaded } from './molstar.js';

// ----------------------------------------------------------------
// Interaction vocabulary
// ----------------------------------------------------------------

// Colours follow the PLIP palette BoltzViewer already uses
const FAMILY = {
    hbond:       { color: '#2563eb', label: 'H-bond' },
    ionic:       { color: '#dc2626', label: 'Salt bridge' },
    pistack:     { color: '#7c3aed', label: 'π-stacking' },
    pication:    { color: '#b45309', label: 'π-cation' },
    halogen:     { color: '#16a34a', label: 'Halogen bond' },
    water:       { color: '#0891b2', label: 'Water bridge' },
    metal:       { color: '#9d174d', label: 'Metal' },
    hydrophobic: { color: '#6b6b6b', label: 'Hydrophobic' },
    vdw:         { color: '#b8b2a7', label: 'vdW contact' },
};
const FAMILY_OF = {
    hbond: 'hbond', HBDonor: 'hbond', HBAcceptor: 'hbond',
    saltbridge: 'ionic', Cationic: 'ionic', Anionic: 'ionic',
    pistacking: 'pistack', PiStacking: 'pistack',
    pication: 'pication', CationPi: 'pication', PiCation: 'pication',
    halogen: 'halogen', XBDonor: 'halogen',
    waterbridge: 'water', metal: 'metal',
    hydrophobic: 'hydrophobic', Hydrophobic: 'hydrophobic',
    VdWContact: 'vdw',
};
const TYPE_LABEL = {
    hbond: 'Hydrogen bonds', saltbridge: 'Salt bridges', pistacking: 'π-stacking', pication: 'π-cation',
    halogen: 'Halogen bonds', waterbridge: 'Water bridges', hydrophobic: 'Hydrophobic contacts', metal: 'Metal complexes',
    HBDonor: 'HBDonor · ligand donates', HBAcceptor: 'HBAcceptor · ligand accepts',
    Cationic: 'Cationic · ligand +', Anionic: 'Anionic · ligand −', PiStacking: 'PiStacking',
    CationPi: 'CationPi · ligand +, protein ring', PiCation: 'PiCation · ligand ring, protein +',
    XBDonor: 'XBDonor · ligand halogen', Hydrophobic: 'Hydrophobic', VdWContact: 'VdWContact',
};
const f2 = (v) => (v == null ? '–' : Number(v).toFixed(2));
const f0 = (v) => (v == null ? '–' : Math.round(v));
const yes = (v) => (v ? 'yes' : 'no');
const COLUMNS = {
    hbond: [['D–A', x => f2(x.detail.dist_d_a)], ['H–A', x => f2(x.detail.dist_h_a)], ['Angle', x => f0(x.detail.don_angle)],
            ['Donor', x => (x.detail.protisdon ? 'protein' : 'ligand')], ['Side chain', x => yes(x.detail.sidechain)]],
    saltbridge: [['Dist', x => f2(x.dist)], ['Protein', x => (x.detail.protispos ? '+' : '−')], ['Ligand group', x => x.detail.lig_group || '–']],
    pistacking: [['Dist', x => f2(x.dist)], ['Angle', x => f0(x.detail.angle)], ['Offset', x => f2(x.detail.offset)], ['Type', x => x.detail.type]],
    pication: [['Dist', x => f2(x.dist)], ['Offset', x => f2(x.detail.offset)], ['Charge on', x => (x.detail.protcharged ? 'protein' : 'ligand')]],
    halogen: [['Dist', x => f2(x.dist)], ['C–X…A', x => f0(x.detail.don_angle)], ['Y–A…X', x => f0(x.detail.acc_angle)]],
    waterbridge: [['A–W', x => f2(x.detail.dist_a_w)], ['D–W', x => f2(x.detail.dist_d_w)], ['W angle', x => f0(x.detail.water_angle)],
                  ['Donor', x => (x.detail.protisdon ? 'protein' : 'ligand')]],
    hydrophobic: [['Dist', x => f2(x.dist)]],
    metal: [['Dist', x => f2(x.dist)], ['Metal', x => x.detail.metal_type || '–']],
    PiStacking: [['Dist', x => f2(x.dist)], ['Plane', x => f0(x.detail.plane_angle)], ['Subtype', x => x.detail.subtype || '–']],
    CationPi: [['Dist', x => f2(x.dist)], ['Angle', x => f0(x.detail.angle)]],
    PiCation: [['Dist', x => f2(x.dist)], ['Angle', x => f0(x.detail.angle)]],
    HBDonor: [['D–A', x => f2(x.dist)], ['Angle', x => f0(x.detail.DHA_angle)]],
    HBAcceptor: [['D–A', x => f2(x.dist)], ['Angle', x => f0(x.detail.DHA_angle)]],
    XBDonor: [['Dist', x => f2(x.dist)], ['A…X–D', x => f0(x.detail.AXD_angle)], ['X…A–R', x => f0(x.detail.XAR_angle)]],
};
const DEFAULT_COLUMNS = [['Dist', x => f2(x.dist)]];

// ----------------------------------------------------------------
// State
// ----------------------------------------------------------------

let _ctx = null;              // { backend: () => bool, switchMode }
let _poses = [];              // [{ name, text, st, ligands, ligandId, results: {key: result|{error}} }]
let _cur = -1;
let _rules = 'plip';          // 'plip' | 'prolif'
let _engine = 'browser';      // 'browser' | 'exact'
let _tab = 'interactions';
const _hidden = new Set(['vdw']);   // families not drawn in 3D

let _viewer = null;
let _drawToken = 0;

const $ = (id) => document.getElementById(id);
const esc = (s) => String(s ?? '').replace(/[&<>"']/g, c => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;' }[c]));
const resultKey = () => `${_rules}|${_engine}`;

// ----------------------------------------------------------------
// Init
// ----------------------------------------------------------------

export function initBoltzContacts(ctx) {
    _ctx = ctx;
    const input = $('bc-file');
    input.addEventListener('change', (e) => { _addFiles([...e.target.files]); input.value = ''; });
    $('bc-add-file').addEventListener('change', (e) => { _addFiles([...e.target.files]); e.target.value = ''; });

    const stage = $('bc-stage');
    stage.addEventListener('dragover', (e) => { e.preventDefault(); stage.classList.add('bc-dragging'); });
    stage.addEventListener('dragleave', (e) => { if (e.target === stage) stage.classList.remove('bc-dragging'); });
    stage.addEventListener('drop', (e) => {
        e.preventDefault();
        stage.classList.remove('bc-dragging');
        _addFiles([...e.dataTransfer.files]);
    });

    const fetchId = () => _fetchPdb($('bc-pdbid').value);
    $('bc-fetch-btn').addEventListener('click', fetchId);
    $('bc-pdbid').addEventListener('keydown', (e) => { if (e.key === 'Enter') fetchId(); });
    document.querySelectorAll('[data-bc-example]').forEach(b =>
        b.addEventListener('click', () => _fetchPdb(b.dataset.bcExample)));

    document.querySelectorAll('[data-bc-rules]').forEach(b => b.addEventListener('click', () => {
        _rules = b.dataset.bcRules;
        _refresh();
    }));
    document.querySelectorAll('[data-bc-engine]').forEach(b => b.addEventListener('click', () => {
        if (b.disabled) return;
        _engine = b.dataset.bcEngine;
        _refresh();
    }));
    document.querySelectorAll('[data-bc-tab]').forEach(b => b.addEventListener('click', () => {
        _tab = b.dataset.bcTab;
        _renderTabs();
    }));
    $('bc-smiles').addEventListener('change', (e) => {
        const p = _poses[_cur];
        if (!p) return;
        p.smiles = e.target.value.trim();
        delete p.results['prolif|exact'];
        _refresh();
    });
    $('bc-ligand').addEventListener('change', (e) => {
        const p = _poses[_cur];
        if (!p) return;
        p.ligandId = e.target.value;
        p.results = {};
        _refresh();
    });
    $('bc-export-jsonl').addEventListener('click', _exportJsonl);
    $('bc-export-csv').addEventListener('click', _exportCsv);
    $('bc-send-triage').addEventListener('click', _sendToTriage);
    $('bc-clear').addEventListener('click', () => {
        _poses = []; _cur = -1;
        _viewer?.plugin.clear();
        _refresh();
    });
    $('bc-run-all').addEventListener('click', _runAll);

    _refresh();
}

/** Called each time the mode is entered: the backend probe may have finished since. */
export function onEnterContacts() {
    _renderEngineButtons();
}

// ----------------------------------------------------------------
// Loading
// ----------------------------------------------------------------

async function _addFiles(files) {
    const ok = files.filter(f => /\.(cif|mmcif|pdb|ent)$/i.test(f.name));
    if (!ok.length) { _status('Drop mmCIF or PDB files (.cif, .pdb).', 'bad'); return; }
    let added = 0;
    for (const f of ok) {
        try {
            _addPose(f.name, await f.text());
            added++;
        } catch (err) {
            _status(`${f.name}: ${err.message}`, 'bad');
        }
    }
    if (added) {
        _status(`${added} structure${added > 1 ? 's' : ''} loaded.`, 'ok');
        _select(_poses.length - added);
    }
}

async function _fetchPdb(raw) {
    const id = String(raw || '').trim().toUpperCase();
    if (!/^[0-9][A-Z0-9]{3}$/.test(id)) { _status('Enter a 4-character PDB ID, e.g. 1STP.', 'bad'); return; }
    _status(`Fetching ${id} from the RCSB…`, 'busy');
    try {
        const res = await fetch(`https://files.rcsb.org/download/${id}.cif`);
        if (!res.ok) throw new Error(`RCSB answered ${res.status}`);
        _addPose(`${id}.cif`, await res.text());
        _status(`${id} loaded.`, 'ok');
        _select(_poses.length - 1);
    } catch (err) {
        _status(`Could not fetch ${id}: ${err.message}`, 'bad');
    }
}

function _addPose(name, text) {
    const st = parseStructure(text, name);
    const ligands = findLigands(st);
    if (!ligands.length) throw new Error('no ligand or peptide found');
    _poses.push({ name, text, st, ligands, ligandId: ligands[0].id, results: {} });
}

function _select(i) {
    _cur = Math.max(0, Math.min(i, _poses.length - 1));
    _refresh({ reload: true });
}

// ----------------------------------------------------------------
// Analysis
// ----------------------------------------------------------------

async function _analyse(pose) {
    const key = resultKey();
    if (pose.results[key]) return pose.results[key];
    let result;
    try {
        result = _engine === 'exact' ? await _runExact(pose) : profile(pose.st, pose.ligandId, { rules: _rules });
    } catch (err) {
        result = { error: err.message };
    }
    pose.results[key] = result;
    return result;
}

async function _runAll() {
    const btn = $('bc-run-all');
    btn.disabled = true;
    for (let i = 0; i < _poses.length; i++) {
        btn.textContent = `Profiling ${i + 1}/${_poses.length}…`;
        await _analyse(_poses[i]);
        if (i % 10 === 9) await new Promise(r => setTimeout(r, 0));
    }
    btn.disabled = false;
    btn.textContent = 'Profile all poses';
    _renderTabs();
    _renderPoseList();
}

/** Exact PLIP / ProLIF via the local server, mapped onto the browser engine's records. */
async function _runExact(pose) {
    const lig = pose.ligands.find(l => l.id === pose.ligandId) || pose.ligands[0];
    if (lig.kind === 'peptide') throw new Error('The exact engines take small-molecule ligands; use the browser engine for peptides.');
    const url = _rules === 'plip' ? '/plip' : '/prolif';
    const headers = { 'Content-Type': 'text/plain', 'X-Filename': pose.name, 'X-Ligand': pose.ligandId };
    if (_rules === 'prolif' && pose.smiles) headers['X-Smiles'] = pose.smiles;
    const res = await fetch(url, { method: 'POST', headers, body: pose.text });
    const data = await res.json().catch(() => ({}));
    if (!res.ok || data.error) throw new Error(data.error || `server answered ${res.status}`);
    const interactions = _rules === 'plip' ? _mapPlip(data, pose, lig) : _mapProlif(data, pose, lig);
    const order = _rules === 'plip' ? PLIP_TYPES : PROLIF_TYPES;
    interactions.sort((a, b) => order.indexOf(a.type) - order.indexOf(b.type) || a.chain.localeCompare(b.chain) || a.resnr - b.resnr);
    return {
        ligand: { id: lig.id, kind: lig.kind, resname: lig.resname, chain: lig.chain, resnr: lig.resnr, nHeavy: lig.nHeavy, atoms: lig.atoms },
        rules: _rules, interactions, exact: true, hydrogens: 'added by the server',
        warnings: data.warnings || [], bondOrders: data.bond_orders || null,
    };
}

const PLIP_KEYS = { hbonds: 'hbond', hydrophobic: 'hydrophobic', pistacking: 'pistacking', pication: 'pication',
                    saltbridges: 'saltbridge', waterbridges: 'waterbridge', halogenbonds: 'halogen', metalcomplexes: 'metal' };

function _mapPlip(data, pose, lig) {
    const sites = data.sites || [];
    const site = sites.find(s => s.id === pose.ligandId) ||
                 sites.find(s => s.ligand && s.ligand.chain === lig.chain && +s.ligand.resnr === +lig.resnr) || sites[0];
    if (!site) return [];
    const out = [];
    for (const [key, list] of Object.entries(site.interactions || {})) {
        const type = PLIP_KEYS[key];
        if (!type) continue;
        for (const x of list) {
            const rec = _locate(pose, lig, x.reschain, x.resnr, x.ligcoo, x.protcoo);
            const dist = x.dist ?? x.dist_d_a ?? x.cent_dist ?? x.dist_a_w;
            const detail = { ...x };
            if (type === 'pistacking') detail.type = x.type;
            out.push({ type, chain: x.reschain, resnr: x.resnr, icode: '', restype: x.restype,
                       dist: dist == null ? null : +dist, detail, ...rec });
        }
    }
    return out;
}

function _mapProlif(data, pose, lig) {
    return (data.interactions || []).map(x => ({
        type: x.itype, chain: x.chain, resnr: x.resnr, icode: '', restype: x.restype,
        dist: x.dist == null ? null : +x.dist,
        detail: { distance: x.dist, subtype: x.subtype, DHA_angle: x.DHA_angle, angle: x.angle, plane_angle: x.plane_angle },
        ..._locate(pose, lig, x.chain, x.resnr, x.ligcoo, x.protcoo),
    }));
}

/** Atoms behind a server-reported coordinate list (an atom, or a ring / charge centre). */
function _locate(pose, lig, chain, resnr, ligcoo, protcoo) {
    const { atoms, residues } = pose.st;
    const ligSet = new Set(lig.atoms);
    const res = residues.find(r => r.chain === String(chain) && r.resnr === +resnr);
    const pick = (pts, pool) => {
        const out = new Set();
        for (const p of pts || []) {
            let best = -1, bestD = Infinity;
            for (const i of pool) {
                const a = atoms[i];
                if (a.el === 'H') continue;
                const d = Math.hypot(a.x - p[0], a.y - p[1], a.z - p[2]);
                if (d < bestD) { bestD = d; best = i; }
            }
            if (best < 0) continue;
            if (bestD < 0.3) out.add(best);
            else for (const i of pool) {             // a centre: the atoms around it
                const a = atoms[i];
                if (a.el !== 'H' && Math.hypot(a.x - p[0], a.y - p[1], a.z - p[2]) < 2.1) out.add(i);
            }
        }
        return [...out];
    };
    const ctr = (pts) => (pts && pts.length ? pts.reduce((c, p) => [c[0] + p[0] / pts.length, c[1] + p[1] / pts.length, c[2] + p[2] / pts.length], [0, 0, 0]) : null);
    return {
        lig: pick(ligcoo, [...ligSet]),
        prot: res ? pick(protcoo, res.atoms) : [],
        ligPt: ctr(ligcoo), protPt: ctr(protcoo),
    };
}

// ----------------------------------------------------------------
// Rendering
// ----------------------------------------------------------------

async function _refresh(opts = {}) {
    const has = _poses.length > 0;
    $('bc-empty').classList.toggle('hidden', has);
    $('bc-workspace').classList.toggle('hidden', !has);
    $('bc-run-all').classList.toggle('hidden', _poses.length < 2);
    document.querySelectorAll('[data-bc-rules]').forEach(b => b.classList.toggle('active', b.dataset.bcRules === _rules));
    _renderEngineButtons();
    $('bc-export-jsonl').classList.toggle('hidden', _rules !== 'plip');
    $('bc-send-triage').classList.toggle('hidden', _rules !== 'plip');
    if (!has) { _renderLegend(null); return; }

    const pose = _poses[_cur];
    _renderPoseList();
    _renderLigandSelect(pose);
    $('bc-results').innerHTML = '<p class="bc-muted">Profiling…</p>';
    const result = await _analyse(pose);
    if (pose !== _poses[_cur]) return;                 // the user moved on meanwhile
    _renderSummary(result);
    _renderTabs();
    _renderPoseList();
    const reload = opts.reload || !_viewer || _viewer._pose !== pose;
    if (reload) await _showStructure(pose);
    _drawInteractions(result, { frame: reload });
}

function _renderEngineButtons() {
    const backend = !!_ctx?.backend();
    const exact = document.querySelector('[data-bc-engine="exact"]');
    exact.disabled = !backend;
    exact.title = backend ? `Run the real ${_rules === 'plip' ? 'PLIP' : 'ProLIF'} on the local server`
                          : 'Needs the local server (python server.py); not available on GitHub Pages';
    if (!backend && _engine === 'exact') _engine = 'browser';
    document.querySelectorAll('[data-bc-engine]').forEach(b => b.classList.toggle('active', b.dataset.bcEngine === _engine));
    const smiles = $('bc-smiles');
    smiles.classList.toggle('hidden', !(_rules === 'prolif' && _engine === 'exact'));
    smiles.value = _poses[_cur]?.smiles || '';
    const tool = _rules === 'plip' ? 'PLIP 3.0' : 'ProLIF 2.2';
    $('bc-engine-note').innerHTML = _engine === 'exact'
        ? `The real ${tool}, run by the local server. Hydrogens are added there (PDBFixer for the protein, OpenBabel for the ligand).`
        : `${tool}'s rules and thresholds, ported to JavaScript so this runs anywhere. ` +
          `Where the file has no hydrogens, donor hydrogens are placed from the geometry.`;
}

function _renderPoseList() {
    const el = $('bc-poses');
    el.innerHTML = _poses.map((p, i) => {
        const r = p.results[resultKey()];
        const n = r && !r.error ? r.interactions.filter(x => x.type !== 'VdWContact').length : null;
        const badge = r?.error ? '<span class="bc-pose-n bc-bad">error</span>'
                    : n == null ? '' : `<span class="bc-pose-n">${n}</span>`;
        return `<button class="bc-pose${i === _cur ? ' active' : ''}" data-i="${i}" title="${esc(p.name)}">
                    <span class="bc-pose-name">${esc(p.name)}</span>${badge}
                    <span class="bc-pose-x" data-x="${i}" title="Remove">×</span>
                </button>`;
    }).join('');
    el.querySelectorAll('.bc-pose').forEach(b => b.addEventListener('click', (e) => {
        const x = e.target.closest('[data-x]');
        if (x) {
            const k = +x.dataset.x;
            _poses.splice(k, 1);
            if (!_poses.length) { _cur = -1; _viewer?.plugin.clear(); _refresh(); return; }
            _select(k <= _cur ? _cur - 1 : _cur);
            return;
        }
        _select(+b.dataset.i);
    }));
    $('bc-pose-count').textContent = _poses.length > 1 ? `${_poses.length} structures` : '1 structure';
}

function _renderLigandSelect(pose) {
    $('bc-ligand').innerHTML = pose.ligands.map(l =>
        `<option value="${esc(l.id)}"${l.id === pose.ligandId ? ' selected' : ''}>${esc(l.label)}</option>`).join('');
}

function _renderSummary(result) {
    const el = $('bc-summary');
    if (result.error) {
        el.innerHTML = `<div class="bc-status bc-bad">${esc(result.error)}</div>`;
        _renderLegend(null);
        return;
    }
    const counts = {};
    for (const x of result.interactions) {
        const fam = FAMILY_OF[x.type] || 'hydrophobic';
        counts[fam] = (counts[fam] || 0) + 1;
    }
    const residues = new Set(result.interactions.filter(x => x.type !== 'VdWContact').map(x => `${x.chain}:${x.resnr}`));
    const approx = result.interactions.some(x => x.approx);
    el.innerHTML = `
        <div class="bc-chips">
            ${Object.keys(FAMILY).filter(f => counts[f]).map(f => `
                <button class="bc-chip${_hidden.has(f) ? ' off' : ''}" data-fam="${f}" title="Show or hide in 3D">
                    <span class="bc-dot" style="background:${FAMILY[f].color}"></span>${FAMILY[f].label}
                    <strong>${counts[f]}</strong>
                </button>`).join('') || '<span class="bc-muted">No interactions found.</span>'}
        </div>
        <p class="bc-meta">${residues.size} contacting residues${result.ms != null ? ` · ${result.ms} ms` : ''}
           ${approx ? ' · <span class="bc-approx" title="The structure has no hydrogens on these donors, so their positions were placed from the heavy atoms">H placed from geometry</span>' : ''}
           ${result.exact ? ' · <span class="bc-exact">exact</span>' : ''}
           ${result.bondOrders === 'perceived' ? ' · ligand bond orders perceived from the geometry' : ''}</p>
        ${(result.warnings || []).map(w => `<p class="bc-meta bc-warn">${esc(w)}</p>`).join('')}`;
    el.querySelectorAll('.bc-chip').forEach(b => b.addEventListener('click', () => {
        const f = b.dataset.fam;
        if (_hidden.has(f)) _hidden.delete(f); else _hidden.add(f);
        b.classList.toggle('off', _hidden.has(f));
        _drawInteractions(result);
    }));
    _renderLegend(result);
}

function _renderLegend(result) {
    const el = $('bc-legend');
    if (!result || result.error) { el.innerHTML = ''; return; }
    const fams = [...new Set(result.interactions.map(x => FAMILY_OF[x.type]))].filter(f => !_hidden.has(f));
    el.innerHTML = fams.map(f => `<span><i style="background:${FAMILY[f].color}"></i>${FAMILY[f].label}</span>`).join('');
}

function _renderTabs() {
    document.querySelectorAll('[data-bc-tab]').forEach(b => b.classList.toggle('active', b.dataset.bcTab === _tab));
    const pose = _poses[_cur];
    const result = pose?.results[resultKey()];
    const el = $('bc-results');
    if (!pose) { el.innerHTML = ''; return; }
    if (_tab === 'poses') { el.innerHTML = _posesMatrix(); _bindMatrix(el); return; }
    if (!result) { el.innerHTML = '<p class="bc-muted">Profiling…</p>'; return; }
    if (result.error) { el.innerHTML = `<p class="bc-muted">${esc(result.error)}</p>`; return; }
    el.innerHTML = _tab === 'residues' ? _residueTable(result) : _interactionTables(result);
    el.querySelectorAll('[data-row]').forEach(tr => tr.addEventListener('click', () => {
        el.querySelectorAll('tr.sel').forEach(r => r.classList.remove('sel'));
        tr.classList.add('sel');
        const [chain, resnr] = tr.dataset.row.split('|');
        _focusResidue(chain, +resnr);
    }));
}

const resLabel = (x) => `${esc(x.restype)}${x.resnr}${esc(x.icode || '')}<span class="bc-chain">${esc(x.chain)}</span>`;

function _interactionTables(result) {
    const order = _rules === 'plip' ? PLIP_TYPES : PROLIF_TYPES;
    const groups = new Map();
    for (const x of result.interactions) {
        if (!groups.has(x.type)) groups.set(x.type, []);
        groups.get(x.type).push(x);
    }
    const types = [...groups.keys()].sort((a, b) => order.indexOf(a) - order.indexOf(b));
    if (!types.length) return '<p class="bc-muted">No interactions within the cut-offs.</p>';
    return types.map(t => {
        const list = groups.get(t);
        const cols = COLUMNS[t] || DEFAULT_COLUMNS;
        const fam = FAMILY[FAMILY_OF[t]] || FAMILY.hydrophobic;
        return `<details class="bc-group" ${t === 'VdWContact' ? '' : 'open'}>
            <summary><span class="bc-dot" style="background:${fam.color}"></span>${esc(TYPE_LABEL[t] || t)}
                <span class="bc-count">${list.length}</span></summary>
            <table class="bc-table">
                <thead><tr><th>Residue</th>${cols.map(c => `<th>${c[0]}</th>`).join('')}</tr></thead>
                <tbody>${list.map(x => `<tr data-row="${esc(x.chain)}|${x.resnr}"${x.approx ? ' class="approx"' : ''}>
                    <td>${resLabel(x)}</td>${cols.map(c => `<td>${esc(c[1](x))}</td>`).join('')}</tr>`).join('')}
                </tbody>
            </table>
        </details>`;
    }).join('');
}

function _residueTable(result) {
    const m = new Map();
    for (const x of result.interactions) {
        const k = `${x.chain}|${x.resnr}`;
        if (!m.has(k)) m.set(k, { ...x, types: new Map(), min: Infinity });
        const r = m.get(k);
        r.types.set(x.type, (r.types.get(x.type) || 0) + 1);
        if (x.dist != null) r.min = Math.min(r.min, x.dist);
    }
    const rows = [...m.values()].sort((a, b) => a.chain.localeCompare(b.chain) || a.resnr - b.resnr);
    return `<table class="bc-table bc-restable">
        <thead><tr><th>Residue</th><th>Interactions</th><th>Closest</th></tr></thead>
        <tbody>${rows.map(r => `<tr data-row="${esc(r.chain)}|${r.resnr}">
            <td>${resLabel(r)}</td>
            <td>${[...r.types].map(([t, n]) => {
                const fam = FAMILY[FAMILY_OF[t]] || FAMILY.hydrophobic;
                return `<span class="bc-tag" style="--c:${fam.color}">${esc(t)}${n > 1 ? ` ×${n}` : ''}</span>`;
            }).join('')}</td>
            <td>${f2(r.min)}</td></tr>`).join('')}
        </tbody></table>`;
}

/** Residue-by-pose grid: which family each pose makes with each residue. */
function _posesMatrix() {
    const done = _poses.filter(p => p.results[resultKey()] && !p.results[resultKey()].error);
    if (_poses.length < 2) return '<p class="bc-muted">Load several poses (one file each) to line their fingerprints up here.</p>';
    if (done.length < _poses.length) {
        return `<p class="bc-muted">${done.length} of ${_poses.length} poses profiled with these settings.
                <button class="bc-link" id="bc-run-all-inline">Profile all</button></p>` + (done.length ? _matrixTable(done) : '');
    }
    return _matrixTable(done);
}

function _matrixTable(poses) {
    const cols = new Map();
    for (const p of poses) for (const x of p.results[resultKey()].interactions) {
        if (x.type === 'VdWContact') continue;
        const k = `${x.chain}|${x.resnr}`;
        if (!cols.has(k)) cols.set(k, { chain: x.chain, resnr: x.resnr, restype: x.restype, n: 0 });
    }
    for (const p of poses) {
        const seen = new Set(p.results[resultKey()].interactions.filter(x => x.type !== 'VdWContact').map(x => `${x.chain}|${x.resnr}`));
        for (const k of seen) cols.get(k).n++;
    }
    const keys = [...cols.keys()].sort((a, b) => { const A = cols.get(a), B = cols.get(b); return A.chain.localeCompare(B.chain) || A.resnr - B.resnr; });
    const famOrder = Object.keys(FAMILY);
    const cell = (p, k) => {
        const fams = new Set(p.results[resultKey()].interactions.filter(x => `${x.chain}|${x.resnr}` === k).map(x => FAMILY_OF[x.type]));
        fams.delete('vdw');
        return [...fams].sort((a, b) => famOrder.indexOf(a) - famOrder.indexOf(b))
            .map(f => `<i style="background:${FAMILY[f].color}" title="${FAMILY[f].label}"></i>`).join('');
    };
    return `<div class="bc-matrix-wrap"><table class="bc-matrix">
        <thead><tr><th></th>${keys.map(k => { const c = cols.get(k); return `<th title="${esc(c.restype)}${c.resnr} ${esc(c.chain)}"><span>${esc(c.restype)}${c.resnr}</span></th>`; }).join('')}</tr>
        <tr class="bc-freq"><td>in % of poses</td>${keys.map(k => `<td>${Math.round(100 * cols.get(k).n / poses.length)}</td>`).join('')}</tr></thead>
        <tbody>${poses.map(p => `<tr data-pose="${_poses.indexOf(p)}"${_poses.indexOf(p) === _cur ? ' class="sel"' : ''}>
            <td title="${esc(p.name)}">${esc(p.name)}</td>${keys.map(k => `<td>${cell(p, k)}</td>`).join('')}</tr>`).join('')}
        </tbody></table></div>`;
}

function _bindMatrix(el) {
    el.querySelector('#bc-run-all-inline')?.addEventListener('click', _runAll);
    el.querySelectorAll('[data-pose]').forEach(tr => tr.addEventListener('click', () => _select(+tr.dataset.pose)));
}

function _status(msg, kind) {
    const el = $('bc-load-status');
    el.textContent = msg;
    el.className = `bc-status bc-${kind}`;
    el.style.display = msg ? 'block' : 'none';
}

// ----------------------------------------------------------------
// 3D view
// ----------------------------------------------------------------

async function _ensureViewer() {
    if (_viewer) return _viewer;
    await ensureMolstarLoaded();
    _viewer = await window.molstar.Viewer.create('bc-viewport', {
        layoutShowControls: false, layoutShowRemoteState: false, layoutShowSequence: false,
        layoutShowLog: false, layoutShowLeftPanel: false, layoutIsExpanded: false,
        viewportShowExpand: false, viewportShowSelectionMode: false, viewportShowAnimation: false,
    });
    try {
        await _viewer.plugin.canvas3d?.setProps({ renderer: { backgroundColor: 0xf7f5f1 } });
    } catch { /* cosmetic */ }
    // The focus representation shows the contacting residues as sticks; keep it
    // to the target alone, without Mol*'s own surroundings and interactions
    try {
        const ref = 'ms-plugin.create-structure-focus-representation';
        const beh = _viewer.plugin.state.behaviors;
        if (beh.cells.has(ref)) {
            const b = beh.build().to(ref).update(old => {
                const t = old.targetParams;
                return {
                    ...old,
                    components: ['target'],
                    targetParams: { ...t, type: { ...t.type, params: { ...t.type.params, xrayShaded: false, sizeFactor: 0.2 } } },
                };
            });
            await _viewer.plugin.runTask(beh.updateTree(b));
        }
    } catch (err) { console.warn('[BoltzContacts] focus setup failed:', err); }
    return _viewer;
}

async function _showStructure(pose) {
    const v = await _ensureViewer();
    const plugin = v.plugin;
    await plugin.clear();
    const format = /\.(pdb|ent)$/i.test(pose.name) ? 'pdb' : 'mmcif';
    // The asymmetric unit, not the biological assembly: the profile is computed
    // on the file's own coordinates, and an assembly would repeat every chain
    const data = await plugin.builders.data.rawData({ data: pose.text, label: pose.name });
    const trajectory = await plugin.builders.structure.parseTrajectory(data, format);
    await plugin.builders.structure.hierarchy.applyPreset(trajectory, 'default', {
        structure: { name: 'model', params: {} }, showUnitcell: false, representationPreset: 'auto',
    });
    v._pose = pose;
    // Waters crowd the pocket (bridging ones come back as sticks), and Mol*'s own
    // non-covalent interactions would compete with ours. Hiding is reliable in
    // the slim bundle where deleting is not; children must be hidden explicitly.
    const hide = (cell) => plugin.state.data.updateCellState(cell.transform.ref, { isHidden: true });
    for (const s of plugin.managers.structure.hierarchy.current.structures) {
        for (const c of s.components) {
            const key = c.key || '';
            if (key.includes('water')) { hide(c.cell); c.representations.forEach(r => hide(r.cell)); continue; }
            for (const r of c.representations) {
                if (r.cell?.transform?.params?.type?.name === 'interactions') hide(r.cell);
                // Boltz yellow for the ligand's carbons
                if (key.includes('ligand') || key.includes('non-polymer')) {
                    try {
                        await plugin.build().to(r.cell).update(old => {
                            old.colorTheme = { name: 'element-symbol', params: { carbonColor: { name: 'uniform', params: { value: 0xf2b600 } } } };
                        }).commit();
                    } catch { /* cosmetic */ }
                }
            }
        }
    }
}

/** Lookup from (chain, residue, atom name) to the Mol* unit and unit-local index. */
function _atomIndex(plugin) {
    const structure = plugin.managers.structure.hierarchy.current.structures[0]?.cell?.obj?.data;
    if (!structure?.units) return null;
    const map = new Map();
    for (const unit of structure.units) {
        const h = unit.model?.atomicHierarchy;
        if (!h) continue;
        const els = unit.elements;
        for (let k = 0; k < els.length; k++) {
            const e = els[k];
            const ri = h.residueAtomSegments.index[e];
            const key = `${h.chains.auth_asym_id.value(h.chainAtomSegments.index[e])}|${h.residues.auth_seq_id.value(ri)}|` +
                        `${h.residues.pdbx_PDB_ins_code.value(ri) || ''}|${h.atoms.auth_atom_id.value(e) || h.atoms.label_atom_id.value(e)}`;
            if (!map.has(key)) map.set(key, { unit, k });
        }
    }
    return { structure, map };
}

function _lociFor(index, st, atomIdxs) {
    const byUnit = new Map();
    for (const i of atomIdxs) {
        const a = st.atoms[i];
        const hit = index.map.get(`${a.chain}|${a.resnr}|${a.icode || ''}|${a.name}`);
        if (!hit) continue;
        if (!byUnit.has(hit.unit)) byUnit.set(hit.unit, []);
        byUnit.get(hit.unit).push(hit.k);
    }
    if (!byUnit.size) return null;
    const elements = [...byUnit].map(([unit, ks]) => ({ unit, indices: new Int32Array([...new Set(ks)].sort((p, q) => p - q)) }));
    return { kind: 'element-loci', structure: index.structure, elements };
}

const _hexToInt = (hex) => parseInt(hex.slice(1), 16);

/** Interactions as dashed distance lines (Mol*'s measurement representation). */
async function _drawInteractions(result, opts = {}) {
    const token = ++_drawToken;
    const v = _viewer;
    if (!v || !result || result.error) return;
    const plugin = v.plugin;
    const pose = _poses[_cur];
    const mm = plugin.managers.structure.measurement;
    try {
        // each line is a selection cell with the distance representation under it
        const refs = new Set((mm.state.distances || []).map(d => d.transform.parent || d.transform.ref));
        for (const ref of refs) await plugin.build().delete(ref).commit();
    } catch (err) { console.warn('[BoltzContacts] clearing lines failed:', err); }
    if (token !== _drawToken) return;

    const index = _atomIndex(plugin);
    if (!index) return;
    const ligLoci = _lociFor(index, pose.st, result.ligand.atoms);
    for (const x of result.interactions) {
        const fam = FAMILY_OF[x.type];
        if (_hidden.has(fam) || !x.prot?.length || !x.lig?.length) continue;
        const a = _lociFor(index, pose.st, x.lig);
        const b = _lociFor(index, pose.st, x.prot);
        if (!a || !b) continue;
        const w = x.water?.length ? _lociFor(index, pose.st, x.water) : null;
        const legs = w ? [[a, w], [w, b]] : [[a, b]];
        try {
            for (const [p, q] of legs) await mm.addDistance(p, q, {
                customText: '',
                visualParams: {
                    linesColor: _hexToInt(FAMILY[fam].color),
                    linesSize: fam === 'hydrophobic' || fam === 'vdw' ? 0.05 : 0.08,
                    dashLength: fam === 'hbond' || fam === 'ionic' ? 0.18 : 0.12,
                    textSize: 0.001, textColor: _hexToInt(FAMILY[fam].color),
                },
            });
        } catch (err) {
            console.warn('[BoltzContacts] line failed:', err);
            break;
        }
        if (token !== _drawToken) return;
    }
    // Contacting residues and bridging waters as sticks (the ligand has its own)
    const ligSet = new Set(result.ligand.atoms);
    const shown = new Set();
    for (const x of result.interactions) {
        if (_hidden.has(FAMILY_OF[x.type])) continue;
        for (const i of x.prot || []) for (const j of pose.st.residues[pose.st.atoms[i].res].atoms) if (!ligSet.has(j)) shown.add(j);
        for (const w of x.water || []) shown.add(w);
    }
    const focus = shown.size ? _lociFor(index, pose.st, [...shown]) : null;
    // Mol*'s focus manager moves the camera as a side effect, and not usefully
    const camera = plugin.canvas3d?.camera;
    const snapshot = camera?.getSnapshot();
    try {
        if (focus) plugin.managers.structure.focus.setFromLoci(focus);
        else plugin.managers.structure.focus.clear();
    } catch (err) { console.warn('[BoltzContacts] focus failed:', err); }
    // Once the focus representation has settled, frame the ligand or put the
    // camera back. (Recolouring the focus sticks here empties the canvas in this
    // Mol* bundle, so they keep the chain colour.)
    await new Promise(r => setTimeout(r, 150));
    if (token !== _drawToken) return;
    try {
        if (opts.frame && ligLoci) plugin.managers.camera.focusLoci(ligLoci, { extraRadius: 5 });
        else if (snapshot) camera.setState(snapshot, 0);
    } catch { /* camera API varies */ }
}

function _focusResidue(chain, resnr) {
    const v = _viewer;
    const pose = _poses[_cur];
    if (!v || !pose) return;
    const index = _atomIndex(v.plugin);
    if (!index) return;
    const res = pose.st.residues.find(r => r.chain === chain && r.resnr === resnr);
    if (!res) return;
    const loci = _lociFor(index, pose.st, res.atoms);
    if (!loci) return;
    try {
        v.plugin.managers.interactivity.lociHighlights.highlightOnly({ loci });
        v.plugin.managers.camera.focusLoci(loci, { extraRadius: 6 });
    } catch (err) { console.warn('[BoltzContacts] focus failed:', err); }
}

// ----------------------------------------------------------------
// Export
// ----------------------------------------------------------------

function _profiled() {
    return _poses.filter(p => p.results[resultKey()] && !p.results[resultKey()].error);
}

function _poseId(p) {
    return p.name.replace(/\.(cif|mmcif|pdb|ent)$/i, '');
}

function _download(name, text, type) {
    const a = document.createElement('a');
    a.href = URL.createObjectURL(new Blob([text], { type }));
    a.download = name;
    a.click();
    setTimeout(() => URL.revokeObjectURL(a.href), 1000);
}

function _jsonl() {
    return _profiled().map(p => JSON.stringify(toTriageRecord(_poseId(p), p.results[resultKey()]))).join('\n') + '\n';
}

function _exportJsonl() {
    if (!_profiled().length) return;
    _download(`contacts_${_rules}_${_engine}.jsonl`, _jsonl(), 'application/x-ndjson');
}

function _exportCsv() {
    const rows = _profiled().flatMap(p => toCsvRows(_poseId(p), p.results[resultKey()]));
    if (!rows.length) return;
    const head = Object.keys(rows[0]);
    const q = (v) => (/[",\n]/.test(String(v)) ? `"${String(v).replace(/"/g, '""')}"` : String(v));
    _download(`contacts_${_rules}_${_engine}.csv`, [head.join(','), ...rows.map(r => head.map(h => q(r[h])).join(','))].join('\n') + '\n', 'text/csv');
}

/** Hand the PLIP profiles to BoltzTriage through its own file input. */
async function _sendToTriage() {
    if (!_profiled().length) return;
    const file = new File([_jsonl()], `contacts_${_engine}.jsonl`, { type: 'application/x-ndjson' });
    await _ctx.switchMode('triage');
    const input = $('tri-file');
    if (!input) return;
    const dt = new DataTransfer();
    dt.items.add(file);
    input.files = dt.files;
    input.dispatchEvent(new Event('change', { bubbles: true }));
}
