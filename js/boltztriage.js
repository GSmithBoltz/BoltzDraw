/**
 * boltztriage.js
 *
 * BoltzTriage — PLIP hit triage, as two steps.
 *
 *   1 · The standard     derived automatically; star the interactions that matter
 *   2 · Board            four tiers, plus off-site and not-assessable
 *
 * TWO ARMS, ONE STANDARD
 * ----------------------
 *   Reference-led   something is bound. Its PLIP footprint is the standard.
 *   Observed-led    nothing is bound. The interactions the library makes most
 *                   often are the standard, polar ones at a lower frequency bar.
 *
 * Both produce the same object — a list of (residue, interaction) pairs with a
 * value read off the PLIP call — so everything after step 1 is identical.
 *
 * The only decision the user makes is which interactions are **key**. Polar
 * versus hydrophobic is not a decision: PLIP already said which is which, and
 * asking again would be asking the user to retype the input.
 *
 * Scoring lives in triagescore.js; demo data in triagedemo.js.
 */

import {
    FAMILY, KEY_MULTIPLIER, MAINTAINED_CUT, isPolar,
    indexScreen, survey, standardFromReference, standardFromLibrary, standardValue, pairValue,
    reachCheck, scoreScreen, tierSummary, flagSummary, sizeCheck, standardHeat,
    parsePlipFile, rowsToCsv,
} from './triagescore.js';
import { listDemoScreens, buildDemoScreen } from './triagedemo.js';

// ----------------------------------------------------------------
// State
// ----------------------------------------------------------------

let _RDKit  = null;
let _inited = false;

let _screen  = null;
let _index   = null;
let _survey  = null;
let _std     = null;    // the standard: [{res, fam, key, restype, libFrac, detail}]
let _result  = null;
let _arm     = null;    // 'reference' | 'observed'
let _step    = null;
let _from    = null;    // where the standard came from, in words
let _reach   = null;    // can one molecule satisfy this standard at all?

const _decisions = new Map();
let _sel  = null;
let _view = 'lanes';
const _svgCache = new Map();

const STEPS = [
    { id: 'standard', label: 'The standard', hint: 'What a pose is judged against' },
    { id: 'board',    label: 'Board',        hint: 'Triage' },
];

const TIER_LANES = [
    { key: 1,     label: 'Tier 1', sub: 'conserved + extended' },
    { key: 2,     label: 'Tier 2', sub: 'conserved' },
    { key: 3,     label: 'Tier 3', sub: 'partial + extended' },
    { key: 4,     label: 'Tier 4', sub: 'partial' },
    { key: 'OFF', label: 'Off-site', sub: 'a different pocket' },
    { key: 'NA',  label: 'Not assessable', sub: 'pocket outside the crop' },
];

// ----------------------------------------------------------------
// Tiny DOM helpers
// ----------------------------------------------------------------

const $ = (id) => document.getElementById(id);
const esc = (s) => String(s == null ? '' : s)
    .replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;').replace(/"/g, '&quot;');
const pct = (v, d = 0) => v == null ? '—' : `${(100 * v).toFixed(d)}%`;
const num = (v, d = 2) => v == null ? '—' : (+v).toFixed(d);
const on = (sel, ev, fn, root = document) => root.querySelectorAll(sel).forEach(n => n.addEventListener(ev, fn));

const AA1 = {
    ALA: 'A', ARG: 'R', ASN: 'N', ASP: 'D', CYS: 'C', GLN: 'Q', GLU: 'E', GLY: 'G',
    HIS: 'H', ILE: 'I', LEU: 'L', LYS: 'K', MET: 'M', PHE: 'F', PRO: 'P', SER: 'S',
    THR: 'T', TRP: 'W', TYR: 'Y', VAL: 'V',
};

function resLabel(key) {
    const rt = (_index && _index.restypes[key]) || '';
    const [chain, nr] = key.split(':');
    return `${chain}·${AA1[rt] || 'X'}${nr}`;
}

const famPill = (fam) =>
    `<span class="tri-fam ${isPolar(fam) ? 'polar' : 'phobic'}">${esc(FAMILY[fam].label)}</span>`;

function molSvg(smiles, w = 150, h = 100) {
    if (!smiles || !_RDKit) return '';
    const ck = `${smiles}|${w}x${h}`;
    if (_svgCache.has(ck)) return _svgCache.get(ck);
    let svg = '', mol = null;
    try {
        mol = _RDKit.get_mol(smiles);
        if (mol && mol.is_valid()) svg = mol.get_svg(w, h);
    } catch (e) { /* an unrenderable SMILES is not worth an error */ }
    finally { if (mol) try { mol.delete(); } catch (e) { /* already freed */ } }
    _svgCache.set(ck, svg);
    return svg;
}

function mwOf(smiles) {
    if (!_RDKit) return null;
    let mol = null;
    try {
        mol = _RDKit.get_mol(smiles);
        if (!mol || !mol.is_valid()) return null;
        const d = JSON.parse(mol.get_descriptors() || '{}');
        return d.amw ? +d.amw : null;
    } catch (e) { return null; }
    finally { if (mol) try { mol.delete(); } catch (e) { /* already freed */ } }
}

// ----------------------------------------------------------------
// Init
// ----------------------------------------------------------------

export function initBoltzTriage(RDKit) {
    if (_inited) return;
    _inited = true;
    _RDKit = RDKit;
    _renderChooser();
    $('tri-back').addEventListener('click', () => {
        $('tri-workspace').classList.add('hidden');
        $('tri-chooser').classList.remove('hidden');
    });
}

// ----------------------------------------------------------------
// Stage 1 — the screen, then the arm
// ----------------------------------------------------------------

function _renderChooser() {
    $('tri-screens').innerHTML = listDemoScreens().map(s => `
        <button class="tri-screen-card" data-screen="${esc(s.id)}">
            <div class="tri-screen-top">
                <span class="tri-screen-target">${esc(s.target)}</span>
                <span class="tri-badge tri-badge-synthetic">synthetic</span>
            </div>
            <p class="tri-screen-lede">${esc(s.lede)}</p>
            <div class="tri-screen-foot">
                <span>${s.n} poses</span><span class="tri-dot">·</span>
                <span>${s.hasReference ? 'reference bound' : 'no reference'}</span>
            </div>
        </button>`).join('');

    on('.tri-screen-card', 'click', (e) => {
        document.querySelectorAll('.tri-screen-card').forEach(c => c.classList.remove('selected'));
        e.currentTarget.classList.add('selected');
        _loadDemo(e.currentTarget.dataset.screen);
    });
    $('tri-file').addEventListener('change', (e) => {
        const f = e.target.files && e.target.files[0];
        if (f) _loadFile(f);
    });
    on('.tri-arm-card', 'click', (e) => {
        if (_screen) _startArm(e.currentTarget.dataset.arm);
    });
}

function _loadDemo(id) {
    _screen = buildDemoScreen(id, mwOf);
    _index = indexScreen(_screen);
    _afterLoad(`${_screen.name} · ${_screen.poses.length} synthetic poses`);
}

async function _loadFile(file) {
    try {
        const text = await file.text();
        const { poses } = parsePlipFile(text, file.name);
        if (!poses.length) throw new Error('no poses found in the file');
        _screen = {
            id: 'upload', name: file.name, target: file.name.replace(/\.[^.]+$/, ''),
            synthetic: false, lede: 'Loaded from a PLIP result file.',
            numbering: 'as given in the file',
            suggestedArm: null, reference: null, guards: {},
            roles: {}, notes: {}, restypes: {}, cationicAnchors: [], poses,
        };
        _index = indexScreen(_screen);
        document.querySelectorAll('.tri-screen-card').forEach(c => c.classList.remove('selected'));
        _afterLoad(`${file.name} · ${poses.length} poses, ${_index.residues.length} residues`);
    } catch (err) {
        const st = $('tri-load-status');
        st.className = 'tri-status tri-status-bad';
        st.style.display = 'block';
        st.textContent = `Could not read that file — ${err.message}`;
    }
}

function _afterLoad(msg) {
    _decisions.clear();
    _std = null; _result = null; _sel = null; _reach = null;
    _survey = survey(_index);

    const st = $('tri-load-status');
    st.className = 'tri-status tri-status-ok';
    st.style.display = 'block';
    st.textContent = `Loaded — ${msg}`;

    $('tri-arms').classList.remove('tri-disabled');
    document.querySelector('.tri-arm-card[data-arm="reference"] .tri-arm-state').textContent =
        _screen.reference
            ? `Reference available: ${_screen.reference.label}`
            : 'No reference in this screen — the observed-led arm is the one that applies.';
    document.querySelector('.tri-arm-card[data-arm="observed"] .tri-arm-state').textContent =
        `${_survey.polarRows.length} distinct polar interactions seen across ${_index.N} poses.`;
    document.querySelectorAll('.tri-arm-card').forEach(c =>
        c.classList.toggle('suggested', c.dataset.arm === _screen.suggestedArm));
}

function _startArm(arm) {
    _arm = arm;
    if (arm === 'reference' && _screen.reference && _screen.reference.contacts) {
        _std = standardFromReference(_screen.reference.contacts, _index);
        _from = `${_screen.reference.label} — ${_screen.reference.kind === 'peptide' ? 'a bound peptide'
                 : _screen.reference.kind === 'pose' ? 'a predicted pose' : 'a bound small molecule'}`;
    } else {
        _std = standardFromLibrary(_index);
        _from = `the ${_index.N} poses themselves — the interactions they make most often`;
    }
    _reach = reachCheck(_std, _index);

    $('tri-chooser').classList.add('hidden');
    $('tri-workspace').classList.remove('hidden');
    $('tri-screen-name').textContent = _screen.name;
    $('tri-screen-badge').classList.toggle('hidden', !_screen.synthetic);
    $('tri-arm-name').textContent = arm === 'reference' ? 'Reference-led' : 'Observed-led';
    _go('standard');
}

// ----------------------------------------------------------------
// Step plumbing
// ----------------------------------------------------------------

function _go(step) {
    _step = step;
    const at = STEPS.findIndex(s => s.id === step);
    $('tri-rail').innerHTML = STEPS.map((s, i) => `
        <button class="tri-rail-step${i === at ? ' active' : ''}${i < at ? ' done' : ''}" data-step="${s.id}">
            <span class="tri-rail-n">${i + 1}</span>
            <span class="tri-rail-text"><strong>${esc(s.label)}</strong><em>${esc(s.hint)}</em></span>
        </button>`).join('');
    on('.tri-rail-step', 'click', (e) => _go(e.currentTarget.dataset.step), $('tri-rail'));

    const body = $('tri-body');
    body.scrollTop = 0;
    $('tri-side').classList.toggle('hidden', step !== 'board');
    $('tri-main').classList.toggle('board-mode', step === 'board');

    if (step === 'standard') _renderStandardStep(body);
    if (step === 'board')    _renderBoard(body);
}

function _rescore() {
    const offSiteRes = Object.entries(_screen.roles || {})
        .filter(([, role]) => role === 'decoy').map(([k]) => k);
    _result = scoreScreen(_screen, _index, _std, offSiteRes);
}

// ----------------------------------------------------------------
// Step 1 — the standard
// ----------------------------------------------------------------

function _renderStandardStep(body) {
    _rescore();

    const polar = _std.filter(p => isPolar(p.fam));
    const phobic = _std.filter(p => !isPolar(p.fam));
    const total = standardValue(_std);
    const keyed = _std.filter(p => p.key);
    const refKind = _arm === 'reference' && _screen.reference ? _screen.reference.kind : null;
    const medianRes = _median(_index.poses.map(p => p.resSet.size));

    body.innerHTML = `
        <div class="tri-step-head">
            <h2>The standard</h2>
            <p>From <strong>${esc(_from)}</strong>. Every interaction is valued by what PLIP called it:
               a directional polar contact is worth about three hydrophobic ones, because it is the
               contact a chemist can build on, and hydrophobic surface is what every greasy molecule
               makes everywhere.</p>
        </div>

        <div class="tri-policy">
            ${Object.entries(FAMILY).map(([fam, f]) => `
            <div class="tri-policy-item${_std.some(p => p.fam === fam) ? '' : ' absent'}">
                ${famPill(fam)}
                <span class="tri-policy-val">${f.value.toFixed(2)}</span>
                ${f.cheap ? '<span class="tri-policy-note">cheap to form — flagged on the board</span>' : ''}
            </div>`).join('')}
        </div>

        <div class="tri-stat-strip">
            ${_stat(_std.length, 'interactions in the standard')}
            ${_stat(polar.length, 'polar')}
            ${_stat(phobic.length, 'hydrophobic')}
            ${_stat(num(total, 1), 'total value')}
            ${_stat(keyed.length || '—', 'starred key')}
            ${_reach ? _stat(`${_reach.bestCount}/${_reach.nResidues}`, 'residues the best pose reaches') : ''}
            ${_reach ? _stat(pct(_reach.medianFrac), 'reached by the median pose') : ''}
        </div>

        ${_arm === 'observed' ? `
        <div class="tri-callout">
            <strong>Nothing is bound, so the library is the reference.</strong>
            An interaction joins the standard if it appears in ≥8% of poses when polar and ≥20% when
            hydrophobic — a hydrophobic contact made by a third of the library says little about the
            site, while an H-bond made by a tenth of it is a real feature of the pocket. Read the result
            as a hypothesis to test: poses agreeing on something means the predictor agrees with itself.
        </div>` : ''}

        ${_reach && _reach.bestFrac < 0.5 ? `
        <div class="tri-callout tri-callout-warn">
            <strong>No molecule in this library can satisfy this standard.</strong>
            The best pose of ${_index.N} — ${esc(_reach.bestId)} — reaches
            ${_reach.bestCount} of the ${_reach.nResidues} residues
            (${pct(_reach.bestFrac)}); the median pose reaches ${pct(_reach.medianFrac)}.
            Either these interactions are not all in the same pocket, or the standard came from something
            much larger than the compounds being scored. Star the few that carry the pharmacophore and
            read the tiers as being about those.
        </div>` : ''}

        ${refKind === 'peptide' ? `
        <div class="tri-callout tri-callout-warn">
            <strong>A peptide footprint is not a small-molecule pharmacophore.</strong>
            This reference engages ${new Set(_std.map(p => p.res)).size} residues; the median pose in
            this library touches ${medianRes}. Star the few that carry the pharmacophore — scoring
            against the whole footprint measures molecular size, which is what happened to the first
            SS18-SSX run.
        </div>` : ''}

        <h3 class="tri-h3">Star the interactions that matter
            <span class="tri-faint">— the only choice on this page</span></h3>
        <p class="tri-sub">A starred interaction is worth ${KEY_MULTIPLIER}× its normal value, so missing
           one costs a pose most of its maintained score and the board names it on the card. Star nothing
           and the automatic weighting stands on its own.</p>

        <div class="tri-table-wrap">
            <table class="tri-table tri-std-table">
                <thead><tr>
                    <th>Key</th><th>Residue</th><th>Interaction</th><th>Value</th>
                    <th>Library</th>${_arm === 'reference' ? '<th>Reference geometry</th>' : ''}<th>Note</th>
                </tr></thead>
                <tbody>
                ${_std.map((p, i) => `<tr class="${p.key ? 'starred' : ''}">
                    <td><button class="tri-star${p.key ? ' on' : ''}" data-star="${i}"
                        title="${p.key ? 'Starred as key' : 'Star as key'}">★</button></td>
                    <td class="tri-res">${esc(resLabel(p.res))}</td>
                    <td>${famPill(p.fam)}</td>
                    <td class="tri-mono">${num(pairValue(p))}${p.key ? ` <span class="tri-faint">(${KEY_MULTIPLIER}×)</span>` : ''}</td>
                    <td><span class="tri-freq-bar sm"><i style="width:${(100 * (p.libFrac || 0)).toFixed(1)}%"></i></span>
                        <span class="tri-faint">${pct(p.libFrac || 0)}</span></td>
                    ${_arm === 'reference' ? `<td class="tri-mono tri-faint">${p.detail && p.detail.dist_ad != null
                        ? `D-A ${num(p.detail.dist_ad)} Å · H-A ${num(p.detail.dist_ah)} Å`
                        : p.detail && p.detail.dist != null ? `${num(p.detail.dist)} Å` : '—'}</td>` : ''}
                    <td class="tri-note">${esc((_screen.notes && _screen.notes[p.res]) || '')}</td>
                </tr>`).join('')}
                </tbody>
            </table>
        </div>
        ${_arm === 'reference' ? `<p class="tri-foot-note">Both H-bond distances are shown on purpose.
           PLIP's <code>distance_ah</code> is the hydrogen–acceptor distance, not the donor–acceptor one;
           printing the first and labelling it "D-A" under-reports every H-bond by about 0.85 Å.</p>` : ''}

        <h3 class="tri-h3">How this becomes a tier</h3>
        ${_tierMapHtml(true)}
        <p class="tri-foot-note">The line between the rows is
           <strong>${pct(MAINTAINED_CUT)} of the standard's value</strong> — an absolute bar, so a tier
           means the same thing on the next screen. The columns count <em>polar</em> interactions only:
           total new contacts rise with molecular weight${_survey.sizeConfound != null
             ? ` (ρ = ${num(_survey.sizeConfound)} here between MW and residues contacted)`
             : ''}, so counting them all would fill Tier 1 with the largest molecules in the library.</p>

        ${_navHtml('Open the board', 'Next: triage')}`;

    on('[data-star]', 'click', (e) => {
        const i = +e.currentTarget.dataset.star;
        _std[i].key = !_std[i].key;
        _renderStandardStep(body);
    });
    on('[data-tier-cell]', 'click', () => _go('board'));
    _bindNav();
}

const _stat = (v, l) => `<div class="tri-stat"><strong>${esc(v)}</strong><span>${esc(l)}</span></div>`;
const _median = (a) => { const s = [...a].sort((x, y) => x - y); return s.length ? s[Math.floor(s.length / 2)] : 0; };

function _navHtml(nextLabel, nextHint) {
    return `<div class="tri-nav">
        ${nextHint ? `<span class="tri-nav-hint">${esc(nextHint)}</span>` : ''}
        <button class="tri-primary" id="tri-next">${esc(nextLabel)} →</button>
    </div>`;
}
function _bindNav() {
    const b = $('tri-next');
    if (b) b.addEventListener('click', () => _go('board'));
}

/** The 2×2 the whole scheme reduces to. Doubles as the board's tier filter. */
function _tierMapHtml(explain) {
    const t = tierSummary(_result.rows);
    const cell = (tier, label) => `
        <button class="tri-tier-cell tier-${tier}${_filters.tiers.has(String(tier)) ? ' on' : ''}"
                data-tier-cell="${tier}">
            <span class="tri-tier-cell-n">${t[tier] || 0}</span>
            <span class="tri-tier-cell-t">Tier ${tier}</span>
            <span class="tri-tier-cell-l">${esc(label)}</span>
        </button>`;
    return `<div class="tri-tier-map">
        <div class="tri-map-grid">
            <span class="tri-map-corner"></span>
            <span class="tri-map-col">adds a new polar interaction</span>
            <span class="tri-map-col">adds none</span>
            <span class="tri-map-row">maintains ≥ ${pct(MAINTAINED_CUT)}
                <em>of the standard's value</em></span>
            ${cell(1, 'conserved + extended')}
            ${cell(2, 'conserved')}
            <span class="tri-map-row">maintains &lt; ${pct(MAINTAINED_CUT)}</span>
            ${cell(3, 'partial + extended')}
            ${cell(4, 'partial')}
        </div>
        <div class="tri-map-aside">
            <button class="tri-tier-cell tier-OFF${_filters.tiers.has('OFF') ? ' on' : ''}" data-tier-cell="OFF">
                <span class="tri-tier-cell-n">${t.OFF || 0}</span>
                <span class="tri-tier-cell-t">Off-site</span>
                <span class="tri-tier-cell-l">a different pocket — wrong, not weak</span>
            </button>
            <button class="tri-tier-cell tier-NA${_filters.tiers.has('NA') ? ' on' : ''}" data-tier-cell="NA">
                <span class="tri-tier-cell-n">${t.NA || 0}</span>
                <span class="tri-tier-cell-t">Not assessable</span>
                <span class="tri-tier-cell-l">pocket outside the crop — unscored</span>
            </button>
            ${explain ? `<p class="tri-map-note">These two cannot sit in the grid: an off-site pose
               maintains nothing and adds plenty, which the 2×2 would read as "partial but extended".</p>` : ''}
        </div>
    </div>`;
}

// ----------------------------------------------------------------
// Step 2 — the board
// ----------------------------------------------------------------

const _filters = { tiers: new Set(), flags: new Set(), minProb: 0, search: '', must: new Set(), mustNot: new Set() };

function _visibleRows() {
    if (!_result) return [];
    return _result.rows.filter(r => {
        if (_filters.tiers.size && !_filters.tiers.has(String(r.tier))) return false;
        if (_filters.flags.size && !r.flags.some(f => _filters.flags.has(f.id))) return false;
        if (r.prob != null && r.prob < _filters.minProb) return false;
        if (_filters.search && !r.id.toLowerCase().includes(_filters.search.toLowerCase())) return false;
        const touches = (k) => r.hitExact.some(p => p.res === k) || r.hitPartial.some(p => p.res === k);
        for (const k of _filters.must)    if (!touches(k)) return false;
        for (const k of _filters.mustNot) if (touches(k))  return false;
        return true;
    });
}

function _renderBoard(body) {
    _rescore();
    const rows = _visibleRows();
    const t = tierSummary(_result.rows);
    const flags = flagSummary(_result.rows);
    const sc = sizeCheck(_result.rows);

    body.innerHTML = `
        ${_tierMapHtml(false)}

        <div class="tri-board-bar">
            <div class="tri-seg-row sm">
                ${[['lanes', 'Lanes'], ['table', 'Table'], ['heat', 'Interaction map']].map(([v, l]) =>
                    `<button class="tri-seg${_view === v ? ' active' : ''}" data-view="${v}">${l}</button>`).join('')}
            </div>
            <input type="search" class="tri-input sm" id="tri-search" placeholder="Find a compound ID"
                   value="${esc(_filters.search)}">
            <div class="tri-filter-group">
                ${flags.map(f => `<button class="tri-flag level-${f.level}${_filters.flags.has(f.id) ? ' on' : ''}"
                    data-flagfilter="${esc(f.id)}">${esc(f.label)} <em>${f.n}</em></button>`).join('')}
            </div>
            <label class="tri-mini-ctrl">min p
                <input type="range" id="tri-minprob" min="0" max="0.9" step="0.05" value="${_filters.minProb}">
                <strong>${num(_filters.minProb)}</strong></label>
            <div class="tri-board-right">
                <span class="tri-faint">${rows.length} of ${_result.rows.length}</span>
                <span class="tri-decided">${_decisions.size} decided</span>
                <button class="tri-secondary sm" id="tri-export">Export CSV</button>
            </div>
        </div>

        ${sc && sc.topMedianMw > sc.libraryMedianMw * 1.15 ? `
        <div class="tri-callout tri-callout-warn tri-slim">
            <strong>Size check:</strong> the top ${sc.k} by maintained value have a median MW of
            ${Math.round(sc.topMedianMw)} against ${Math.round(sc.libraryMedianMw)} for the library
            (ρ = ${num(sc.rhoMaintained)}). Counting only new <em>polar</em> interactions keeps the
            left-right split off that axis — ρ with new polar is ${num(sc.rhoNewPolar)}, and with all
            new contacts it would be ${num(sc.rhoNewAll)}.
        </div>` : ''}

        ${_filters.must.size || _filters.mustNot.size ? `
        <div class="tri-active-filters">
            ${[..._filters.must].map(k => `<span class="tri-chip on">must touch ${esc(resLabel(k))}
                <button data-unmust="${esc(k)}">×</button></span>`).join('')}
            ${[..._filters.mustNot].map(k => `<span class="tri-chip off">must not touch ${esc(resLabel(k))}
                <button data-unmustnot="${esc(k)}">×</button></span>`).join('')}
        </div>` : ''}

        <div id="tri-board-body">${_boardBodyHtml(rows, t)}</div>`;

    on('[data-view]', 'click', (e) => { _view = e.currentTarget.dataset.view; _renderBoard(body); });
    on('[data-tier-cell]', 'click', (e) => {
        const k = e.currentTarget.dataset.tierCell;
        if (_filters.tiers.has(k)) _filters.tiers.delete(k); else _filters.tiers.add(k);
        _renderBoard(body);
    });
    on('[data-flagfilter]', 'click', (e) => {
        const f = e.currentTarget.dataset.flagfilter;
        if (_filters.flags.has(f)) _filters.flags.delete(f); else _filters.flags.add(f);
        _renderBoard(body);
    });
    $('tri-search').addEventListener('input', (e) => {
        _filters.search = e.target.value;
        $('tri-board-body').innerHTML = _boardBodyHtml(_visibleRows(), t);
        _bindCards(body);
    });
    $('tri-minprob').addEventListener('change', (e) => { _filters.minProb = +e.target.value; _renderBoard(body); });
    $('tri-export').addEventListener('click', _exportCsv);
    on('[data-unmust]', 'click', (e) => { _filters.must.delete(e.currentTarget.dataset.unmust); _renderBoard(body); });
    on('[data-unmustnot]', 'click', (e) => { _filters.mustNot.delete(e.currentTarget.dataset.unmustnot); _renderBoard(body); });

    _bindCards(body);
    if (!_sel && rows.length) _sel = rows[0].id;
    _renderInspector();
}

const _boardBodyHtml = (rows, t) =>
    _view === 'lanes' ? _lanesHtml(rows, t) : _view === 'table' ? _tableHtml(rows) : _heatHtml(rows);

function _bindCards(body) {
    on('[data-pose-card]', 'click', (e) => {
        _sel = e.currentTarget.dataset.poseCard;
        document.querySelectorAll('[data-pose-card]').forEach(c =>
            c.classList.toggle('selected', c.dataset.poseCard === _sel));
        _renderInspector();
    }, body);
    on('[data-res-filter]', 'click', (e) => {
        e.stopPropagation();
        const k = e.currentTarget.dataset.resFilter;
        if (_filters.must.has(k)) { _filters.must.delete(k); _filters.mustNot.add(k); }
        else if (_filters.mustNot.has(k)) _filters.mustNot.delete(k);
        else _filters.must.add(k);
        _renderBoard($('tri-body'));
    }, body);
}

function _lanesHtml(rows, tierTotals) {
    const byTier = new Map(TIER_LANES.map(l => [String(l.key), []]));
    for (const r of rows) {
        const k = String(r.tier);
        if (byTier.has(k)) byTier.get(k).push(r);
    }
    return `<div class="tri-lanes">
        ${TIER_LANES.map(l => {
            const list = byTier.get(String(l.key)) || [];
            return `<div class="tri-lane lane-${l.key}">
                <div class="tri-lane-head">
                    <div class="tri-lane-top">
                        <span class="tri-tier-pill tier-${l.key}">${esc(l.label)}</span>
                        <span class="tri-lane-n">${list.length}${list.length !== (tierTotals[l.key] || 0) ? ` / ${tierTotals[l.key] || 0}` : ''}</span>
                    </div>
                    <span class="tri-lane-sub">${esc(l.sub)}</span>
                </div>
                <div class="tri-lane-body">
                    ${list.slice(0, 40).map(_cardHtml).join('')}
                    ${list.length > 40 ? `<div class="tri-lane-more">+${list.length - 40} more — narrow the filters</div>` : ''}
                    ${!list.length ? '<div class="tri-lane-empty">—</div>' : ''}
                </div>
            </div>`;
        }).join('')}
    </div>`;
}

function _cardHtml(r) {
    const d = _decisions.get(r.id);
    return `<div class="tri-card${r.id === _sel ? ' selected' : ''}${d ? ` decided-${d.verdict}` : ''}" data-pose-card="${esc(r.id)}">
        <div class="tri-card-top">
            <span class="tri-card-id">${esc(r.id)}</span>
            ${d ? `<span class="tri-verdict v-${d.verdict}">${esc(d.verdict)}</span>` : ''}
        </div>
        <div class="tri-card-mol">${molSvg(r.smiles, 132, 76) || '<span class="tri-faint">no structure</span>'}</div>
        <div class="tri-card-cov">
            <span class="tri-cov-bar"><i style="width:${Math.min(100, 100 * r.maintainedFrac).toFixed(0)}%"></i></span>
            <span class="tri-cov-text">${pct(r.maintainedFrac)}</span>
        </div>
        <div class="tri-card-meta">
            <span title="New polar interactions outside the standard">+${r.nNewPolar} polar</span>
            <span title="Boltz binding probability">p ${num(r.prob)}</span>
            <span title="Molecular weight">${r.mw ? Math.round(r.mw) : '—'} Da</span>
        </div>
        ${r.flags.length ? `<div class="tri-card-flags">
            ${r.flags.map(f => `<span class="tri-pip level-${f.level}" title="${esc(f.text)}">${esc(f.label)}</span>`).join('')}
        </div>` : ''}
    </div>`;
}

function _tableHtml(rows) {
    return `<div class="tri-table-wrap scroll">
        <table class="tri-table dense">
            <thead><tr>
                <th>Rank</th><th>ID</th><th>Tier</th><th>Maintained</th><th>Exact</th><th>Partial</th>
                <th>Missed</th><th>New polar</th><th>New phobic</th><th>MW</th><th>p</th><th>Flags</th><th>Decision</th>
            </tr></thead>
            <tbody>
            ${rows.slice(0, 400).map(r => {
                const d = _decisions.get(r.id);
                return `<tr data-pose-card="${esc(r.id)}" class="${r.id === _sel ? 'selected' : ''}">
                    <td class="tri-mono">${r.rankLo}${r.rankHi > r.rankLo ? `–${r.rankHi}` : ''}</td>
                    <td class="tri-mono">${esc(r.id)}</td>
                    <td><span class="tri-tier-pill sm tier-${r.tier}">${esc(String(r.tier))}</span></td>
                    <td class="tri-mono">${pct(r.maintainedFrac)}</td>
                    <td>${r.nExact}/${r.nStd}</td>
                    <td>${r.nPartial}</td>
                    <td>${r.missed.length}${r.nKeyMissed ? ` <span class="tri-faint">(${r.nKeyMissed}★)</span>` : ''}</td>
                    <td class="tri-mono">${r.nNewPolar}</td>
                    <td class="tri-mono">${r.nNewPhobic}</td>
                    <td class="tri-mono">${r.mw ? Math.round(r.mw) : '—'}</td>
                    <td class="tri-mono">${num(r.prob)}</td>
                    <td>${r.flags.map(f => `<span class="tri-pip level-${f.level}" title="${esc(f.text)}">${esc(f.label)}</span>`).join('')}</td>
                    <td>${d ? `<span class="tri-verdict v-${d.verdict}">${esc(d.verdict)}</span>` : ''}</td>
                </tr>`;
            }).join('')}
            </tbody>
        </table>
        ${rows.length > 400 ? `<p class="tri-foot-note">Showing the first 400 of ${rows.length}.</p>` : ''}
    </div>`;
}

function _heatHtml(rows) {
    const heat = standardHeat(rows, _std, 70);
    return `<div class="tri-heat-wrap">
        <p class="tri-sub">One column per interaction in the standard. Click a column to require that
           residue, click again to exclude it. Grey means the residue is not in that pose's crop — an
           absence of evidence, not evidence of absence.</p>
        <div class="tri-heat-scroll">
        <table class="tri-heat">
            <thead><tr><th class="tri-heat-corner">Pose</th>
                ${heat.pairs.map(p => `<th class="hrole-${isPolar(p.fam) ? 'polar' : 'phobic'}${p.key ? ' starred' : ''}">
                    <button data-res-filter="${esc(p.res)}"
                        class="${_filters.must.has(p.res) ? 'must' : _filters.mustNot.has(p.res) ? 'mustnot' : ''}"
                        title="${esc(resLabel(p.res))} ${esc(FAMILY[p.fam].label)}${p.key ? ' (key)' : ''} — ${pct(p.libFrac || 0)} of the library">
                        ${p.key ? '★ ' : ''}${esc(resLabel(p.res))} ${esc(FAMILY[p.fam].short)}</button></th>`).join('')}
                <th class="tri-heat-corner">Tier</th></tr></thead>
            <tbody>
            ${heat.rows.map(hr => `<tr data-pose-card="${esc(hr.id)}" class="${hr.id === _sel ? 'selected' : ''}">
                <td class="tri-mono">${esc(hr.id)}</td>
                ${hr.cells.map(c => `<td class="hc hc${c}"></td>`).join('')}
                <td><span class="tri-tier-pill sm tier-${hr.tier}">${esc(String(hr.tier))}</span></td>
            </tr>`).join('')}
            </tbody>
        </table></div>
        <div class="tri-heat-legend">
            <span><i class="hc hc2"></i> made, same chemistry</span>
            <span><i class="hc hc1"></i> right residue, different chemistry</span>
            <span><i class="hc hc0"></i> in the crop, missed</span>
            <span><i class="hc hc-1"></i> not in the crop</span>
        </div>
    </div>`;
}

// ----------------------------------------------------------------
// Inspector
// ----------------------------------------------------------------

function _renderInspector() {
    const side = $('tri-side-body');
    if (!_result) { side.innerHTML = ''; return; }
    const r = _result.rows.find(x => x.id === _sel);
    if (!r) { side.innerHTML = '<p class="tri-faint">Pick a compound.</p>'; return; }
    const d = _decisions.get(r.id) || {};

    const line = (p, kind, instead) => `<div class="tri-ix tri-ix-${kind}">
        <span class="tri-res">${p.key ? '★ ' : ''}${esc(resLabel(p.res))}</span>
        ${famPill(p.fam)}
        ${instead ? `<span class="tri-faint">made ${esc(FAMILY[instead].label)} instead</span>` : ''}
        <span class="tri-ix-val">${kind === 'missed' ? '—'
            : kind === 'partial' ? `+${num(pairValue(p) * 0.4)}` : `+${num(pairValue(p))}`}</span>
    </div>`;

    side.innerHTML = `
        <div class="tri-insp-head">
            <div>
                <span class="tri-tier-pill tier-${r.tier}">${esc(String(r.tier))}</span>
                <h3>${esc(r.id)}</h3>
            </div>
            <span class="tri-faint tri-mono">rank ${r.rankLo}${r.rankHi > r.rankLo ? `–${r.rankHi}` : ''} of ${_result.rows.length}</span>
        </div>
        <div class="tri-insp-mol">${molSvg(r.smiles, 260, 150) || '<span class="tri-faint">no structure</span>'}</div>
        <p class="tri-why"><strong>Why this tier:</strong> ${esc(r.why)}.
            ${r.rankHi > r.rankLo ? `Its maintained value is shared with ${r.rankHi - r.rankLo} other pose(s), so the rank is a range.` : ''}</p>

        <div class="tri-insp-stats">
            ${_stat(pct(r.maintainedFrac), 'maintained')}
            ${_stat(`${num(r.maintained, 1)}/${num(r.standardTotal, 1)}`, 'value')}
            ${_stat(r.nNewPolar, 'new polar')}
            ${_stat(r.nNewPhobic, 'new phobic')}
            ${_stat(r.mw ? Math.round(r.mw) : '—', 'MW')}
            ${_stat(num(r.prob), 'Boltz p')}
        </div>

        ${r.flags.length ? `<div class="tri-insp-flags">
            ${r.flags.map(f => `<div class="tri-insp-flag level-${f.level}">
                <strong>${esc(f.label)}</strong><span>${esc(f.text)}</span></div>`).join('')}
        </div>` : ''}

        <h4 class="tri-h4">Decision</h4>
        <div class="tri-decide">
            ${['pick', 'hold', 'reject'].map(v =>
                `<button class="tri-verdict-btn v-${v}${d.verdict === v ? ' on' : ''}" data-verdict="${v}">${v}</button>`).join('')}
        </div>
        <input type="text" class="tri-input sm" id="tri-note" placeholder="Reason — the thing you will want in three months"
               value="${esc(d.note || '')}">

        <h4 class="tri-h4">Maintained <span class="tri-faint">${r.nExact} of ${r.nStd}</span></h4>
        ${r.hitExact.map(p => line(p, 'exact')).join('') || '<p class="tri-faint">None.</p>'}

        ${r.hitPartial.length ? `<h4 class="tri-h4">Right residue, different chemistry</h4>
            ${r.hitPartial.map(p => line(p, 'partial', p.insteadOf)).join('')}` : ''}

        ${r.missed.length ? `<h4 class="tri-h4">Missed</h4>
            ${r.missed.map(p => line(p, 'missed')).join('')}` : ''}

        ${r.newPolar.length ? `<h4 class="tri-h4">New polar interactions</h4>
            ${r.newPolar.map(n => `<div class="tri-ix tri-ix-new">
                <span class="tri-res">${esc(resLabel(n.res))}</span>
                ${n.fams.map(famPill).join(' ')}
            </div>`).join('')}` : ''}

        ${r.newPhobic.length ? `<p class="tri-faint tri-newphobic">${r.newPhobic.length} new hydrophobic
            contact(s), not counted: ${r.newPhobic.slice(0, 8).map(n => esc(resLabel(n.res))).join(', ')}${r.newPhobic.length > 8 ? '…' : ''}</p>` : ''}

        ${r.offHit.length ? `<p class="tri-missed"><strong>Off-site contacts:</strong>
            ${r.offHit.map(k => esc(resLabel(k))).join(', ')}</p>` : ''}`;

    on('[data-verdict]', 'click', (e) => {
        const v = e.currentTarget.dataset.verdict;
        const cur = _decisions.get(r.id);
        if (cur && cur.verdict === v) _decisions.delete(r.id);
        else _decisions.set(r.id, { verdict: v, note: (cur && cur.note) || $('tri-note').value || '' });
        _renderBoard($('tri-body'));
    }, side);
    $('tri-note').addEventListener('change', (e) => {
        const cur = _decisions.get(r.id) || { verdict: 'hold' };
        _decisions.set(r.id, { ...cur, note: e.target.value });
    });
}

// ----------------------------------------------------------------
// Export
// ----------------------------------------------------------------

function _exportCsv() {
    const header = [
        `# BoltzTriage export — ${new Date().toISOString()}`,
        `# screen: ${_screen.name}${_screen.synthetic ? ' (SYNTHETIC DEMO DATA — not a result)' : ''}`,
        `# numbering: ${_screen.numbering}`,
        `# arm: ${_arm === 'reference' ? 'reference-led' : 'observed-led'}`,
        `# standard: ${_from}`,
        `# standard interactions: ${_std.map(p => `${p.res}:${FAMILY[p.fam].short}${p.key ? '*' : ''}`).join(' ')}`,
        `# values: ${Object.values(FAMILY).map(f => `${f.short}=${f.value}`).join(' ')}; key = ${KEY_MULTIPLIER}x`,
        `# tiers: T1 >=${pct(MAINTAINED_CUT)} maintained + new polar | T2 >=${pct(MAINTAINED_CUT)}, no new polar`
            + ` | T3 <${pct(MAINTAINED_CUT)} + new polar | T4 <${pct(MAINTAINED_CUT)}, no new polar`,
        '# off-site and not-assessable poses sit outside the tiers, not at the bottom of them.',
        '# A tier is a mechanism annotation, not a potency ranking. Read it alongside the assay.',
    ].join('\n');

    const blob = new Blob([header + '\n' + rowsToCsv(_result.rows, _decisions)], { type: 'text/csv' });
    const a = document.createElement('a');
    a.href = URL.createObjectURL(blob);
    a.download = `triage_${_screen.id}_${_arm}.csv`;
    a.click();
    URL.revokeObjectURL(a.href);
}
