/**
 * boltztable.js
 * BoltzTable — a DataWarrior-style structure table.
 *
 * Loads a CSV/TSV or an SD file (either may be gzipped), renders every row with
 * its 2D structure drawn from SMILES, and lets you narrow the set with live
 * filters that compose: numeric ranges, category picks, text search,
 * substructure (SMARTS) and Tanimoto similarity. An SD file's mol blocks are
 * converted to SMILES on load; its data items become ordinary columns.
 *
 * The table is virtualized — only the rows in view are in the DOM — so a
 * 50k-row file scrolls at the same speed as a 500-row one.
 */

import { parseSdf } from './fileparse.js';

// ----------------------------------------------------------------
// State
// ----------------------------------------------------------------

let _RDKit = null;
let _hooks = {};                 // { onSendTo(mode, csvText, filename) }

let rows       = [];             // raw parsed row objects
let colNames   = [];             // every column in file order
let colMeta    = new Map();      // name -> { type, min, max, bins, uniques, order }
let hiddenCols = new Set();
let smilesCol  = '';
let idCol      = '';

let filtered   = [];             // row indices surviving the filters
let order      = [];             // filtered, after sort — this is what the table shows
let selected   = new Set();      // row indices ticked in the checkbox column
let activeRow  = null;           // row index shown in the Record tab

let sortCol = null;
let sortDir = 1;                 // 1 asc, -1 desc
let rowHeight = 72;
let panelTab = 'filters';
let lastClickedRow = null;       // anchor for shift-click range selection

let filters = new Map();         // colName -> filter state

// Columns drawn as structures: colName -> the column holding the SMILES.
// A column mapped to itself (the primary structure column) owns its SMILES and
// exports normally; one mapped to a different source is draw-only, carrying no
// data of its own, and is skipped by filters and by export.
const renderCols = new Map();

let originalColOrder = [];       // file order, so a mangled layout can be reset

// When the table came from an SD file: each row's mol block as read, the
// column its SMILES went to, and the columns made up on load (title, SMILES).
// SDF export writes those mol blocks back untouched — 3D coordinates and all —
// and leaves the made-up columns out, so a filtered file round-trips cleanly.
let molblocks    = [];
let sdfSmilesCol = '';
let sdfMadeCols  = new Set();
let structFilter = {
    smarts: '', qmol: null, qmolErr: '',
    simSmiles: '', simFp: null, simCutoff: 0.7, simErr: '',
};

// Caches. svgCache is keyed on smiles+size+highlight so a substructure query
// invalidates only the drawings, not the parse.
const svgCache       = new Map();
const substructCache = new Map();   // rowIdx -> {atoms,bonds} | null
let   fpsReady       = false;

let scriptsLoaded = false;
let searchTerm    = '';
let _searchTimer  = null;
let _renderQueued = false;

const CDN_SCRIPTS = [
    'https://cdn.jsdelivr.net/npm/papaparse@5.4.1/papaparse.min.js',
];

// A column with more distinct values than this is free text, not a category.
const MAX_CATEGORIES = 60;
// Rows rendered above and below the viewport, to hide scroll latency.
const OVERSCAN   = 6;
const HIST_BINS  = 24;
const CHECK_W    = 34;           // checkbox gutter width, px
// SD records converted between yields to the browser while loading.
const SDF_CHUNK  = 500;
const MOLBLOCK_OPTS = JSON.stringify({ removeHs: true });

// ----------------------------------------------------------------
// Init
// ----------------------------------------------------------------

export async function initBoltzTable(rdkitInstance, hooks = {}) {
    _RDKit = rdkitInstance;
    _hooks = hooks;

    if (!scriptsLoaded) {
        _setStatus('Loading libraries…', 'loading');
        await Promise.all(CDN_SCRIPTS.map(_loadScript));
        scriptsLoaded = true;
    }

    _wireListeners();
    _renderEmptyState();
    _setStatus('', '');
}

function _loadScript(src) {
    return new Promise((resolve, reject) => {
        if (document.querySelector(`script[src="${src}"]`)) { resolve(); return; }
        const s = document.createElement('script');
        s.src = src;
        s.onload = resolve;
        s.onerror = () => reject(new Error(`Failed to load ${src}`));
        document.head.appendChild(s);
    });
}

function _wireListeners() {
    document.getElementById('bt-file-input')
        .addEventListener('change', e => { if (e.target.files[0]) _handleFile(e.target.files[0]); });

    document.getElementById('bt-smiles-col').addEventListener('change', e => {
        _setStructureColumn(e.target.value);
        svgCache.clear(); fpCaches.clear(); substructCache.clear();
        fpsReady = false;
        _buildStructureFilters();
        _rebuildHeader();
        _applyFilters();
    });
    document.getElementById('bt-id-col').addEventListener('change', e => {
        idCol = e.target.value;
        _renderRecord();
    });

    // Panel tabs
    document.querySelectorAll('.bt-panel-tab').forEach(btn => {
        btn.addEventListener('click', () => _setPanelTab(btn.dataset.tab));
    });

    document.getElementById('bt-reset-filters').addEventListener('click', _resetAllFilters);
    document.getElementById('bt-reset-from-empty').addEventListener('click', _resetAllFilters);
    document.getElementById('bt-show-all-cols').addEventListener('click', () => {
        hiddenCols.clear();
        _buildColumnList();
        _rebuildHeader();
        _renderRows();
    });

    // Toolbar
    document.getElementById('bt-search').addEventListener('input', e => {
        clearTimeout(_searchTimer);
        const v = e.target.value;
        _searchTimer = setTimeout(() => { searchTerm = v.trim().toLowerCase(); _applyFilters(); }, 180);
    });

    document.getElementById('bt-rowheight').addEventListener('click', e => {
        const btn = e.target.closest('.bt-seg-btn');
        if (!btn) return;
        document.querySelectorAll('#bt-rowheight .bt-seg-btn')
            .forEach(b => b.classList.toggle('active', b === btn));
        rowHeight = parseInt(btn.dataset.h, 10);
        _rebuildHeader();
        _renderRows();
    });

    document.getElementById('bt-clear-sel-btn').addEventListener('click', () => {
        selected.clear();
        _updateCounts();
        _renderRows();
    });

    // Export menu
    const expBtn  = document.getElementById('bt-export-btn');
    const expMenu = document.getElementById('bt-export-menu');
    expBtn.addEventListener('click', e => {
        e.stopPropagation();
        expMenu.classList.toggle('hidden');
    });
    expMenu.addEventListener('click', e => {
        const btn = e.target.closest('button[data-act]');
        if (!btn) return;
        expMenu.classList.add('hidden');
        _runExport(btn.dataset.act);
    });
    document.addEventListener('click', () => expMenu.classList.add('hidden'));

    // Table interaction — delegated, because rows are recycled on every scroll
    const scroll = document.getElementById('bt-scroll');
    scroll.addEventListener('scroll', _queueRender, { passive: true });
    new ResizeObserver(_queueRender).observe(scroll);

    document.getElementById('bt-rows').addEventListener('click', _onRowClick);
    const head = document.getElementById('bt-header-row');
    head.addEventListener('click', _onHeaderClick);

    // Both containers outlive their contents, so these bind exactly once —
    // wiring them inside the rebuilds would stack a handler per redraw.
    _wireColumnDrag(head, 'x', _afterReorder);
    _wireColumnDrag(document.getElementById('bt-column-list'), 'y', _afterReorder);

    document.getElementById('bt-reset-order').addEventListener('click', () => {
        colNames = originalColOrder.filter(c => colNames.includes(c))
            .concat(colNames.filter(c => !originalColOrder.includes(c)));
        _afterReorder();
    });

    _wireAnalysis();
}

function _renderEmptyState() {
    document.getElementById('bt-empty-state').innerHTML = `
        <svg width="64" height="64" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1">
            <rect x="2.5" y="4" width="19" height="16" rx="1.5"/>
            <line x1="2.5" y1="9" x2="21.5" y2="9"/>
            <line x1="9"  y1="9" x2="9"  y2="20"/>
            <line x1="15" y1="9" x2="15" y2="20"/>
            <line x1="2.5" y1="14.5" x2="21.5" y2="14.5"/>
        </svg>
        <p>Upload a CSV, TSV or SDF to browse it as a structure table.<br>
           Structures are drawn from a SMILES column or the SD mol blocks;<br>
           every other column becomes a live filter.</p>
    `;
}

// ----------------------------------------------------------------
// File load + column typing
// ----------------------------------------------------------------

async function _handleFile(file) {
    _setStatus('Parsing…', 'loading');
    _resetAll();

    let table;
    try {
        table = await _readTable(file, msg => _setStatus(msg, 'loading'));
    } catch (err) {
        _setStatus('Parse error: ' + err.message, 'error');
        return;
    }

    rows         = table.rows;
    colNames     = table.fields;
    molblocks    = table.molblocks || [];
    sdfSmilesCol = table.smilesCol || '';
    sdfMadeCols  = new Set(table.madeCols || []);

    if (!rows.length || !colNames.length) {
        _setStatus('No data rows found in that file', 'error');
        return;
    }

    originalColOrder = colNames.slice();
    _typeColumns();
    _populateSelects();
    _autoDetect();
    // An SD file's structures live in the column we wrote them to, whatever a
    // data item of its own happens to be called.
    if (table.smilesCol) {
        _setStructureColumn(table.smilesCol);
        document.getElementById('bt-smiles-col').value = table.smilesCol;
    }
    _buildFilters();
    _buildStructureFilters();
    _buildColumnList();
    _rebuildHeader();
    _applyFilters();

    document.getElementById('bt-structure-config').classList.remove('hidden');
    document.getElementById('bt-panel-tabs').classList.remove('hidden');
    document.getElementById('bt-toolbar').classList.remove('hidden');
    document.getElementById('bt-table-wrap').classList.remove('hidden');
    document.getElementById('bt-empty-state').classList.add('hidden');
    _setPanelTab('filters');

    _setStatus(`${rows.length.toLocaleString()} rows · ${colNames.length} columns${_failedNote(table)}`,
               'success');
}

/**
 * Read a CSV/TSV or SD file, gzipped or not, into { rows, fields }. SD files
 * also report the column their structures were written to and how many mol
 * blocks RDKit could not read.
 */
async function _readTable(file, onProgress) {
    const gz   = /\.gz$/i.test(file.name);
    const name = file.name.replace(/\.gz$/i, '');
    const src  = gz ? await _gunzip(file) : file;

    // Trust the contents over the name: SD files turn up as .txt, .mol, or
    // with a second extension a file browser hides.
    const head = typeof src === 'string' ? src.slice(0, 4096) : await src.slice(0, 4096).text();
    if (/\.(sdf|sd|mol)$/i.test(name) || _looksLikeSdf(head)) {
        return _sdfToTable(typeof src === 'string' ? src : await src.text(), onProgress);
    }

    return new Promise((resolve, reject) => {
        Papa.parse(src, {
            header: true,
            skipEmptyLines: true,
            delimiter: /\.(tsv|txt)$/i.test(name) ? '\t' : '',
            complete: res => resolve({ rows: res.data, fields: (res.meta.fields || []).filter(Boolean) }),
            error: reject,
        });
    });
}

function _gunzip(file) {
    return new Response(file.stream().pipeThrough(new DecompressionStream('gzip'))).text();
}

/** A mol block's fourth line is its counts line, ending in V2000 or V3000. */
function _looksLikeSdf(head) {
    const counts = head.split(/\r?\n/, 4)[3];
    return counts !== undefined && /V[23]000\s*$/.test(counts);
}

/**
 * Lay an SD file out as table rows: the title line, the SMILES read from the
 * mol block, then every data item in file order. A record RDKit can't read
 * keeps its row and data with a blank SMILES, rather than vanishing.
 */
async function _sdfToTable(text, onProgress) {
    const { records, fields } = parseSdf(text);

    // Our two columns lead, and step aside from any data item already using
    // the name — compared case-blind, so "Name" and "NAME" don't sit together.
    const taken = new Set(fields.map(f => f.toLowerCase()));
    const free = base => {
        let n = base, k = 2;
        while (taken.has(n.toLowerCase())) n = `${base} (${k++})`;
        taken.add(n.toLowerCase());
        return n;
    };
    const titleCol  = records.some(r => r.title) ? free('Name') : '';
    const smilesCol = free('SMILES');

    const out = [];
    let failed = 0;
    for (let i = 0; i < records.length; i++) {
        const r = records[i];
        const row = {};
        if (titleCol) row[titleCol] = r.title;
        row[smilesCol] = _molblockToSmiles(r.molblock);
        if (!row[smilesCol]) failed++;
        for (const f of fields) row[f] = r.data[f] ?? '';
        out.push(row);

        if (i % SDF_CHUNK === SDF_CHUNK - 1) {
            onProgress(`Reading structures ${(i + 1).toLocaleString()} / ${records.length.toLocaleString()}…`);
            await _yield();
        }
    }

    return {
        rows: out,
        fields: [titleCol, smilesCol, ...fields].filter(Boolean),
        smilesCol, failed,
        molblocks: records.map(r => r.molblock),
        madeCols:  [titleCol, smilesCol].filter(Boolean),
    };
}

function _molblockToSmiles(molblock) {
    let mol;
    try {
        // Explicit hydrogens off, so a 3D pose reads as the same SMILES its
        // 2D drawing would. Stereo is still perceived from the coordinates.
        mol = _RDKit.get_mol(molblock, MOLBLOCK_OPTS);
        return mol && mol.is_valid() ? mol.get_smiles() : '';
    } catch { return ''; }
    finally { if (mol) mol.delete(); }
}

function _failedNote(table) {
    if (!table.failed) return '';
    return ` · ${table.failed.toLocaleString()} structure${table.failed === 1 ? '' : 's'} unreadable (blank SMILES)`;
}

function _resetAll() {
    rows = []; colNames = []; colMeta.clear();
    hiddenCols.clear(); filters.clear(); renderCols.clear();
    filtered = []; order = []; selected.clear();
    activeRow = null; lastClickedRow = null;
    sortCol = null; sortDir = 1;
    smilesCol = ''; idCol = ''; searchTerm = '';
    molblocks = []; sdfSmilesCol = ''; sdfMadeCols = new Set();
    svgCache.clear(); fpCaches.clear(); substructCache.clear();
    fpsReady = false;
    structFilter = { smarts: '', qmol: null, qmolErr: '',
                     simSmiles: '', simFp: null, simCutoff: 0.7, simErr: '' };

    // The reference file survives a new main file — matching it against a
    // fresh table is the whole point — but its results do not.
    nearestMatch.clear();
    nnHits = [];
    refFps = null;
    importedRefCols = [];
    document.getElementById('bt-nn-results').innerHTML = '';
    document.getElementById('bt-nn-select').classList.add('hidden');
    document.getElementById('bt-ref-summary').innerHTML = '';
    _setAnStatus('bt-nn-status', '', '');

    document.getElementById('bt-search').value = '';
    document.getElementById('bt-table-wrap').classList.add('hidden');
    document.getElementById('bt-toolbar').classList.add('hidden');
    document.getElementById('bt-panel-tabs').classList.add('hidden');
    document.getElementById('bt-structure-config').classList.add('hidden');
    document.getElementById('bt-empty-state').classList.remove('hidden');
    document.querySelectorAll('.bt-tab-body').forEach(el => el.classList.add('hidden'));
}

/**
 * Decide what each column is. Numbers get range filters, low-cardinality
 * strings get checkbox lists, everything else gets a substring box.
 */
function _typeColumns() {
    for (const name of colNames) _typeOneColumn(name);
}

function _typeOneColumn(name) {
    const vals = [];
    let numeric = 0, nonEmpty = 0;
    const uniques = new Map();

    for (const row of rows) {
        const raw = row[name];
        if (raw === undefined || raw === null || raw === '') continue;
        nonEmpty++;
        const s = String(raw);
        if (_isNum(s)) { numeric++; vals.push(parseFloat(s)); }
        if (uniques.size <= MAX_CATEGORIES) {
            uniques.set(s, (uniques.get(s) || 0) + 1);
        }
    }

    // A column is numeric if essentially all its populated cells parse.
    if (nonEmpty > 0 && numeric / nonEmpty >= 0.95) {
        // Looped rather than Math.min(...vals): spreading a 100k-element
        // array overflows the argument limit and throws.
        let min = Infinity, max = -Infinity, sum = 0;
        for (const v of vals) {
            if (v < min) min = v;
            if (v > max) max = v;
            sum += v;
        }
        colMeta.set(name, {
            type: 'number', min, max, n: vals.length,
            mean: sum / vals.length,
            median: _median(vals),
            bins: _histogram(vals, min, max),
        });
    } else if (uniques.size > 0 && uniques.size <= MAX_CATEGORIES) {
        const cats = [...uniques.entries()].sort((a, b) => b[1] - a[1]);
        colMeta.set(name, { type: 'category', uniques: new Map(cats), n: nonEmpty });
    } else {
        colMeta.set(name, { type: 'text', n: nonEmpty });
    }
}

/**
 * Fold a computed series into the table as a first-class column, so it sorts,
 * filters and exports like anything that came out of the file. Re-running an
 * analysis overwrites in place rather than piling up duplicate columns.
 */
function _addComputedColumn(name, valueByRowIdx, { filterable = true } = {}) {
    for (let i = 0; i < rows.length; i++) {
        const v = valueByRowIdx.get(i);
        rows[i][name] = v === undefined || v === null ? '' : v;
    }
    if (!colNames.includes(name)) colNames.push(name);

    _typeOneColumn(name);

    if (!filterable) { filters.delete(name); return; }

    const meta = colMeta.get(name);
    if (meta.type === 'number')        filters.set(name, { type: 'number', lo: meta.min, hi: meta.max, active: false });
    else if (meta.type === 'category') filters.set(name, { type: 'category', excluded: new Set(), active: false });
    else                               filters.set(name, { type: 'text', q: '', active: false });
}

/** Drop a computed column and every trace of it. */
function _removeColumn(name) {
    const i = colNames.indexOf(name);
    if (i !== -1) colNames.splice(i, 1);
    colMeta.delete(name);
    filters.delete(name);
    hiddenCols.delete(name);
    renderCols.delete(name);
    if (sortCol === name) { sortCol = null; sortDir = 1; }
    for (const row of rows) delete row[name];
}

/** First free variant of `base`, so an imported column can't shadow one of ours. */
function _uniqueColName(base) {
    if (!colNames.includes(base)) return base;
    let n = 2;
    while (colNames.includes(`${base} (${n})`)) n++;
    return `${base} (${n})`;
}

/**
 * Add a column that draws the structure held as SMILES in `sourceCol`. It owns
 * no data, so it is excluded from filters and from exported CSV — the source
 * column carries the SMILES that actually needs to travel.
 */
function _addRenderColumn(name, sourceCol) {
    if (!colNames.includes(name)) colNames.push(name);
    renderCols.set(name, sourceCol);
    colMeta.set(name, { type: 'structure', n: 0 });
    filters.delete(name);
}

function _isNum(s) {
    if (s === '' || s === null) return false;
    const n = Number(s);
    return Number.isFinite(n);
}

function _median(sortedish) {
    const v = [...sortedish].sort((a, b) => a - b);
    const m = v.length >> 1;
    return v.length % 2 ? v[m] : (v[m - 1] + v[m]) / 2;
}

function _histogram(vals, min, max) {
    const bins = new Array(HIST_BINS).fill(0);
    if (max === min) { bins[0] = vals.length; return bins; }
    for (const v of vals) {
        let i = Math.floor(((v - min) / (max - min)) * HIST_BINS);
        if (i >= HIST_BINS) i = HIST_BINS - 1;
        if (i < 0) i = 0;
        bins[i]++;
    }
    return bins;
}

function _populateSelects() {
    const blank = '<option value="">— none —</option>';
    const opts  = colNames.map(c => `<option value="${_esc(c)}">${_esc(c)}</option>`).join('');
    document.getElementById('bt-smiles-col').innerHTML = blank + opts;
    document.getElementById('bt-id-col').innerHTML     = blank + opts;
}

function _autoDetect() {
    const lower = colNames.map(c => c.toLowerCase());
    const pick = (...keys) => {
        for (const k of keys) {
            const i = lower.indexOf(k);
            if (i !== -1) return colNames[i];
        }
        return '';
    };

    smilesCol = pick('smiles', 'smi', 'canonical_smiles', 'structure', 'molecule', 'mol');
    idCol     = pick('id', 'name', 'mol_id', 'compound_id', 'cmpd_id', 'molecule_id', 'title');

    // Fall back to any column whose first populated value parses as a molecule.
    if (!smilesCol && _RDKit) {
        for (const name of colNames) {
            const sample = rows.find(r => r[name])?.[name];
            if (!sample || String(sample).length > 400) continue;
            if (_parseOk(String(sample))) { smilesCol = name; break; }
        }
    }

    _setStructureColumn(smilesCol);
    document.getElementById('bt-smiles-col').value = smilesCol;
    document.getElementById('bt-id-col').value     = idCol;
}

function _parseOk(smiles) {
    let mol;
    try {
        mol = _RDKit.get_mol(smiles);
        return !!(mol && mol.is_valid());
    } catch { return false; }
    finally { if (mol) mol.delete(); }
}

// ----------------------------------------------------------------
// Filters — build the panel
// ----------------------------------------------------------------

function _buildFilters() {
    filters.clear();
    for (const name of colNames) {
        const meta = colMeta.get(name);
        if (meta.type === 'number') {
            filters.set(name, { type: 'number', lo: meta.min, hi: meta.max, active: false });
        } else if (meta.type === 'category') {
            filters.set(name, { type: 'category', excluded: new Set(), active: false });
        } else {
            filters.set(name, { type: 'text', q: '', active: false });
        }
    }
    _renderFilterCards();
}

function _renderFilterCards() {
    const host = document.getElementById('bt-column-filters');
    // Only columns that actually own a filter — draw-only and opt-out columns
    // have no state to bind, and wiring them would throw on first input.
    host.innerHTML = colNames.filter(n => filters.has(n))
        .map(name => _filterCardHtml(name)).join('');
    _wireFilterCards(host);
}

function _filterCardHtml(name) {
    const meta = colMeta.get(name);
    const f    = filters.get(name);
    const badge = meta.type === 'number' ? '123' : meta.type === 'category' ? 'cat' : 'txt';

    let body = '';
    if (meta.type === 'number') {
        const maxBin = Math.max(...meta.bins, 1);
        const hist = meta.bins.map((c, i) =>
            `<div class="bt-hist-bar" data-bin="${i}" style="height:${Math.max(2, (c / maxBin) * 100)}%"></div>`
        ).join('');
        body = `
            <div class="bt-range" data-col="${_esc(name)}">
                <div class="bt-hist">${hist}</div>
                <div class="bt-range-track"><div class="bt-range-fill"></div></div>
                <input type="range" class="bt-range-lo" min="0" max="1000" step="1" value="0">
                <input type="range" class="bt-range-hi" min="0" max="1000" step="1" value="1000">
            </div>
            <div class="bt-range-vals">
                <input type="text" class="bt-num-in bt-num-lo" value="${_fmtNum(meta.min)}">
                <span class="bt-range-stats">med ${_fmtNum(meta.median)}</span>
                <input type="text" class="bt-num-in bt-num-hi" value="${_fmtNum(meta.max)}">
            </div>`;
    } else if (meta.type === 'category') {
        const items = [...meta.uniques.entries()].map(([v, c]) => `
            <label class="bt-cat-row">
                <input type="checkbox" data-val="${_esc(v)}" checked>
                <span class="bt-cat-name" title="${_esc(v)}">${_esc(v)}</span>
                <span class="bt-cat-count">${c}</span>
            </label>`).join('');
        body = `
            <div class="bt-cat-actions">
                <button class="bt-link-btn" data-cat-act="all">All</button>
                <button class="bt-link-btn" data-cat-act="none">None</button>
            </div>
            <div class="bt-cat-list">${items}</div>`;
    } else {
        body = `<input type="text" class="bt-text-in" placeholder="contains…">`;
    }

    return `
        <div class="bt-filter-card" data-col="${_esc(name)}">
            <div class="bt-filter-card-head">
                <span class="bt-fc-dot"></span>
                <span class="bt-fc-name" title="${_esc(name)}">${_esc(name)}</span>
                <span class="bt-fc-badge">${badge}</span>
                <span class="bt-fc-arrow">▸</span>
            </div>
            <div class="bt-filter-card-body hidden">${body}</div>
        </div>`;
}

function _wireFilterCards(host) {
    host.querySelectorAll('.bt-filter-card').forEach(card => {
        const name = card.dataset.col;
        const f    = filters.get(name);
        const meta = colMeta.get(name);
        const body = card.querySelector('.bt-filter-card-body');

        card.querySelector('.bt-filter-card-head').addEventListener('click', () => {
            body.classList.toggle('hidden');
            card.classList.toggle('open', !body.classList.contains('hidden'));
        });

        if (meta.type === 'number') {
            const lo    = card.querySelector('.bt-range-lo');
            const hi    = card.querySelector('.bt-range-hi');
            const numLo = card.querySelector('.bt-num-lo');
            const numHi = card.querySelector('.bt-num-hi');

            const fromSlider = () => {
                let a = +lo.value, b = +hi.value;
                if (a > b) { [a, b] = [b, a]; }
                f.lo = _unscale(a, meta);
                f.hi = _unscale(b, meta);
                numLo.value = _fmtNum(f.lo);
                numHi.value = _fmtNum(f.hi);
                _syncRangeVisual(card, meta, f);
                _markActive(name);
                _applyFilters();
            };
            lo.addEventListener('input', fromSlider);
            hi.addEventListener('input', fromSlider);

            const fromText = () => {
                const a = parseFloat(numLo.value), b = parseFloat(numHi.value);
                f.lo = Number.isFinite(a) ? a : meta.min;
                f.hi = Number.isFinite(b) ? b : meta.max;
                if (f.lo > f.hi) [f.lo, f.hi] = [f.hi, f.lo];
                lo.value = _scale(f.lo, meta);
                hi.value = _scale(f.hi, meta);
                numLo.value = _fmtNum(f.lo);
                numHi.value = _fmtNum(f.hi);
                _syncRangeVisual(card, meta, f);
                _markActive(name);
                _applyFilters();
            };
            numLo.addEventListener('change', fromText);
            numHi.addEventListener('change', fromText);

            _syncRangeVisual(card, meta, f);

        } else if (meta.type === 'category') {
            card.querySelectorAll('.bt-cat-row input').forEach(cb => {
                cb.addEventListener('change', () => {
                    if (cb.checked) f.excluded.delete(cb.dataset.val);
                    else            f.excluded.add(cb.dataset.val);
                    _markActive(name);
                    _applyFilters();
                });
            });
            card.querySelectorAll('[data-cat-act]').forEach(btn => {
                btn.addEventListener('click', e => {
                    e.stopPropagation();
                    const all = btn.dataset.catAct === 'all';
                    f.excluded = all ? new Set() : new Set(meta.uniques.keys());
                    card.querySelectorAll('.bt-cat-row input').forEach(cb => { cb.checked = all; });
                    _markActive(name);
                    _applyFilters();
                });
            });

        } else {
            const inp = card.querySelector('.bt-text-in');
            let t = null;
            inp.addEventListener('input', () => {
                clearTimeout(t);
                t = setTimeout(() => {
                    f.q = inp.value.trim().toLowerCase();
                    _markActive(name);
                    _applyFilters();
                }, 180);
            });
        }
    });
}

// Sliders run 0–1000 so a column spanning 1e-6 to 1e6 still moves smoothly.
function _scale(v, meta)   { return meta.max === meta.min ? 0 : Math.round(((v - meta.min) / (meta.max - meta.min)) * 1000); }
function _unscale(v, meta) { return meta.min + (v / 1000) * (meta.max - meta.min); }

function _syncRangeVisual(card, meta, f) {
    const a = _scale(f.lo, meta) / 10;   // → percent
    const b = _scale(f.hi, meta) / 10;
    const fill = card.querySelector('.bt-range-fill');
    fill.style.left  = `${a}%`;
    fill.style.width = `${Math.max(0, b - a)}%`;
    card.querySelectorAll('.bt-hist-bar').forEach((bar, i) => {
        const c = ((i + 0.5) / HIST_BINS) * 100;
        bar.classList.toggle('out', c < a || c > b);
    });
}

function _markActive(name) {
    const f = filters.get(name);
    const meta = colMeta.get(name);
    if (f.type === 'number')        f.active = f.lo > meta.min || f.hi < meta.max;
    else if (f.type === 'category') f.active = f.excluded.size > 0;
    else                            f.active = f.q.length > 0;

    const card = document.querySelector(`.bt-filter-card[data-col="${CSS.escape(name)}"]`);
    if (card) card.classList.toggle('active', f.active);
}

function _resetAllFilters() {
    for (const [name, f] of filters) {
        const meta = colMeta.get(name);
        if (f.type === 'number')        { f.lo = meta.min; f.hi = meta.max; }
        else if (f.type === 'category') { f.excluded.clear(); }
        else                            { f.q = ''; }
        f.active = false;
    }
    structFilter.smarts = '';
    structFilter.simSmiles = '';
    if (structFilter.qmol) { try { structFilter.qmol.delete(); } catch {} }
    structFilter.qmol = null;
    structFilter.simFp = null;
    structFilter.qmolErr = '';
    structFilter.simErr = '';
    substructCache.clear();
    searchTerm = '';
    document.getElementById('bt-search').value = '';

    _renderFilterCards();
    _buildStructureFilters();
    _applyFilters();
}

// ----------------------------------------------------------------
// Structure filters — substructure (SMARTS) and similarity
// ----------------------------------------------------------------

function _buildStructureFilters() {
    const host = document.getElementById('bt-structure-filters');
    if (!smilesCol) { host.innerHTML = ''; return; }

    host.innerHTML = `
        <div class="bt-filter-card bt-struct-card ${structFilter.smarts ? 'active open' : ''}" id="bt-sub-card">
            <div class="bt-filter-card-head">
                <span class="bt-fc-dot"></span>
                <span class="bt-fc-name">Substructure</span>
                <span class="bt-fc-badge">SMARTS</span>
                <span class="bt-fc-arrow">▸</span>
            </div>
            <div class="bt-filter-card-body ${structFilter.smarts ? '' : 'hidden'}">
                <input type="text" class="bt-text-in" id="bt-smarts-in"
                       placeholder="e.g. c1ccccc1 or [NX3;H2]"
                       value="${_esc(structFilter.smarts)}">
                <div class="bt-fc-note" id="bt-smarts-note">Matches are highlighted in the table.</div>
            </div>
        </div>
        <div class="bt-filter-card bt-struct-card ${structFilter.simSmiles ? 'active open' : ''}" id="bt-sim-card">
            <div class="bt-filter-card-head">
                <span class="bt-fc-dot"></span>
                <span class="bt-fc-name">Similarity</span>
                <span class="bt-fc-badge">ECFP4</span>
                <span class="bt-fc-arrow">▸</span>
            </div>
            <div class="bt-filter-card-body ${structFilter.simSmiles ? '' : 'hidden'}">
                <input type="text" class="bt-text-in" id="bt-sim-in"
                       placeholder="reference SMILES"
                       value="${_esc(structFilter.simSmiles)}">
                <div class="bt-sim-row">
                    <span class="bt-sim-label">Tanimoto ≥</span>
                    <input type="range" id="bt-sim-cut" min="0" max="100" step="1"
                           value="${Math.round(structFilter.simCutoff * 100)}">
                    <span class="bt-sim-val" id="bt-sim-val">${structFilter.simCutoff.toFixed(2)}</span>
                </div>
                <div class="bt-fc-note" id="bt-sim-note">Morgan radius 2, 2048 bits.</div>
            </div>
        </div>`;

    host.querySelectorAll('.bt-filter-card-head').forEach(head => {
        head.addEventListener('click', () => {
            const card = head.parentElement;
            const body = card.querySelector('.bt-filter-card-body');
            body.classList.toggle('hidden');
            card.classList.toggle('open', !body.classList.contains('hidden'));
        });
    });

    let subT = null;
    document.getElementById('bt-smarts-in').addEventListener('input', e => {
        clearTimeout(subT);
        const v = e.target.value.trim();
        subT = setTimeout(() => _setSmarts(v), 250);
    });

    let simT = null;
    document.getElementById('bt-sim-in').addEventListener('input', e => {
        clearTimeout(simT);
        const v = e.target.value.trim();
        simT = setTimeout(() => _setSimRef(v), 250);
    });

    const cut = document.getElementById('bt-sim-cut');
    cut.addEventListener('input', e => {
        structFilter.simCutoff = +e.target.value / 100;
        document.getElementById('bt-sim-val').textContent = structFilter.simCutoff.toFixed(2);
        if (structFilter.simFp) _applyFilters();
    });
}

function _setSmarts(text) {
    if (structFilter.qmol) { try { structFilter.qmol.delete(); } catch {} structFilter.qmol = null; }
    structFilter.smarts  = text;
    structFilter.qmolErr = '';
    substructCache.clear();
    svgCache.clear();                     // drawings carry the highlight, so drop them

    const note = document.getElementById('bt-smarts-note');
    if (text) {
        try {
            let q = _RDKit.get_qmol(text);
            if (!q || !q.is_valid()) {
                if (q) q.delete();
                q = _RDKit.get_mol(text);   // plain SMILES queries are common
            }
            if (q && q.is_valid()) structFilter.qmol = q;
            else { if (q) q.delete(); structFilter.qmolErr = 'Not a valid query'; }
        } catch (err) {
            structFilter.qmolErr = 'Not a valid query';
        }
    }

    if (note) {
        note.textContent = structFilter.qmolErr || 'Matches are highlighted in the table.';
        note.classList.toggle('err', !!structFilter.qmolErr);
    }
    document.getElementById('bt-sub-card')?.classList.toggle('active', !!structFilter.qmol);
    _applyFilters();
}

function _setSimRef(text) {
    structFilter.simSmiles = text;
    structFilter.simErr    = '';
    structFilter.simFp     = null;

    const note = document.getElementById('bt-sim-note');
    if (text) {
        const fp = _fpFromSmiles(text);
        if (fp) {
            structFilter.simFp = fp;
            if (!fpsReady) _precomputeFingerprints();
        } else {
            structFilter.simErr = 'Could not parse that SMILES';
        }
    }
    if (note) {
        note.textContent = structFilter.simErr || 'Morgan radius 2, 2048 bits.';
        note.classList.toggle('err', !!structFilter.simErr);
    }
    document.getElementById('bt-sim-card')?.classList.toggle('active', !!structFilter.simFp);
    _applyFilters();
}

// ----------------------------------------------------------------
// Fingerprints and similarity coefficients
// ----------------------------------------------------------------

/**
 * The fingerprint flavours MinimalLib exposes, and what each is good at.
 * Bit counts verified against RDKit 2025.03.4.
 */
const FP_KINDS = {
    morgan:   { label: 'ECFP4 (Morgan r=2)',   short: 'ECFP4',    call: m => m.get_morgan_fp(JSON.stringify({ radius: 2, nBits: 2048 })) },
    rdkit:    { label: 'RDKit topological',    short: 'RDKit',    call: m => m.get_rdkit_fp(JSON.stringify({ nBits: 2048 })) },
    atompair: { label: 'Atom pairs',           short: 'AtomPair', call: m => m.get_atom_pair_fp(JSON.stringify({ nBits: 2048 })) },
    torsion:  { label: 'Topological torsions', short: 'Torsion',  call: m => m.get_topological_torsion_fp(JSON.stringify({ nBits: 2048 })) },
    maccs:    { label: 'MACCS keys',           short: 'MACCS',    call: m => m.get_maccs_fp() },
};

const COEFFS = {
    tanimoto: { label: 'Tanimoto', fn: (c, a, b) => (a + b - c === 0 ? 0 : c / (a + b - c)) },
    dice:     { label: 'Dice',     fn: (c, a, b) => (a + b === 0 ? 0 : (2 * c) / (a + b)) },
    cosine:   { label: 'Cosine',   fn: (c, a, b) => (a === 0 || b === 0 ? 0 : c / Math.sqrt(a * b)) },
};

// kind -> Map(rowIdx -> packed fp | null). Built lazily, per fingerprint type.
const fpCaches = new Map();

function _cacheFor(kind) {
    if (!fpCaches.has(kind)) fpCaches.set(kind, new Map());
    return fpCaches.get(kind);
}

function _fpFromSmiles(smiles, kind = 'morgan') {
    let mol;
    try {
        mol = _RDKit.get_mol(smiles);
        if (!mol || !mol.is_valid()) return null;
        return _packFp(FP_KINDS[kind].call(mol));
    } catch { return null; }
    finally { if (mol) mol.delete(); }
}

/** Fingerprint every row of the loaded table under `kind`, memoised. */
function _precomputeFingerprints(kind = 'morgan') {
    const cache = _cacheFor(kind);
    if (cache.size >= rows.length) return cache;

    _setStatus('Computing fingerprints…', 'loading');
    for (let i = 0; i < rows.length; i++) {
        if (cache.has(i)) continue;
        const smi = String(rows[i][smilesCol] ?? '');
        cache.set(i, smi ? _fpFromSmiles(smi, kind) : null);
    }
    if (kind === 'morgan') fpsReady = true;
    _setStatus(`${rows.length.toLocaleString()} rows · ${colNames.length} columns`, 'success');
    return cache;
}

/** Bit string → Uint32 words, with the popcount cached alongside. */
function _packFp(bitString) {
    const n = bitString.length;
    const words = new Uint32Array(Math.ceil(n / 32));
    let pop = 0;
    for (let i = 0; i < n; i++) {
        if (bitString.charCodeAt(i) === 49) {   // '1'
            words[i >>> 5] |= (1 << (i & 31));
            pop++;
        }
    }
    words.popcount = pop;
    return words;
}

/** Population count of a & b — the intersection both coefficients need. */
function _intersection(a, b) {
    let inter = 0;
    const n = Math.min(a.length, b.length);
    for (let i = 0; i < n; i++) {
        let v = a[i] & b[i];
        v = v - ((v >>> 1) & 0x55555555);
        v = (v & 0x33333333) + ((v >>> 2) & 0x33333333);
        inter += (((v + (v >>> 4)) & 0x0f0f0f0f) * 0x01010101) >>> 24;
    }
    return inter;
}

function _similarity(a, b, coeff = 'tanimoto') {
    return COEFFS[coeff].fn(_intersection(a, b), a.popcount, b.popcount);
}

function _tanimoto(a, b) { return _similarity(a, b, 'tanimoto'); }

/**
 * The best score two fingerprints could possibly reach, from popcounts alone —
 * every coefficient here is maximised when the intersection is min(a,b).
 * Lets a best-match scan skip the word loop for most candidates.
 */
function _simUpperBound(popA, popB, coeff) {
    return COEFFS[coeff].fn(Math.min(popA, popB), popA, popB);
}

function _substructMatch(rowIdx) {
    if (substructCache.has(rowIdx)) return substructCache.get(rowIdx);

    let result = null;
    const smi = String(rows[rowIdx][smilesCol] ?? '');
    if (smi && structFilter.qmol) {
        let mol;
        try {
            mol = _RDKit.get_mol(smi);
            if (mol && mol.is_valid()) {
                const raw = mol.get_substruct_match(structFilter.qmol);
                const hit = JSON.parse(raw || '{}');
                if (hit && Array.isArray(hit.atoms) && hit.atoms.length) result = hit;
            }
        } catch { /* unparseable row — treated as no match */ }
        finally { if (mol) mol.delete(); }
    }
    substructCache.set(rowIdx, result);
    return result;
}

// ----------------------------------------------------------------
// Filter evaluation
// ----------------------------------------------------------------

function _applyFilters() {
    const active = [...filters.entries()].filter(([, f]) => f.active);
    const searchCols = colNames.filter(c => !hiddenCols.has(c));

    filtered = [];
    for (let i = 0; i < rows.length; i++) {
        const row = rows[i];
        let keep = true;

        for (const [name, f] of active) {
            const raw = row[name];
            if (f.type === 'number') {
                const v = parseFloat(raw);
                if (!Number.isFinite(v) || v < f.lo || v > f.hi) { keep = false; break; }
            } else if (f.type === 'category') {
                if (f.excluded.has(String(raw ?? ''))) { keep = false; break; }
            } else {
                if (!String(raw ?? '').toLowerCase().includes(f.q)) { keep = false; break; }
            }
        }
        if (!keep) continue;

        if (searchTerm) {
            let hit = false;
            for (const c of searchCols) {
                if (String(row[c] ?? '').toLowerCase().includes(searchTerm)) { hit = true; break; }
            }
            if (!hit) continue;
        }

        // Structure filters last — they are the expensive ones.
        if (structFilter.qmol && !_substructMatch(i)) continue;

        if (structFilter.simFp) {
            const fp = _cacheFor('morgan').get(i);
            if (!fp || _tanimoto(fp, structFilter.simFp) < structFilter.simCutoff) continue;
        }

        filtered.push(i);
    }

    _applySort();
    _updateFilterSummary();
    _updateCounts();
    _renderRows();
}

function _applySort() {
    order = filtered.slice();
    if (!sortCol) return;

    const meta = colMeta.get(sortCol);
    const dir  = sortDir;

    if (meta && meta.type === 'number') {
        order.sort((a, b) => {
            const va = parseFloat(rows[a][sortCol]);
            const vb = parseFloat(rows[b][sortCol]);
            const na = !Number.isFinite(va), nb = !Number.isFinite(vb);
            if (na && nb) return 0;
            if (na) return 1;              // unparseable always sinks, both directions
            if (nb) return -1;
            return (va - vb) * dir;
        });
    } else {
        order.sort((a, b) => {
            const va = String(rows[a][sortCol] ?? '');
            const vb = String(rows[b][sortCol] ?? '');
            if (va === vb) return 0;
            if (va === '') return 1;
            if (vb === '') return -1;
            return va.localeCompare(vb, undefined, { numeric: true }) * dir;
        });
    }
}

function _updateFilterSummary() {
    const n = [...filters.values()].filter(f => f.active).length
            + (structFilter.qmol ? 1 : 0)
            + (structFilter.simFp ? 1 : 0)
            + (searchTerm ? 1 : 0);
    document.getElementById('bt-active-filter-count').textContent =
        n === 0 ? 'No active filters' : `${n} active filter${n === 1 ? '' : 's'}`;
}

function _updateCounts() {
    document.getElementById('bt-count-visible').textContent = order.length.toLocaleString();
    document.getElementById('bt-count-total').textContent   = rows.length.toLocaleString();
    const sel = document.getElementById('bt-count-selected');
    sel.textContent = `· ${selected.size.toLocaleString()} selected`;
    sel.classList.toggle('hidden', selected.size === 0);
    _syncHeaderState();
}

// ----------------------------------------------------------------
// Table rendering
// ----------------------------------------------------------------

function _shownCols() {
    return colNames.filter(c => !hiddenCols.has(c));
}

/** True for columns that only draw another column's SMILES and hold no data. */
function _isDrawOnly(c) {
    return renderCols.has(c) && renderCols.get(c) !== c;
}

/** Point the primary structure column at `name`, drawing it in place. */
function _setStructureColumn(name) {
    for (const [k, v] of renderCols) if (k === v) renderCols.delete(k);
    smilesCol = name;
    if (name) renderCols.set(name, name);
}

/** Move `name` so it sits immediately before `beforeName` (or last if null). */
function _reorderColumn(name, beforeName) {
    const from = colNames.indexOf(name);
    if (from === -1 || name === beforeName) return false;
    colNames.splice(from, 1);
    const at = beforeName === null ? colNames.length : colNames.indexOf(beforeName);
    colNames.splice(at === -1 ? colNames.length : at, 0, name);
    return true;
}

function _afterReorder() {
    _buildColumnList();
    _renderFilterCards();
    _rebuildHeader();
    _renderRows();
    _renderRecord();
}

function _structWidth() { return Math.max(84, Math.round(rowHeight * 1.75)); }

function _gridTemplate() {
    const parts = [`${CHECK_W}px`];
    for (const c of _shownCols()) {
        if (renderCols.has(c)) { parts.push(`${_structWidth()}px`); continue; }
        const meta = colMeta.get(c);
        parts.push(meta.type === 'number' ? '104px' : 'minmax(120px, 200px)');
    }
    return parts.join(' ');
}

function _rebuildHeader() {
    const head = document.getElementById('bt-header-row');
    const tpl  = _gridTemplate();
    head.style.gridTemplateColumns = tpl;
    document.getElementById('bt-rows').style.setProperty('--bt-cols', tpl);

    const arrow = c => (sortCol === c ? (sortDir === 1 ? '▲' : '▼') : '');
    let html = `<div class="bt-th bt-th-check">
        <input type="checkbox" id="bt-select-all" title="Select all filtered rows">
    </div>`;
    html += _shownCols().map(c => {
        // Structure columns aren't sortable — the picture has no useful order.
        // Sorting the matching ID column groups them identically.
        if (renderCols.has(c)) {
            return `<div class="bt-th bt-th-struct" data-col="${_esc(c)}" draggable="true"
                         title="${_esc(c)} — drag to reorder">
                <span class="bt-th-label">${_esc(c)}</span>
            </div>`;
        }
        const meta = colMeta.get(c);
        const f = filters.get(c);
        return `<div class="bt-th ${meta.type === 'number' ? 'num' : ''} ${sortCol === c ? 'sorted' : ''}"
                     data-col="${_esc(c)}" draggable="true"
                     title="${_esc(c)} — click to sort, drag to reorder">
            <span class="bt-th-label">${_esc(c)}</span>
            <span class="bt-th-dot ${f && f.active ? '' : 'off'}"></span>
            <span class="bt-th-arrow">${arrow(c)}</span>
        </div>`;
    }).join('');
    head.innerHTML = html;

    const all = document.getElementById('bt-select-all');
    _syncHeaderState();
    all.addEventListener('click', e => {
        e.stopPropagation();
        if (e.target.checked) order.forEach(i => selected.add(i));
        else                  order.forEach(i => selected.delete(i));
        _updateCounts();
        _renderRows();
    });
}

/**
 * Refresh the parts of the header that depend on filter/selection state,
 * without rebuilding it — _applyFilters runs on every keystroke.
 */
function _syncHeaderState() {
    for (const th of document.querySelectorAll('#bt-header-row .bt-th[data-col]')) {
        const f = filters.get(th.dataset.col);
        th.querySelector('.bt-th-dot')?.classList.toggle('off', !(f && f.active));
    }
    const all = document.getElementById('bt-select-all');
    if (all) all.checked = order.length > 0 && order.every(i => selected.has(i));
}

function _onHeaderClick(e) {
    const th = e.target.closest('.bt-th[data-col]');
    if (!th) return;
    const col = th.dataset.col;
    if (renderCols.has(col)) return;   // a drawing has no meaningful order
    if (sortCol !== col)      { sortCol = col; sortDir = 1; }
    else if (sortDir === 1)   { sortDir = -1; }
    else                      { sortCol = null; sortDir = 1; }
    _applySort();
    _rebuildHeader();
    _renderRows();
}

// ----------------------------------------------------------------
// Column reordering — drag a header, or drag a row in the Columns tab
// ----------------------------------------------------------------

let _dragCol = null;

/**
 * Wire drag-to-reorder onto a container of elements carrying data-col.
 * `axis` decides which half of an element counts as "insert before":
 * horizontal for table headers, vertical for the Columns tab list.
 */
function _wireColumnDrag(host, axis, onDrop) {
    host.addEventListener('dragstart', e => {
        const el = e.target.closest('[data-col]');
        if (!el) return;
        _dragCol = el.dataset.col;
        el.classList.add('bt-dragging');
        e.dataTransfer.effectAllowed = 'move';
        // Firefox refuses to start a drag without payload.
        try { e.dataTransfer.setData('text/plain', _dragCol); } catch {}
    });

    host.addEventListener('dragend', () => {
        _dragCol = null;
        host.querySelectorAll('.bt-dragging, .bt-drop-before, .bt-drop-after')
            .forEach(n => n.classList.remove('bt-dragging', 'bt-drop-before', 'bt-drop-after'));
    });

    host.addEventListener('dragover', e => {
        if (_dragCol === null) return;
        const el = e.target.closest('[data-col]');
        if (!el) return;
        e.preventDefault();
        e.dataTransfer.dropEffect = 'move';
        host.querySelectorAll('.bt-drop-before, .bt-drop-after')
            .forEach(n => n.classList.remove('bt-drop-before', 'bt-drop-after'));
        if (el.dataset.col === _dragCol) return;
        el.classList.add(_isBefore(e, el, axis) ? 'bt-drop-before' : 'bt-drop-after');
    });

    host.addEventListener('drop', e => {
        if (_dragCol === null) return;
        const el = e.target.closest('[data-col]');
        if (!el || el.dataset.col === _dragCol) return;
        e.preventDefault();

        const target = el.dataset.col;
        let before;
        if (_isBefore(e, el, axis)) {
            before = target;
        } else {
            // Drop after `target` — insert before whatever currently follows it.
            const i = colNames.indexOf(target);
            before = i === -1 || i + 1 >= colNames.length ? null : colNames[i + 1];
        }
        if (_reorderColumn(_dragCol, before)) onDrop();
    });
}

function _isBefore(e, el, axis) {
    const r = el.getBoundingClientRect();
    return axis === 'x'
        ? e.clientX < r.left + r.width / 2
        : e.clientY < r.top + r.height / 2;
}

function _queueRender() {
    if (_renderQueued) return;
    _renderQueued = true;
    requestAnimationFrame(() => { _renderQueued = false; _renderRows(); });
}

function _renderRows() {
    const scroll = document.getElementById('bt-scroll');
    const host   = document.getElementById('bt-rows');
    const noMatch = document.getElementById('bt-no-match');

    noMatch.classList.toggle('hidden', order.length > 0 || rows.length === 0);
    if (order.length === 0) { host.innerHTML = ''; host.style.padding = '0'; return; }

    const headH = document.getElementById('bt-header-row').offsetHeight || 0;
    const viewH = scroll.clientHeight || 600;
    const top   = Math.max(0, scroll.scrollTop - headH);

    // Clamped so a filter that shrinks the set while we're scrolled deep can
    // never leave start past end (which would render a blank table until the
    // browser got round to clamping scrollTop).
    const perView = Math.ceil(viewH / rowHeight) + 2 * OVERSCAN;
    let start = Math.floor(top / rowHeight) - OVERSCAN;
    start = Math.max(0, Math.min(start, Math.max(0, order.length - perView)));
    const end = Math.min(order.length, start + perView);

    host.style.paddingTop    = `${start * rowHeight}px`;
    host.style.paddingBottom = `${(order.length - end) * rowHeight}px`;
    host.style.gridTemplateColumns = '';   // rows are grids themselves, not the host

    const shown = _shownCols();
    const tpl   = _gridTemplate();
    let html = '';
    for (let k = start; k < end; k++) {
        html += _rowHtml(order[k], shown, tpl);
    }
    host.innerHTML = html;
}

function _rowHtml(idx, shown, tpl) {
    const row = rows[idx];
    const isSel = selected.has(idx);
    const isAct = activeRow === idx;

    let cells = `<div class="bt-td bt-td-check">
        <input type="checkbox" data-check="${idx}" ${isSel ? 'checked' : ''}>
    </div>`;

    for (const c of shown) {
        if (renderCols.has(c)) {
            const src = String(row[renderCols.get(c)] ?? '');
            // The row's own structure carries any active substructure highlight;
            // a matched reference structure is drawn plain.
            const svg = c === smilesCol
                ? _structureSvg(idx, src)
                : _drawSvg(src, _structWidth() - 6, rowHeight - 6);
            cells += `<div class="bt-td bt-td-struct" title="${_esc(src)}">${svg}</div>`;
            continue;
        }
        const meta = colMeta.get(c);
        const raw  = row[c];
        const txt  = raw === undefined || raw === null ? '' : String(raw);
        const disp = meta.type === 'number' && txt !== '' ? _fmtNum(parseFloat(txt)) : txt;
        cells += `<div class="bt-td ${meta.type === 'number' ? 'num' : ''}" title="${_esc(txt)}">${_esc(disp)}</div>`;
    }

    return `<div class="bt-tr ${isSel ? 'sel' : ''} ${isAct ? 'active' : ''}"
                 data-row="${idx}"
                 style="height:${rowHeight}px;grid-template-columns:${tpl}">${cells}</div>`;
}

/** The row's own structure, carrying any active substructure highlight. */
function _structureSvg(idx, smiles) {
    const hit = structFilter.qmol ? _substructMatch(idx) : null;
    return _drawSvg(smiles, _structWidth() - 6, rowHeight - 6, hit);
}

function _drawSvg(smiles, w, h, hit = null) {
    if (!smiles || !_RDKit) return '<span class="bt-nostruct">—</span>';

    const key = `${smiles}|${w}x${h}|${hit ? hit.atoms.join(',') : ''}`;
    if (svgCache.has(key)) return svgCache.get(key);

    let svg = '<span class="bt-nostruct">?</span>';
    let mol;
    try {
        mol = _RDKit.get_mol(smiles);
        if (mol && mol.is_valid()) {
            svg = hit
                ? mol.get_svg_with_highlights(JSON.stringify({
                      atoms: hit.atoms, bonds: hit.bonds || [], width: w, height: h,
                  }))
                : mol.get_svg(w, h);
        }
    } catch { /* leave the placeholder */ }
    finally { if (mol) mol.delete(); }

    svgCache.set(key, svg);
    return svg;
}

function _onRowClick(e) {
    const cb = e.target.closest('input[data-check]');
    if (cb) {
        const idx = +cb.dataset.check;
        if (e.shiftKey && lastClickedRow !== null) {
            const a = order.indexOf(lastClickedRow);
            const b = order.indexOf(idx);
            if (a !== -1 && b !== -1) {
                const [lo, hi] = a < b ? [a, b] : [b, a];
                for (let k = lo; k <= hi; k++) {
                    if (cb.checked) selected.add(order[k]); else selected.delete(order[k]);
                }
            }
        } else {
            if (cb.checked) selected.add(idx); else selected.delete(idx);
        }
        lastClickedRow = idx;
        _updateCounts();
        _renderRows();
        return;
    }

    const tr = e.target.closest('.bt-tr');
    if (!tr) return;
    activeRow = +tr.dataset.row;
    _setPanelTab('record');
    _renderRecord();
    _renderRows();
}

// ----------------------------------------------------------------
// Panel: tabs, column manager, record view
// ----------------------------------------------------------------

function _setPanelTab(tab) {
    panelTab = tab;
    document.querySelectorAll('.bt-panel-tab')
        .forEach(b => b.classList.toggle('active', b.dataset.tab === tab));
    document.getElementById('bt-tab-filters').classList.toggle('hidden',  tab !== 'filters');
    document.getElementById('bt-tab-analysis').classList.toggle('hidden', tab !== 'analysis');
    document.getElementById('bt-tab-columns').classList.toggle('hidden',  tab !== 'columns');
    document.getElementById('bt-tab-record').classList.toggle('hidden',   tab !== 'record');
}

function _buildColumnList() {
    const host = document.getElementById('bt-column-list');
    // Checkbox ids are index-based: column names carry spaces and punctuation
    // that would make an invalid id and break the label's `for` pairing.
    host.innerHTML = colNames.map((c, i) => `
        <div class="bt-col-row" data-col="${_esc(c)}" draggable="true">
            <span class="bt-col-grip" title="Drag to reorder">⠿</span>
            <input type="checkbox" id="bt-colcb-${i}" ${hiddenCols.has(c) ? '' : 'checked'}>
            <label class="bt-col-name" for="bt-colcb-${i}" title="${_esc(c)}">${_esc(c)}</label>
            <span class="bt-col-type">${colMeta.get(c).type}</span>
        </div>`).join('');

    host.querySelectorAll('.bt-col-row').forEach(row => {
        const cb = row.querySelector('input');
        cb.addEventListener('change', () => {
            if (cb.checked) hiddenCols.delete(row.dataset.col);
            else            hiddenCols.add(row.dataset.col);
            _updateColCount();
            _rebuildHeader();
            _renderRows();
        });
    });

    _updateColCount();
}

function _updateColCount() {
    document.getElementById('bt-visible-col-count').textContent =
        `${colNames.length - hiddenCols.size} of ${colNames.length} shown`;
}

const DESCRIPTORS = [
    ['amw',                'MW'],
    ['exactmw',            'Exact MW'],
    ['CrippenClogP',       'cLogP'],
    ['tpsa',               'TPSA'],
    ['NumHBD',             'HBD'],
    ['NumHBA',             'HBA'],
    ['NumRotatableBonds',  'Rot. bonds'],
    ['NumHeavyAtoms',      'Heavy atoms'],
    ['NumRings',           'Rings'],
    ['NumAromaticRings',   'Aromatic rings'],
    ['FractionCSP3',       'Fsp3'],
    ['NumAtomStereoCenters', 'Stereocentres'],
];

function _renderRecord() {
    const host = document.getElementById('bt-record-body');
    if (activeRow === null || !rows[activeRow]) {
        host.innerHTML = '<p class="bt-record-empty">Click a row to inspect the record.</p>';
        return;
    }

    const row   = rows[activeRow];
    const id    = idCol ? String(row[idCol] ?? '—') : `Row ${activeRow + 1}`;
    const smi   = smilesCol ? String(row[smilesCol] ?? '') : '';

    let svg = '', desc = '';
    if (smi && _RDKit) {
        let mol;
        try {
            mol = _RDKit.get_mol(smi);
            if (mol && mol.is_valid()) {
                svg = mol.get_svg(280, 200);
                const d = JSON.parse(mol.get_descriptors() || '{}');
                desc = DESCRIPTORS
                    .filter(([k]) => d[k] !== undefined)
                    .map(([k, label]) => `
                        <div class="bt-desc-item">
                            <span class="bt-desc-label">${label}</span>
                            <span class="bt-desc-value">${_fmtNum(d[k])}</span>
                        </div>`).join('');
            }
        } catch { /* fall through to the raw fields */ }
        finally { if (mol) mol.delete(); }
    }

    // Draw-only columns hold no value of their own — the structure they render
    // is already shown above, so listing them here would just be a blank row.
    const fields = colNames.filter(c => !_isDrawOnly(c)).map(c => `
        <div class="bt-rec-field">
            <span class="bt-rec-key" title="${_esc(c)}">${_esc(c)}</span>
            <span class="bt-rec-val" title="${_esc(String(row[c] ?? ''))}">${_esc(String(row[c] ?? '—'))}</span>
        </div>`).join('');

    // If a cross-file match has been run, show what this row paired with.
    let nearest = '';
    const nm = nearestMatch.get(activeRow);
    if (nm) {
        let nsvg = '';
        let nmol;
        try {
            nmol = _RDKit.get_mol(nm.smiles);
            if (nmol && nmol.is_valid()) nsvg = nmol.get_svg(260, 150);
        } catch { /* label alone still tells the story */ }
        finally { if (nmol) nmol.delete(); }

        nearest = `
            <div class="bt-rec-section-label">Nearest in ${_esc(refName || 'reference')}</div>
            <div class="bt-rec-nearest">
                <div class="bt-rec-nearest-head">
                    <span class="bt-rec-nearest-id" title="${_esc(nm.label)}">${_esc(nm.label)}</span>
                    <span class="bt-rec-nearest-score">${nm.score.toFixed(3)}</span>
                </div>
                ${nsvg ? `<div class="bt-rec-struct">${nsvg}</div>` : ''}
            </div>`;
    }

    host.innerHTML = `
        <div class="bt-rec-head">
            <span class="bt-rec-id" title="${_esc(id)}">${_esc(id)}</span>
            <div class="bt-rec-actions">
                ${smi ? `<button class="bt-link-btn" id="bt-rec-copy">Copy SMILES</button>` : ''}
            </div>
        </div>
        ${svg ? `<div class="bt-rec-struct">${svg}</div>` : ''}
        ${desc ? `<div class="bt-rec-section-label">Computed properties</div>
                  <div class="bt-desc-grid">${desc}</div>` : ''}
        ${nearest}
        <div class="bt-rec-section-label">Fields</div>
        <div class="bt-rec-fields">${fields}</div>`;

    document.getElementById('bt-rec-copy')?.addEventListener('click', e => {
        navigator.clipboard.writeText(smi).catch(() => {});
        e.target.textContent = 'Copied!';
        setTimeout(() => { e.target.textContent = 'Copy SMILES'; }, 1400);
    });
}

// ================================================================
// ANALYSIS TAB
// ================================================================

// Reference set loaded for cross-file matching.
let refRows     = [];
let refCols     = [];
let refSmilesCol = '';
let refIdCol     = '';
let refName      = '';
let refFps       = null;    // { kind, list: [{idx, fp, label, smiles}] }

// Row index -> { label, smiles, score } from the last cross-file run.
let nearestMatch = new Map();
// Columns pulled in from the reference file last run, so a re-run can retire
// them instead of leaving stale ones behind when the reference changes.
let importedRefCols = [];
let nnHits       = [];      // last nearest-neighbour result, for "Select these"

const NEAREST_LABEL_COL  = 'Nearest Match';
const NEAREST_STRUCT_COL = 'Nearest Structure';
const NEAREST_SCORE_COL  = 'Nearest Score';
const NEAREST_SMILES_COL = 'Nearest SMILES';

function _wireAnalysis() {
    document.getElementById('bt-nn-run').addEventListener('click', _runNearestNeighbours);
    document.getElementById('bt-nn-query').addEventListener('keydown', e => {
        if (e.key === 'Enter') _runNearestNeighbours();
    });
    document.getElementById('bt-nn-use-record').addEventListener('click', () => {
        if (activeRow === null || !smilesCol) {
            _setAnStatus('bt-nn-status', 'Open a record first', 'error');
            return;
        }
        document.getElementById('bt-nn-query').value = String(rows[activeRow][smilesCol] ?? '');
    });
    document.getElementById('bt-nn-select').addEventListener('click', () => {
        nnHits.forEach(h => selected.add(h.idx));
        _updateCounts();
        _renderRows();
    });

    document.getElementById('bt-ref-file-input')
        .addEventListener('change', e => { if (e.target.files[0]) _handleRefFile(e.target.files[0]); });
    document.getElementById('bt-ref-smiles-col')
        .addEventListener('change', e => { refSmilesCol = e.target.value; refFps = null; });
    document.getElementById('bt-ref-id-col')
        .addEventListener('change', e => { refIdCol = e.target.value; refFps = null; });
    document.getElementById('bt-ref-run').addEventListener('click', _runCrossFileMatch);

    // Changing the metric invalidates the reference fingerprints
    document.getElementById('bt-an-fp').addEventListener('change', () => { refFps = null; });
}

function _anKind()  { return document.getElementById('bt-an-fp').value; }
function _anCoeff() { return document.getElementById('bt-an-coeff').value; }

// ---- 1. Nearest neighbours to a query SMILES ----

function _runNearestNeighbours() {
    const q = document.getElementById('bt-nn-query').value.trim();
    const host = document.getElementById('bt-nn-results');
    document.getElementById('bt-nn-select').classList.add('hidden');
    nnHits = [];
    host.innerHTML = '';

    if (!smilesCol)  { _setAnStatus('bt-nn-status', 'Set a structure column first', 'error'); return; }
    if (!q)          { _setAnStatus('bt-nn-status', 'Enter a query SMILES', 'error'); return; }

    const kind  = _anKind();
    const coeff = _anCoeff();
    const qfp = _fpFromSmiles(q, kind);
    if (!qfp) { _setAnStatus('bt-nn-status', 'Could not parse that SMILES', 'error'); return; }

    _setAnStatus('bt-nn-status', 'Searching…', 'loading');

    const topN  = Math.max(1, Math.min(50, parseInt(document.getElementById('bt-nn-topn').value, 10) || 5));
    const scope = document.getElementById('bt-nn-filtered').checked
        ? order
        : rows.map((_, i) => i);

    const cache = _precomputeFingerprints(kind);
    const scored = [];
    for (const i of scope) {
        const fp = cache.get(i);
        if (!fp) continue;
        scored.push({ idx: i, score: _similarity(qfp, fp, coeff) });
    }
    scored.sort((a, b) => b.score - a.score);
    const hits = scored.slice(0, topN);

    if (!hits.length) {
        _setAnStatus('bt-nn-status', 'No parseable structures in scope', 'error');
        return;
    }

    // Score each hit under every coefficient and every other fingerprint, so
    // the ranking is visible rather than taken on trust.
    const others = Object.keys(FP_KINDS).filter(k => k !== kind);
    const otherQfps = new Map(others.map(k => [k, _fpFromSmiles(q, k)]));

    for (const h of hits) {
        const fp = cache.get(h.idx);
        h.coeffs = Object.fromEntries(
            Object.keys(COEFFS).map(c => [c, _similarity(qfp, fp, c)]));
        h.alt = {};
        for (const k of others) {
            const oq = otherQfps.get(k);
            const of = _fpFromSmiles(String(rows[h.idx][smilesCol] ?? ''), k);
            h.alt[k] = oq && of ? _similarity(oq, of, coeff) : null;
        }
    }

    nnHits = hits;
    host.innerHTML = hits.map((h, n) => _nnCardHtml(h, n + 1, coeff)).join('');
    host.querySelectorAll('.bt-nn-card').forEach(card => {
        card.addEventListener('click', () => {
            const idx = +card.dataset.idx;
            activeRow = idx;
            _renderRecord();
            _scrollToRow(idx);
            _renderRows();
        });
    });

    document.getElementById('bt-nn-select').classList.remove('hidden');
    _setAnStatus('bt-nn-status',
        `Top ${hits.length} of ${scored.length.toLocaleString()} by ${COEFFS[coeff].label} · ${FP_KINDS[kind].label}`,
        'success');
}

function _nnCardHtml(h, rank, coeff) {
    const row = rows[h.idx];
    const id  = idCol ? String(row[idCol] ?? '—') : `Row ${h.idx + 1}`;
    const smi = String(row[smilesCol] ?? '');
    const inScope = order.includes(h.idx);

    let svg = '';
    let mol;
    try {
        mol = _RDKit.get_mol(smi);
        if (mol && mol.is_valid()) svg = mol.get_svg(150, 92);
    } catch { /* card renders without a picture */ }
    finally { if (mol) mol.delete(); }

    const otherCoeffs = Object.keys(COEFFS).filter(c => c !== coeff)
        .map(c => `<span class="bt-nn-metric"><em>${COEFFS[c].label}</em>${h.coeffs[c].toFixed(3)}</span>`)
        .join('');
    const altFps = Object.entries(h.alt)
        .map(([k, v]) => `<span class="bt-nn-metric"><em>${FP_KINDS[k].short}</em>${v === null ? '—' : v.toFixed(3)}</span>`)
        .join('');

    return `<div class="bt-nn-card ${inScope ? '' : 'out-of-scope'}" data-idx="${h.idx}">
        <div class="bt-nn-card-top">
            <span class="bt-nn-rank">${rank}</span>
            <span class="bt-nn-id" title="${_esc(id)}">${_esc(id)}</span>
            <span class="bt-nn-score">${h.score.toFixed(3)}</span>
        </div>
        <div class="bt-nn-struct">${svg}</div>
        <div class="bt-nn-metrics">${otherCoeffs}</div>
        <div class="bt-nn-metrics bt-nn-alt">${altFps}</div>
    </div>`;
}

function _scrollToRow(rowIdx) {
    const pos = order.indexOf(rowIdx);
    if (pos === -1) return;
    const scroll = document.getElementById('bt-scroll');
    const headH  = document.getElementById('bt-header-row').offsetHeight || 0;
    scroll.scrollTop = Math.max(0, headH + pos * rowHeight - scroll.clientHeight / 2);
    _renderRows();
}

// ---- 2. Cross-file nearest match ----

async function _handleRefFile(file) {
    _setAnStatus('bt-ref-status', 'Parsing…', 'loading');
    refFps = null;
    refName = file.name.replace(/\.gz$/i, '').replace(/\.(csv|tsv|txt|sdf|sd)$/i, '');

    let table;
    try {
        table = await _readTable(file, msg => _setAnStatus('bt-ref-status', msg, 'loading'));
    } catch (err) {
        _setAnStatus('bt-ref-status', 'Parse error: ' + err.message, 'error');
        return;
    }

    refRows = table.rows;
    refCols = table.fields;
    if (!refRows.length || !refCols.length) {
        _setAnStatus('bt-ref-status', 'No data rows in that file', 'error');
        return;
    }

    const opts = refCols.map(c => `<option value="${_esc(c)}">${_esc(c)}</option>`).join('');
    document.getElementById('bt-ref-smiles-col').innerHTML = opts;
    document.getElementById('bt-ref-id-col').innerHTML =
        '<option value="">— row number —</option>' + opts;

    const lower = refCols.map(c => c.toLowerCase());
    // Exact header match first, then a substring sweep — reference files
    // routinely name the key column "RefID", "Cmpd Name" and the like.
    const pick = (...keys) => {
        for (const k of keys) { const i = lower.indexOf(k); if (i !== -1) return refCols[i]; }
        for (const k of keys) {
            const i = lower.findIndex(c => c.includes(k));
            if (i !== -1) return refCols[i];
        }
        return '';
    };
    refSmilesCol = table.smilesCol
        || pick('smiles', 'smi', 'canonical_smiles', 'structure', 'molecule', 'mol') || refCols[0];
    refIdCol     = pick('id', 'name', 'title') || '';
    document.getElementById('bt-ref-smiles-col').value = refSmilesCol;
    document.getElementById('bt-ref-id-col').value     = refIdCol;

    document.getElementById('bt-ref-config').classList.remove('hidden');
    _setAnStatus('bt-ref-status',
        `${refRows.length.toLocaleString()} reference molecules loaded${_failedNote(table)}`, 'success');
}

function _buildRefFps(kind) {
    if (refFps && refFps.kind === kind) return refFps.list;
    const list = [];
    for (let i = 0; i < refRows.length; i++) {
        const smi = String(refRows[i][refSmilesCol] ?? '');
        if (!smi) continue;
        const fp = _fpFromSmiles(smi, kind);
        if (!fp) continue;
        list.push({
            idx: i, fp, smiles: smi,
            label: refIdCol ? String(refRows[i][refIdCol] ?? '') : `ref ${i + 1}`,
        });
    }
    refFps = { kind, list };
    return list;
}

async function _runCrossFileMatch() {
    if (!smilesCol)    { _setAnStatus('bt-ref-status', 'Set a structure column first', 'error'); return; }
    if (!refRows.length) { _setAnStatus('bt-ref-status', 'Load a reference file first', 'error'); return; }

    const kind  = _anKind();
    const coeff = _anCoeff();
    const btn   = document.getElementById('bt-ref-run');
    const prog  = document.getElementById('bt-ref-progress');
    const bar   = document.getElementById('bt-ref-progress-bar');
    const scope = document.getElementById('bt-ref-filtered').checked
        ? order.slice()
        : rows.map((_, i) => i);

    btn.disabled = true;
    prog.classList.remove('hidden');
    bar.style.width = '0%';
    _setAnStatus('bt-ref-status', 'Fingerprinting reference set…', 'loading');
    await _yield();

    const refList = _buildRefFps(kind);
    if (!refList.length) {
        _setAnStatus('bt-ref-status', 'No parseable structures in the reference file', 'error');
        btn.disabled = false; prog.classList.add('hidden');
        return;
    }

    _setAnStatus('bt-ref-status', 'Fingerprinting this table…', 'loading');
    await _yield();
    const cache = _precomputeFingerprints(kind);

    const labels = new Map(), scores = new Map(), smiles = new Map();
    const refRowIdx = new Map();     // main row -> index of its matched reference row
    nearestMatch = new Map();
    const allScores = [];
    const CHUNK = 250;

    for (let n = 0; n < scope.length; n++) {
        const i  = scope[n];
        const fp = cache.get(i);
        if (fp) {
            let best = -1, bestRef = null;
            for (const r of refList) {
                // Skip the word loop when the popcounts alone rule this out.
                if (_simUpperBound(fp.popcount, r.fp.popcount, coeff) <= best) continue;
                const s = _similarity(fp, r.fp, coeff);
                if (s > best) { best = s; bestRef = r; }
            }
            if (bestRef) {
                labels.set(i, bestRef.label);
                scores.set(i, +best.toFixed(4));
                smiles.set(i, bestRef.smiles);
                refRowIdx.set(i, bestRef.idx);
                nearestMatch.set(i, { label: bestRef.label, smiles: bestRef.smiles, score: best });
                allScores.push(best);
            }
        }

        if (n % CHUNK === CHUNK - 1) {
            bar.style.width = `${((n + 1) / scope.length) * 100}%`;
            _setAnStatus('bt-ref-status',
                `Matching ${(n + 1).toLocaleString()} / ${scope.length.toLocaleString()}…`, 'loading');
            await _yield();
        }
    }

    // Added in display order. The draw-only column resolves its source at
    // render time, so the SMILES it reads can be appended after it.
    _addComputedColumn(NEAREST_LABEL_COL, labels);
    _addRenderColumn(NEAREST_STRUCT_COL, NEAREST_SMILES_COL);
    _addComputedColumn(NEAREST_SCORE_COL, scores);
    _addComputedColumn(NEAREST_SMILES_COL, smiles, { filterable: false });

    // Carry the matched reference row's own data across. Retire the previous
    // run's imports first, so pointing at a different reference file doesn't
    // leave columns behind that no longer mean anything.
    for (const name of importedRefCols) _removeColumn(name);
    importedRefCols = [];

    if (document.getElementById('bt-ref-import').checked) {
        // The structure and label columns are already surfaced above.
        const shown = new Set([refSmilesCol, refIdCol].filter(Boolean));
        for (const rc of refCols) {
            if (shown.has(rc)) continue;
            const colName = _uniqueColName(`Ref: ${rc}`);
            const vals = new Map();
            for (const [rowIdx, ri] of refRowIdx) vals.set(rowIdx, refRows[ri][rc]);
            _addComputedColumn(colName, vals);
            importedRefCols.push(colName);
        }
    }

    _renderFilterCards();
    _buildColumnList();
    _rebuildHeader();
    _applyFilters();
    _renderRecord();

    bar.style.width = '100%';
    setTimeout(() => prog.classList.add('hidden'), 600);
    btn.disabled = false;

    const med  = allScores.length ? _median(allScores) : 0;
    const high = allScores.filter(s => s >= 0.7).length;
    document.getElementById('bt-ref-summary').innerHTML = `
        <div class="bt-an-summary">
            <div class="bt-an-stat"><span>matched</span><b>${allScores.length.toLocaleString()}</b></div>
            <div class="bt-an-stat"><span>median</span><b>${med.toFixed(3)}</b></div>
            <div class="bt-an-stat"><span>≥ 0.70</span><b>${high.toLocaleString()}</b></div>
        </div>
        <p class="bt-an-hint">Added <b>${NEAREST_LABEL_COL}</b>, <b>${NEAREST_STRUCT_COL}</b>,
           <b>${NEAREST_SCORE_COL}</b> and <b>${NEAREST_SMILES_COL}</b>${
             importedRefCols.length
               ? `, plus <b>${importedRefCols.length}</b> column${importedRefCols.length === 1 ? '' : 's'}
                  carried across from the reference file as <b>Ref:&nbsp;…</b>`
               : ''}. Sort or filter on them like any other column; hide what you don't
           need under Columns.</p>`;

    _setAnStatus('bt-ref-status',
        `Matched against ${refList.length.toLocaleString()} reference molecules · ${COEFFS[coeff].label} on ${FP_KINDS[kind].label}`,
        'success');
}

/**
 * Hand control back to the browser between chunks so the progress bar paints
 * and the tab stays responsive. MessageChannel rather than setTimeout(0):
 * timers are clamped to ~4 ms when nested and to a full second in a background
 * tab, which would dominate the runtime of a long match.
 */
function _yield() {
    return new Promise(resolve => {
        const ch = new MessageChannel();
        ch.port1.onmessage = () => { ch.port1.close(); resolve(); };
        ch.port2.postMessage(0);
    });
}

function _setAnStatus(id, msg, type) {
    const el = document.getElementById(id);
    el.textContent = msg;
    el.className = 'bt-an-status' + (type ? ' ' + type : '') + (msg ? '' : ' hidden');
}

// ----------------------------------------------------------------
// Export
// ----------------------------------------------------------------

function _runExport(act) {
    const sel = act.endsWith('-selected');
    const subset = sel ? order.filter(i => selected.has(i)) : order;

    if (!subset.length) {
        _setStatus(sel ? 'Nothing selected' : 'Nothing to export', 'error');
        return;
    }

    if (act === 'csv-filtered' || act === 'csv-selected') {
        _download(_toCsv(subset), sel ? 'selected.csv' : 'filtered.csv');
    } else if (act === 'sdf-filtered' || act === 'sdf-selected') {
        _exportSdf(subset, sel ? 'selected.sdf' : 'filtered.sdf');
    } else if (act === 'copy-smiles') {
        if (!smilesCol) { _setStatus('No structure column set', 'error'); return; }
        _copy(subset.map(i => String(rows[i][smilesCol] ?? '')).filter(Boolean).join('\n'),
              `${subset.length} SMILES copied`);
    } else if (act === 'copy-ids') {
        const col = idCol || colNames[0];
        _copy(subset.map(i => String(rows[i][col] ?? '')).filter(Boolean).join('\n'),
              `${subset.length} IDs copied`);
    } else if (act === 'send-space' || act === 'send-plot') {
        if (!_hooks.onSendTo) { _setStatus('Hand-off not available', 'error'); return; }
        _hooks.onSendTo(act === 'send-space' ? 'space' : 'plot', _toCsv(subset));
    }
}

function _toCsv(idxs) {
    const cols = colNames.filter(c => !_isDrawOnly(c));   // draw-only cols hold no data
    const esc = v => {
        const s = v === undefined || v === null ? '' : String(v);
        return /[",\n\r]/.test(s) ? `"${s.replace(/"/g, '""')}"` : s;
    };
    const lines = [cols.map(esc).join(',')];
    for (const i of idxs) lines.push(cols.map(c => esc(rows[i][c])).join(','));
    return lines.join('\n');
}

/**
 * One SD record per row, data items in column order. Rows read from an SD file
 * keep their own mol block, unless the structure column has been pointed
 * elsewhere; every other row is drawn fresh from its SMILES, titled with its ID.
 * A row with no readable structure is written as an empty molecule rather than
 * dropped, so its data still travels.
 */
async function _exportSdf(idxs, filename) {
    if (!smilesCol) { _setStatus('No structure column set', 'error'); return; }

    const fromFile = smilesCol === sdfSmilesCol && molblocks.length === rows.length;
    const cols = colNames.filter(c => !_isDrawOnly(c) && !sdfMadeCols.has(c));
    const tags = cols.map(c => `> <${c.replace(/[<>]/g, '')}>\n`);

    const parts = [];
    let empty = 0;
    for (let n = 0; n < idxs.length; n++) {
        const i = idxs[n];
        let mb = fromFile ? molblocks[i] : '';
        if (!mb) {
            const title = idCol ? String(rows[i][idCol] ?? '') : '';
            mb = _molblockFromSmiles(String(rows[i][smilesCol] ?? ''), title);
            if (!mb) { mb = _emptyMolblock(title); empty++; }
        }

        let rec = mb.replace(/\s+$/, '') + '\n';
        for (let k = 0; k < cols.length; k++) rec += tags[k] + _sdValue(rows[i][cols[k]]) + '\n\n';
        parts.push(rec + '$$$$\n');

        if (n % SDF_CHUNK === SDF_CHUNK - 1) {
            _setStatus(`Writing SDF ${(n + 1).toLocaleString()} / ${idxs.length.toLocaleString()}…`, 'loading');
            await _yield();
        }
    }

    _download(parts.join(''), filename, 'chemical/x-mdl-sdfile');
    if (empty) {
        const s = empty === 1 ? '' : 's';
        _setStatus(`Downloaded ${filename} · ${empty.toLocaleString()} row${s} with no readable ` +
                   `structure written as empty molecule${s}`, 'success');
    }
}

function _molblockFromSmiles(smiles, title) {
    let mol, mb = '';
    try {
        mol = smiles ? _RDKit.get_mol(smiles) : null;
        if (mol && mol.is_valid()) mb = mol.get_molblock();
    } catch { /* caller writes an empty molecule */ }
    finally { if (mol) mol.delete(); }
    // Line one of a mol block is its title; RDKit leaves it blank.
    return mb ? _sdTitle(title) + mb.slice(mb.indexOf('\n')) : '';
}

function _emptyMolblock(title) {
    return `${_sdTitle(title)}\n  BoltzDraw\n\n  0  0  0  0  0  0  0  0  0  0999 V2000\nM  END`;
}

/** The title line holds one line of at most 80 characters. */
function _sdTitle(s) {
    return String(s).replace(/[\r\n]+/g, ' ').slice(0, 80);
}

/** A blank line ends an SD value, so any inside one are dropped. */
function _sdValue(v) {
    if (v === undefined || v === null) return '';
    return String(v).split(/\r?\n/).filter(l => l.trim()).join('\n');
}

function _download(text, filename, type = 'text/csv') {
    const url = URL.createObjectURL(new Blob([text], { type }));
    const a = document.createElement('a');
    a.href = url; a.download = filename;
    document.body.appendChild(a); a.click(); a.remove();
    setTimeout(() => URL.revokeObjectURL(url), 1000);
    _setStatus(`Downloaded ${filename}`, 'success');
}

function _copy(text, msg) {
    navigator.clipboard.writeText(text)
        .then(() => _setStatus(msg, 'success'))
        .catch(() => _setStatus('Clipboard blocked by the browser', 'error'));
}

// ----------------------------------------------------------------
// Helpers
// ----------------------------------------------------------------

function _setStatus(msg, type) {
    const el = document.getElementById('bt-status');
    el.textContent = msg;
    el.className = 'star-status' + (type ? ' ' + type : '') + (msg ? '' : ' hidden');
}

function _fmtNum(v) {
    if (v === null || v === undefined || v === '') return '';
    const n = typeof v === 'number' ? v : parseFloat(v);
    if (!Number.isFinite(n)) return String(v);
    if (Number.isInteger(n) && Math.abs(n) < 1e7) return String(n);
    const a = Math.abs(n);
    if (a !== 0 && (a < 1e-3 || a >= 1e7)) return n.toExponential(2);
    return String(parseFloat(n.toPrecision(5)));
}

function _esc(str) {
    return String(str ?? '')
        .replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;')
        .replace(/"/g, '&quot;');
}
