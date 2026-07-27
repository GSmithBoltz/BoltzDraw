/**
 * boltzspace.js
 * Chemical space explorer — UMAP + t-SNE projections with Plotly.
 */

// ----------------------------------------------------------------
// State
// ----------------------------------------------------------------

let _RDKit       = null;
let csvData      = [];
let validIndices = [];       // indices into csvData of rows with valid SMILES
let fingerprints = [];       // packed ECFP4, index-aligned with validIndices
let projections  = null;     // { umap: [[x,y],...], tsne: [[x,y],...] }
let highlightIds = new Set();
let activeScore  = 'score1'; // 'score1' | 'score2'
let selectedIds  = new Set();
let columnMap    = { smiles: '', id: '', score1: '', score2: '' };
let scriptsLoaded    = false;
let _crossUpdating   = false; // guard against infinite cross-plot update loop
let _activeSettingsTab = 'umap'; // 'umap' | 'tsne'
let _pickMethod    = 'butina';   // 'butina' | 'maxmin'
let _methodsBase   = '';         // projection half of the methods bar, so a pick can append

const COLORSCALE = [[0, '#f0ede8'], [0.5, '#67826b'], [1, '#35583d']];

// Butina needs the full pairwise distance matrix; above this it is slow enough
// to freeze the tab, and a pool that large hasn't really been filtered anyway.
const BUTINA_MAX = 3000;

const PICK_EXPLAINERS = {
    butina:
        '<strong>Butina (sphere exclusion).</strong> Groups the survivors into chemotypes by ' +
        'fingerprint similarity, then keeps the best-scoring molecule from each group.<br>' +
        '<em>Best for:</em> one representative per scaffold, when covering chemotypes matters more ' +
        'than hitting an exact count. This is the standard hit cherry-pick.',
    maxmin:
        '<strong>MaxMin (max–min dissimilarity).</strong> Repeatedly adds whichever molecule sits ' +
        'furthest from everything picked so far.<br>' +
        '<em>Best for:</em> filling an exact number of wells with maximum spread. It gravitates ' +
        'toward outliers, so filter hard before you run it.',
};

const CDN_SCRIPTS = [
    'https://cdn.jsdelivr.net/npm/papaparse@5.4.1/papaparse.min.js',
    'https://cdn.jsdelivr.net/npm/plotly.js-dist@2.27.0/plotly.min.js',
    'https://cdn.jsdelivr.net/npm/umap-js@1.3.3/lib/umap-js.min.js',
    // tsne-js removed: worker now uses a self-contained t-SNE implementation
];

// ----------------------------------------------------------------
// Init
// ----------------------------------------------------------------

export async function initBoltzSpace(rdkitInstance) {
    _RDKit = rdkitInstance;

    if (!scriptsLoaded) {
        _setUploadStatus('Loading libraries…', 'loading');
        await Promise.all(CDN_SCRIPTS.map(_loadScript));
        scriptsLoaded = true;
    }

    _wireListeners();
    _renderEmptyState();
    _setUploadStatus('', '');
    document.getElementById('space-upload-status').classList.add('hidden');
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
    document.getElementById('csv-file-input')
        .addEventListener('change', e => { if (e.target.files[0]) handleCsvUpload(e.target.files[0]); });
    document.getElementById('run-projection-btn')
        .addEventListener('click', runProjection);
    document.getElementById('apply-highlight-btn')
        .addEventListener('click', handleHighlightInput);
    document.getElementById('score1-btn')
        .addEventListener('click', () => setActiveScore('score1'));
    document.getElementById('score2-btn')
        .addEventListener('click', () => setActiveScore('score2'));

    // Settings tab switcher
    document.getElementById('settings-tab-umap')
        .addEventListener('click', () => _switchSettingsTab('umap'));
    document.getElementById('settings-tab-tsne')
        .addEventListener('click', () => _switchSettingsTab('tsne'));

    // Manual Mode toggle
    document.getElementById('manual-mode-toggle')
        .addEventListener('change', e => _setManualMode(e.target.checked));

    // Methods bar copy button
    document.getElementById('space-methods-copy-btn')
        .addEventListener('click', () => {
            const text = document.getElementById('space-methods-text').textContent;
            navigator.clipboard.writeText(text).catch(() => {});
            const btn = document.getElementById('space-methods-copy-btn');
            btn.textContent = 'Copied!';
            setTimeout(() => { btn.textContent = 'Copy'; }, 1500);
        });

    // ---- Diversity pick ----
    document.getElementById('pick-tab-butina')
        .addEventListener('click', () => _setPickMethod('butina'));
    document.getElementById('pick-tab-maxmin')
        .addEventListener('click', () => _setPickMethod('maxmin'));
    document.getElementById('pick-advanced-toggle')
        .addEventListener('change', () => _setPickMethod(_pickMethod));
    document.getElementById('pick-filter-toggle')
        .addEventListener('change', e => _setFilterEnabled(e.target.checked));
    document.getElementById('pick-filter-mode')
        .addEventListener('change', _onFilterModeChange);
    document.getElementById('run-pick-btn')
        .addEventListener('click', runDiversityPick);
    document.getElementById('clear-pick-btn')
        .addEventListener('click', () => { clearSelection(); _setPickStatus('', ''); });
    document.getElementById('pick-copy-btn')
        .addEventListener('click', _copyPickedIds);

    for (const id of ['pick-rank-col', 'pick-direction', 'pick-filter-value'])
        document.getElementById(id).addEventListener('input', _updateSurvivorCount);

    _setPickMethod('butina');
}

function _updateMethodsPanel(p) {
    const pcaPart  = p.pcaDims ? ` → RP ${p.pcaDims} dims` : '';
    const skipPart = p.skipped > 0 ? `, ${p.skipped} invalid` : '';
    const parts = [`ECFP4 r=2 2048-bit${pcaPart}`];
    if (p.ranUmap) parts.push(`UMAP: neighbors=${p.umapNeighbors} min_dist=${p.minDist}`);
    if (p.ranTsne) parts.push(`t-SNE: perplexity=${p.perplexity} iter=${p.nIter} lr=${p.lr}`);
    parts.push(`${p.nMols} molecules${skipPart}`);
    _methodsBase = parts.join('  ·  ');
    document.getElementById('space-methods-text').textContent = _methodsBase;
    document.getElementById('space-methods-bar').classList.remove('hidden');
}

function _appendPickToMethods(text) {
    document.getElementById('space-methods-text').textContent =
        text ? `${_methodsBase}  ·  ${text}` : _methodsBase;
}

function _switchSettingsTab(tab) {
    _activeSettingsTab = tab;
    document.getElementById('settings-tab-umap').classList.toggle('active', tab === 'umap');
    document.getElementById('settings-tab-tsne').classList.toggle('active', tab === 'tsne');
    document.getElementById('umap-settings-panel').classList.toggle('hidden', tab !== 'umap');
    document.getElementById('tsne-settings-panel').classList.toggle('hidden', tab !== 'tsne');
}

function _setManualMode(enabled) {
    // Enable or disable all settings inputs
    document.querySelectorAll('#umap-settings-panel .space-settings-input, #tsne-settings-panel .space-settings-input')
        .forEach(inp => { inp.disabled = !enabled; });
}

function _renderEmptyState() {
    document.getElementById('space-empty-state').innerHTML = `
        <svg width="64" height="64" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1">
            <circle cx="4"  cy="5"  r="2"/><circle cx="12" cy="3"  r="2"/><circle cx="20" cy="6"  r="2"/>
            <circle cx="3"  cy="14" r="2"/><circle cx="9"  cy="19" r="2"/><circle cx="17" cy="16" r="2"/>
            <circle cx="21" cy="20" r="2"/><circle cx="8"  cy="10" r="1.5"/><circle cx="15" cy="11" r="1.5"/>
        </svg>
        <p>Upload a CSV with a SMILES column<br>to explore your chemical space</p>
    `;
}

// ----------------------------------------------------------------
// CSV upload
// ----------------------------------------------------------------

export function handleCsvUpload(file) {
    _setUploadStatus('Parsing CSV…', 'loading');

    // Clear any plots/selections from a previous run before the new analysis
    _resetForNewCsv();

    Papa.parse(file, {
        header: true,
        skipEmptyLines: true,
        complete(results) {
            csvData = results.data;
            const cols = results.meta.fields || [];
            _populateSelects(cols);
            _autoDetect(cols);
            document.getElementById('space-column-config').classList.remove('hidden');
            _setUploadStatus(`${csvData.length} rows loaded — configure columns below`, 'success');
        },
        error(err) {
            _setUploadStatus('Parse error: ' + err.message, 'error');
        },
    });
}

function _resetForNewCsv() {
    // Reset state
    projections   = null;
    validIndices  = [];
    fingerprints  = [];
    highlightIds  = new Set();
    selectedIds   = new Set();
    _methodsBase  = '';

    // Tear down existing Plotly instances so a fresh CSV gets a clean canvas
    for (const id of ['umap-plot', 'tsne-plot']) {
        const el = document.getElementById(id);
        if (el && el._bsInitted) {
            try { Plotly.purge(el); } catch {}
            el._bsInitted = false;
        }
        el?.classList.remove('hidden');
    }

    // Hide plots grid, methods bar, card strip, downstream sections; show empty state
    document.getElementById('space-plots-grid').classList.add('hidden');
    document.getElementById('space-plots-grid').classList.remove('single-plot');
    document.getElementById('space-methods-bar').classList.add('hidden');
    document.getElementById('space-card-strip').classList.add('hidden');
    document.getElementById('space-highlight-section').classList.add('hidden');
    document.getElementById('space-score-section').classList.add('hidden');
    document.getElementById('space-pick-section').classList.add('hidden');
    document.getElementById('pick-copy-btn').classList.add('hidden');
    document.getElementById('space-empty-state').classList.remove('hidden');
    _setPickStatus('', '');

    // Reset highlight input
    const hi = document.getElementById('highlight-input');
    if (hi) hi.value = '';
}

function _populateSelects(cols) {
    const blank = '<option value="">— none —</option>';
    for (const id of ['smiles-col-select', 'id-col-select', 'score1-col-select', 'score2-col-select']) {
        document.getElementById(id).innerHTML =
            blank + cols.map(c => `<option value="${_esc(c)}">${_esc(c)}</option>`).join('');
    }
}

function _autoDetect(cols) {
    const lower = cols.map(c => c.toLowerCase());
    const pick = (...keys) => {
        for (const k of keys) {
            const i = lower.indexOf(k);
            if (i !== -1) return cols[i];
        }
        return '';
    };

    const smilesCol = pick('smiles', 'smi', 'canonical_smiles', 'structure');
    const idCol     = pick('id', 'name', 'mol_id', 'compound_id', 'cmpd_id', 'molecule_id');
    const s1Col     = pick('score1', 'score_1', 'docking_score', 'score', 'affinity', 'activity');
    const s2Col     = pick('score2', 'score_2', 'mw', 'logp');

    if (smilesCol) document.getElementById('smiles-col-select').value = smilesCol;
    if (idCol)     document.getElementById('id-col-select').value     = idCol;
    if (s1Col)     document.getElementById('score1-col-select').value = s1Col;
    if (s2Col)     document.getElementById('score2-col-select').value = s2Col;
}

// ----------------------------------------------------------------
// Projection
// ----------------------------------------------------------------

export async function runProjection() {
    columnMap.smiles = document.getElementById('smiles-col-select').value;
    columnMap.id     = document.getElementById('id-col-select').value;
    columnMap.score1 = document.getElementById('score1-col-select').value;
    columnMap.score2 = document.getElementById('score2-col-select').value;

    if (!columnMap.smiles) { _setUploadStatus('Select a SMILES column', 'error'); return; }
    if (!columnMap.score1) { _setUploadStatus('Select at least Score 1 column', 'error'); return; }

    // Fingerprints
    _setUploadStatus('Computing fingerprints…', 'loading');
    await _tick();

    const fps = [];
    validIndices = [];
    let skipped = 0;

    for (let i = 0; i < csvData.length; i++) {
        const fp = _computeFingerprint(csvData[i][columnMap.smiles]);
        if (fp) { fps.push(fp); validIndices.push(i); }
        else skipped++;
    }

    if (fps.length < 5) {
        _setUploadStatus(`Too few valid molecules (${fps.length}). Check your SMILES column.`, 'error');
        return;
    }

    // Bit-packed copies for Tanimoto; the dense arrays above stay for UMAP/t-SNE.
    fingerprints = fps.map(_packFp);

    const skipNote = skipped > 0 ? ` · ${skipped} invalid skipped` : '';

    // Decide which algorithms to run.
    // Auto mode = both. Manual mode = whichever mini-tab is active.
    const manual  = document.getElementById('manual-mode-toggle').checked;
    const ranUmap = !manual || _activeSettingsTab === 'umap';
    const ranTsne = !manual || _activeSettingsTab === 'tsne';

    // Reset projections — clears any previously rendered plot when the user
    // switches between manual algorithms.
    projections = { umap: null, tsne: null };

    // Read all params up front so they're available for the methods panel.
    const nNeighbors = Math.min(
        parseInt(document.getElementById('umap-neighbors').value) || 15,
        fps.length - 1
    );
    const minDist = parseFloat(document.getElementById('umap-min-dist').value) || 0.1;

    const tsnePerplexity = Math.min(
        parseFloat(document.getElementById('tsne-perplexity').value) || 30,
        Math.floor((fps.length - 1) / 3)   // perplexity must be < n/3
    );
    const tsneIter = parseInt(document.getElementById('tsne-iterations').value) || 200;
    const tsneLR   = parseFloat(document.getElementById('tsne-lr').value) || 100;
    const pcaDims  = Math.max(2, Math.min(50, fps.length - 2));

    // Show the grid (with correct single/dual layout) before any rendering
    _showPlotsGrid({ ranUmap, ranTsne });

    // ---- UMAP ----
    if (ranUmap) {
        _setUploadStatus(`${fps.length} molecules${skipNote} — Running UMAP…`, 'loading');
        await _tick();
        try {
            const umap = new UMAP({ nNeighbors, minDist, nComponents: 2 });
            projections.umap = await umap.fitAsync(fps);
        } catch (e) {
            _setUploadStatus('UMAP failed: ' + e.message, 'error');
            return;
        }
        renderPlots();
    }

    // ---- t-SNE ----
    if (ranTsne) {
        _setUploadStatus(
            ranUmap ? 'UMAP done — t-SNE running… 0%' : 't-SNE running… 0%',
            'loading'
        );
        await _tick();
        try {
            projections.tsne = await _runTsneWorker(fps, pcaDims, {
                dim: 2, perplexity: tsnePerplexity, earlyExaggeration: 4,
                learningRate: tsneLR, nIter: tsneIter, metric: 'euclidean',
            });
        } catch (e) {
            _setUploadStatus('t-SNE failed: ' + e.message, 'error');
            return;
        }
        renderPlots();
    }

    // Update score labels to actual column names
    document.getElementById('score1-btn').textContent = columnMap.score1 || 'Score 1';
    document.getElementById('score2-btn').textContent = columnMap.score2 || 'Score 2';

    document.getElementById('space-highlight-section').classList.remove('hidden');
    document.getElementById('space-score-section').classList.remove('hidden');
    document.getElementById('space-pick-section').classList.remove('hidden');
    _initPickControls();
    _setUploadStatus(`Done — ${fps.length} molecules${skipNote}`, 'success');

    _updateMethodsPanel({
        nMols: fps.length, skipped,
        ranUmap, ranTsne,
        umapNeighbors: nNeighbors, minDist,
        perplexity: tsnePerplexity, nIter: tsneIter, lr: tsneLR,
        pcaDims,
    });
}

function _computeFingerprint(smiles) {
    if (!smiles || typeof smiles !== 'string') return null;
    const s = smiles.trim();
    if (!s) return null;
    let mol;
    try {
        mol = _RDKit.get_mol(s);
        if (!mol || !mol.is_valid()) return null;
        const fpStr = mol.get_morgan_fp(JSON.stringify({ radius: 2, nBits: 2048 }));
        if (!fpStr || fpStr.length !== 2048) return null;
        return Array.from(fpStr).map(Number);
    } catch {
        return null;
    } finally {
        if (mol) mol.delete();
    }
}

function _showPlotsGrid({ ranUmap = true, ranTsne = true } = {}) {
    document.getElementById('space-empty-state').classList.add('hidden');
    const grid = document.getElementById('space-plots-grid');
    grid.classList.remove('hidden');

    // Single-plot layout: collapse to one column and hide the unused plot.
    const single = !(ranUmap && ranTsne);
    grid.classList.toggle('single-plot', single);
    document.getElementById('umap-plot').classList.toggle('hidden', !ranUmap);
    document.getElementById('tsne-plot').classList.toggle('hidden', !ranTsne);

    // Card strip is always visible once a projection exists; start with placeholder
    document.getElementById('space-card-strip').classList.remove('hidden');
    _showCardStripPlaceholder();
}

function _showCardStripPlaceholder() {
    document.getElementById('space-card-strip-inner').innerHTML =
        '<div class="space-card-strip-placeholder">Lasso-select molecules on the plots above to compare them here</div>';
}

// ----------------------------------------------------------------
// Plot rendering
// ----------------------------------------------------------------

export function renderPlots() {
    if (!projections) return;
    if (projections.umap) _renderOnePlot('umap-plot', projections.umap, 'UMAP');
    if (projections.tsne) _renderOnePlot('tsne-plot', projections.tsne, 't-SNE');
}

function _renderOnePlot(plotId, coords, title) {
    if (!coords) return;

    const scoreKey = columnMap[activeScore];

    // Two arrays: normal points and highlighted points
    const nx = [], ny = [], nColor = [], nHover = [], nCustom = [], nSel = [];
    const hx = [], hy = [],             hHover = [], hCustom = [], hSel = [];

    validIndices.forEach((rowIdx, i) => {
        const row   = csvData[rowIdx];
        const id    = _rowId(rowIdx);
        const score = parseFloat(row[scoreKey]);
        const x     = Array.isArray(coords[i]) ? coords[i][0] : coords[i * 2];
        const y     = Array.isArray(coords[i]) ? coords[i][1] : coords[i * 2 + 1];
        const hover = `<b>${_esc(id)}</b><br>${_esc(columnMap.score1)}: ${_fmtScore(row[columnMap.score1])}`
                    + (columnMap.score2 ? `<br>${_esc(columnMap.score2)}: ${_fmtScore(row[columnMap.score2])}` : '');

        if (highlightIds.has(id)) {
            if (selectedIds.has(id)) hSel.push(hx.length);
            hx.push(x); hy.push(y); hHover.push(hover); hCustom.push(rowIdx);
        } else {
            if (selectedIds.has(id)) nSel.push(nx.length);
            nx.push(x); ny.push(y);
            nColor.push(isNaN(score) ? null : score);
            nHover.push(hover); nCustom.push(rowIdx);
        }
    });

    // Plotly reads an empty array as "nothing selected, dim everything", so an
    // absent selection has to be null rather than [].
    const anySelected = selectedIds.size > 0;

    const traces = [
        {
            x: nx, y: ny, mode: 'markers', type: 'scatter',
            marker: {
                size: 6, opacity: 0.75,
                color: nColor,
                colorscale: COLORSCALE,
                showscale: false,
                line: { width: 0 },
            },
            selectedpoints: anySelected ? nSel : null,
            selected:   { marker: { size: 10, opacity: 1 } },
            unselected: { marker: { opacity: 0.12 } },
            text: nHover, hovertemplate: '%{text}<extra></extra>',
            customdata: nCustom, name: 'molecules',
        },
        {
            x: hx, y: hy, mode: 'markers', type: 'scatter',
            marker: {
                size: 12, opacity: 1.0, symbol: 'star',
                color: '#FCC400', line: { color: '#1a1a1a', width: 1.5 },
            },
            selectedpoints: anySelected ? hSel : null,
            selected:   { marker: { opacity: 1 } },
            unselected: { marker: { opacity: 0.15 } },
            text: hHover, hovertemplate: '%{text}<extra></extra>',
            customdata: hCustom, name: 'highlighted',
        },
    ];

    const layout = {
        title: { text: title, font: { size: 13, color: '#1a1a1a' }, x: 0.5, xanchor: 'center' },
        paper_bgcolor: 'rgba(0,0,0,0)',
        plot_bgcolor: '#f7f5f1',
        margin: { t: 36, r: 16, b: 40, l: 44 },
        font: { family: 'system-ui, sans-serif', color: '#1a1a1a' },
        xaxis: { showgrid: true, gridcolor: '#ddd9d2', zeroline: false, title: '' },
        yaxis: { showgrid: true, gridcolor: '#ddd9d2', zeroline: false, title: '' },
        showlegend: false,
        dragmode: 'lasso',
    };

    const config = { responsive: true, displayModeBar: true,
        modeBarButtonsToRemove: ['toImage', 'sendDataToCloud'],
        displaylogo: false };

    const el = document.getElementById(plotId);
    if (el._bsInitted) {
        Plotly.react(el, traces, layout, config);
    } else {
        Plotly.newPlot(el, traces, layout, config).then(() => {
            el._bsInitted = true;
            _attachPlotEvents(plotId);
        });
    }
}

// ----------------------------------------------------------------
// Lasso selection + cross-linking
// ----------------------------------------------------------------

function _attachPlotEvents(plotId) {
    const el      = document.getElementById(plotId);
    const otherId = plotId === 'umap-plot' ? 'tsne-plot' : 'umap-plot';

    el.on('plotly_selected', eventData => {
        if (_crossUpdating) return;
        if (!eventData || !eventData.points || !eventData.points.length) {
            clearSelection(); return;
        }
        _extractSelection(eventData.points);
        renderCardStrip();
        _crossUpdating = true;
        _syncSelection(otherId);
        _crossUpdating = false;
    });

    el.on('plotly_deselect', () => {
        if (_crossUpdating) return;
        clearSelection();
    });
}

function _extractSelection(points) {
    selectedIds = new Set();
    for (const pt of points) selectedIds.add(_rowId(pt.customdata));
}

function _syncSelection(plotId) {
    const el = document.getElementById(plotId);
    if (!el._bsInitted) return;

    // Rebuild which trace-local indices are selected for each trace
    const t0 = [], t1 = [];
    let ni = 0, hi = 0;

    validIndices.forEach(rowIdx => {
        const id = _rowId(rowIdx);
        if (highlightIds.has(id)) {
            if (selectedIds.has(id)) t1.push(hi);
            hi++;
        } else {
            if (selectedIds.has(id)) t0.push(ni);
            ni++;
        }
    });

    Plotly.restyle(el, { selectedpoints: [t0, t1] });
}

export function clearSelection() {
    selectedIds = new Set();
    _showCardStripPlaceholder();
    document.getElementById('pick-copy-btn').classList.add('hidden');
    _appendPickToMethods('');
    for (const id of ['umap-plot', 'tsne-plot']) {
        const el = document.getElementById(id);
        if (el._bsInitted) Plotly.restyle(el, { selectedpoints: [null, null] });
    }
}

function _runTsneWorker(fps, pcaDims, opts) {
    return new Promise((resolve, reject) => {
        // Fully self-contained worker — no importScripts, no network requests.
        // Implements random projection + t-SNE from scratch so we own the step loop
        // and can post real progress messages every 20 iterations.
        const code = `
onmessage = function(e) {
    var fps  = e.data.fps;
    var opts = e.data.opts;
    var k    = e.data.pcaDims;

    var data = fps;
    if (k && fps.length > 1 && fps[0].length > k) {
        try { data = _randProj(fps, k); } catch(_) {}
    }

    var result = _tsne(data, opts, function(pct) {
        postMessage({ type: 'progress', pct: pct });
    });
    postMessage({ type: 'done', data: result });
};

// ── Random projection (Johnson-Lindenstrauss) ───────────────────────────────
function _randProj(fps, k) {
    var n = fps.length, d = fps[0].length, sqK = Math.sqrt(k);
    var seed = 0x4d425350;
    function rng() { seed = (seed * 1664525 + 1013904223) & 0x7fffffff; return seed / 0x7fffffff; }
    var out = Array.from({length: n}, function() { return new Array(k).fill(0); });
    for (var f = 0; f < d; f++) {
        var row = new Array(k);
        for (var j = 0; j < k; j++) row[j] = rng() > 0.5 ? 1/sqK : -1/sqK;
        for (var i = 0; i < n; i++) {
            if (!fps[i][f]) continue;
            for (var j = 0; j < k; j++) out[i][j] += row[j];
        }
    }
    return out;
}

// ── t-SNE ───────────────────────────────────────────────────────────────────
function _tsne(data, opts, onProgress) {
    var n          = data.length;
    var d          = data[0].length;
    var nIter      = opts.nIter              || 200;
    var perplexity = Math.min(opts.perplexity || 30, Math.floor((n - 1) / 3));
    var lr         = opts.learningRate       || 100;
    var earlyExag  = opts.earlyExaggeration  || 4;

    var D2 = _dist2(data, n, d);
    var P  = _computeP(D2, n, perplexity);

    // Initialise embedding + optimiser state
    var Y     = Array.from({length: n}, function() { return [_randn()*1e-4, _randn()*1e-4]; });
    var iY    = Array.from({length: n}, function() { return [0, 0]; });
    var gains = Array.from({length: n}, function() { return [1, 1]; });
    // Pre-allocate per-step arrays
    var Q  = Array.from({length: n}, function() { return new Float32Array(n); });
    var dY = Array.from({length: n}, function() { return [0, 0]; });

    for (var iter = 0; iter < nIter; iter++) {
        var exag = iter < 100 ? earlyExag : 1;
        var mom  = iter < 250 ? 0.5 : 0.8;
        _step(Y, iY, gains, P, Q, dY, n, exag, lr, mom);
        if ((iter + 1) % 20 === 0 || iter === nIter - 1)
            onProgress(Math.round((iter + 1) / nIter * 100));
    }

    // Scale output to [0, 1]
    var mn0 = Infinity, mx0 = -Infinity, mn1 = Infinity, mx1 = -Infinity;
    for (var i = 0; i < n; i++) {
        if (Y[i][0] < mn0) mn0 = Y[i][0]; if (Y[i][0] > mx0) mx0 = Y[i][0];
        if (Y[i][1] < mn1) mn1 = Y[i][1]; if (Y[i][1] > mx1) mx1 = Y[i][1];
    }
    var r0 = mx0 - mn0 || 1, r1 = mx1 - mn1 || 1;
    return Y.map(function(y) { return [(y[0]-mn0)/r0, (y[1]-mn1)/r1]; });
}

function _randn() {
    var u = 1 - Math.random(), v = Math.random();
    return Math.sqrt(-2 * Math.log(u)) * Math.cos(2 * Math.PI * v);
}

function _dist2(data, n, d) {
    var D = Array.from({length: n}, function() { return new Float32Array(n); });
    for (var i = 0; i < n; i++)
        for (var j = i + 1; j < n; j++) {
            var s = 0;
            for (var k = 0; k < d; k++) { var v = data[i][k]-data[j][k]; s += v*v; }
            D[i][j] = D[j][i] = s;
        }
    return D;
}

function _computeP(D2, n, perplexity) {
    var logU = Math.log(perplexity);
    var P = Array.from({length: n}, function() { return new Float64Array(n); });
    for (var i = 0; i < n; i++) {
        var Di = D2[i];
        var beta = 1, bMin = -Infinity, bMax = Infinity;
        for (var t = 0; t < 50; t++) {
            var sum = 0, H = 0;
            for (var j = 0; j < n; j++) { if (j===i) continue; var p = Math.exp(-beta*Di[j]); P[i][j]=p; sum+=p; }
            if (sum < 1e-12) sum = 1e-12;
            for (var j = 0; j < n; j++) { if (j===i) continue; var p=P[i][j]/sum; if(p>1e-7) H-=p*Math.log(p); }
            var diff = H - logU;
            if (Math.abs(diff) < 1e-5) break;
            if (diff > 0) { bMin = beta; beta = bMax===Infinity ? beta*2 : (beta+bMax)/2; }
            else          { bMax = beta; beta = bMin===-Infinity ? beta/2 : (beta+bMin)/2; }
            H = 0;
        }
        var s = 0;
        for (var j = 0; j < n; j++) s += P[i][j];
        if (s > 0) for (var j = 0; j < n; j++) P[i][j] /= s;
    }
    // Symmetrise: P_ij = (P_i|j + P_j|i) / (2n)
    var sc = 1 / (2 * n);
    for (var i = 0; i < n; i++)
        for (var j = i+1; j < n; j++) {
            var v = Math.max((P[i][j]+P[j][i])*sc, 1e-12);
            P[i][j] = P[j][i] = v;
        }
    return P;
}

function _step(Y, iY, gains, P, Q, dY, n, exag, lr, mom) {
    // Q_ij and normaliser Z
    var Z = 0;
    for (var i = 0; i < n; i++) {
        for (var j = i+1; j < n; j++) {
            var a = Y[i][0]-Y[j][0], b = Y[i][1]-Y[j][1];
            var q = 1/(1+a*a+b*b);
            Q[i][j] = Q[j][i] = q; Z += 2*q;
        }
        dY[i][0] = dY[i][1] = 0;
    }
    if (Z < 1e-12) Z = 1e-12;

    // Gradient: dC/dy_i = 4 * sum_j (exag*P_ij - Q_ij/Z) * Q_ij * (y_i - y_j)
    for (var i = 0; i < n; i++) {
        for (var j = 0; j < n; j++) {
            if (i===j) continue;
            var pq = (exag*P[i][j] - Q[i][j]/Z) * Q[i][j] * 4;
            dY[i][0] += pq*(Y[i][0]-Y[j][0]);
            dY[i][1] += pq*(Y[i][1]-Y[j][1]);
        }
    }

    // Adaptive gains + momentum update + re-centre
    var m0 = 0, m1 = 0;
    for (var i = 0; i < n; i++) {
        for (var k = 0; k < 2; k++) {
            var sg = dY[i][k] >= 0, sm = iY[i][k] >= 0;
            gains[i][k] = sg===sm ? Math.max(gains[i][k]*0.8, 0.01) : gains[i][k]+0.2;
            iY[i][k] = mom*iY[i][k] - lr*gains[i][k]*dY[i][k];
            Y[i][k] += iY[i][k];
        }
        m0 += Y[i][0]; m1 += Y[i][1];
    }
    m0 /= n; m1 /= n;
    for (var i = 0; i < n; i++) { Y[i][0] -= m0; Y[i][1] -= m1; }
}`;

        const url = URL.createObjectURL(new Blob([code], { type: 'text/javascript' }));
        const w   = new Worker(url);
        URL.revokeObjectURL(url);

        w.onmessage = e => {
            if (e.data.type === 'progress') {
                _setUploadStatus(`UMAP done — t-SNE running… ${e.data.pct}%`, 'loading');
            } else {
                w.terminate();
                resolve(e.data.data);
            }
        };
        w.onerror = e => { w.terminate(); reject(new Error(e.message || 't-SNE worker failed')); };
        w.postMessage({ fps, opts, pcaDims });
    });
}

// ----------------------------------------------------------------
// Card strip
// ----------------------------------------------------------------

export function renderCardStrip() {
    if (selectedIds.size === 0) {
        _showCardStripPlaceholder();
        return;
    }

    const scoreKey = columnMap[activeScore];
    const rows = validIndices
        .filter(rowIdx => selectedIds.has(_rowId(rowIdx)))
        .map(rowIdx => csvData[rowIdx])
        .sort((a, b) => {
            const va = parseFloat(a[scoreKey]);
            const vb = parseFloat(b[scoreKey]);
            if (isNaN(va) && isNaN(vb)) return 0;
            if (isNaN(va)) return 1;
            if (isNaN(vb)) return -1;
            return vb - va;
        });

    const inner = document.getElementById('space-card-strip-inner');
    inner.innerHTML = rows.map(row => _buildCard(row)).join('');

    inner.querySelectorAll('.copy-btn').forEach(btn => {
        btn.addEventListener('click', () =>
            navigator.clipboard.writeText(btn.dataset.smiles).catch(() => {}));
    });
}

function _buildCard(row) {
    const id     = String(row[columnMap.id] ?? '—');
    const smiles = String(row[columnMap.smiles] ?? '');
    const s1val  = _fmtScore(row[columnMap.score1]);
    const s2val  = columnMap.score2 ? _fmtScore(row[columnMap.score2]) : '—';
    const s1lbl  = columnMap.score1 || 'Score 1';
    const s2lbl  = columnMap.score2 || 'Score 2';

    let svgContent = '';
    if (smiles && _RDKit) {
        let mol;
        try {
            mol = _RDKit.get_mol(smiles);
            if (mol && mol.is_valid()) svgContent = mol.get_svg(160, 100);
        } catch { /**/ }
        finally { if (mol) mol.delete(); }
    }

    return `<div class="space-card">
        <div class="space-card-header">
            <span class="space-card-id" title="${_esc(id)}">${_esc(id)}</span>
            <button class="copy-btn" data-smiles="${_esc(smiles)}" title="Copy SMILES">📋</button>
        </div>
        <div class="space-card-structure">${svgContent}</div>
        <div class="space-card-scores">
            <div class="space-score-item">
                <span class="space-score-label" title="${_esc(s1lbl)}">${_esc(s1lbl)}</span>
                <span class="space-score-value">${s1val}</span>
            </div>
            <div class="space-score-item">
                <span class="space-score-label" title="${_esc(s2lbl)}">${_esc(s2lbl)}</span>
                <span class="space-score-value">${s2val}</span>
            </div>
        </div>
    </div>`;
}

// ----------------------------------------------------------------
// Highlight input
// ----------------------------------------------------------------

export function handleHighlightInput() {
    const lines = document.getElementById('highlight-input').value
        .split('\n').map(l => l.trim()).filter(Boolean);

    highlightIds = new Set();

    for (const line of lines) {
        // Try ID match first, then SMILES match
        const byId     = csvData.find(r => String(r[columnMap.id])     === line);
        const bySmiles = csvData.find(r => String(r[columnMap.smiles]) === line);
        const match    = byId || bySmiles;
        if (match) highlightIds.add(String(match[columnMap.id]));
    }

    const statusEl = document.getElementById('highlight-status');
    statusEl.textContent = `Highlighted ${highlightIds.size} of ${lines.length} molecules`;
    statusEl.className = `star-status ${highlightIds.size > 0 ? 'success' : 'error'}`;
    statusEl.classList.remove('hidden');

    renderPlots();
}

// ----------------------------------------------------------------
// Score toggle
// ----------------------------------------------------------------

export function setActiveScore(scoreKey) {
    activeScore = scoreKey;
    document.getElementById('score1-btn').classList.toggle('active', scoreKey === 'score1');
    document.getElementById('score2-btn').classList.toggle('active', scoreKey === 'score2');
    renderPlots();
    if (selectedIds.size > 0) renderCardStrip();
}

// ----------------------------------------------------------------
// Diversity pick — fingerprint-space selection
//
// Both methods run on ECFP4 Tanimoto distances, never on the 2-D projection
// coordinates: t-SNE discards global distances entirely and UMAP's embedded
// distances are not a metric, so a pick made from the plot would be selecting
// on projection artefacts and would not survive a reseed.
//
//   Butina — Taylor/Butina sphere exclusion (Butina, J Chem Inf Comput Sci
//            1999, 39:747). Leader-follower clustering at a Tanimoto cutoff;
//            one representative per cluster.
//   MaxMin — greedy maximum-dissimilarity selection (Ashton et al.,
//            Quant Struct-Act Relat 2002, 21:598), as in RDKit's MaxMinPicker.
// ----------------------------------------------------------------

function _packFp(bits) {
    const words = new Uint32Array(bits.length >> 5);
    let pc = 0;
    for (let i = 0; i < bits.length; i++) {
        if (bits[i]) { words[i >> 5] |= (1 << (i & 31)); pc++; }
    }
    return { words, pc };
}

function _tanimoto(a, b) {
    const aw = a.words, bw = b.words;
    let c = 0;
    for (let k = 0; k < aw.length; k++) {
        let v = aw[k] & bw[k];
        v -= (v >>> 1) & 0x55555555;
        v = (v & 0x33333333) + ((v >>> 2) & 0x33333333);
        c += (((v + (v >>> 4)) & 0x0f0f0f0f) * 0x01010101) >>> 24;
    }
    const denom = a.pc + b.pc - c;
    return denom === 0 ? 0 : c / denom;
}

// Condensed (upper-triangle) index for the pairwise distance matrix, i < j.
function _dIdx(n, i, j) { return n * i - ((i * (i + 1)) >> 1) + j - i - 1; }

function _condensedDistances(fps) {
    const n = fps.length;
    const D = new Float32Array((n * (n - 1)) / 2);
    let k = 0;
    for (let i = 0; i < n; i++)
        for (let j = i + 1; j < n; j++)
            D[k++] = 1 - _tanimoto(fps[i], fps[j]);
    return D;
}

function _butina(D, n, distCut) {
    const nbrs = Array.from({ length: n }, () => []);
    for (let i = 0; i < n; i++)
        for (let j = i + 1; j < n; j++)
            if (D[_dIdx(n, i, j)] <= distCut) { nbrs[i].push(j); nbrs[j].push(i); }

    // Densest molecule seeds each cluster and absorbs its unassigned neighbours.
    // Ties break on descending index to match RDKit, which sorts (count, index)
    // tuples with reverse=True — the tie-break changes the cluster count.
    const order = Array.from({ length: n }, (_, i) => i)
        .sort((a, b) => nbrs[b].length - nbrs[a].length || b - a);

    const assigned = new Uint8Array(n);
    const clusters = [];
    for (const c of order) {
        if (assigned[c]) continue;
        assigned[c] = 1;
        const members = [c];                    // members[0] is the cluster centre
        for (const j of nbrs[c]) if (!assigned[j]) { assigned[j] = 1; members.push(j); }
        clusters.push(members);
    }
    return clusters;
}

// Simple mode: cluster count rises monotonically with the similarity cutoff,
// so bisect until it lands on the number of picks the user asked for.
function _butinaAutoCutoff(D, n, targetClusters) {
    let lo = 0.10, hi = 0.95, best = null;
    for (let it = 0; it < 18; it++) {
        const sim = (lo + hi) / 2;
        const clusters = _butina(D, n, 1 - sim);
        const diff = clusters.length - targetClusters;
        if (!best || Math.abs(diff) < Math.abs(best.diff)) best = { sim, clusters, diff };
        if (diff === 0) break;
        if (diff < 0) lo = sim; else hi = sim;
    }
    return best;
}

function _maxMin(fps, count, seedIdx) {
    const n = fps.length;
    const picked = [seedIdx];
    const minD = new Float64Array(n);
    for (let i = 0; i < n; i++) minD[i] = 1 - _tanimoto(fps[i], fps[seedIdx]);
    minD[seedIdx] = -1;   // -1 marks "already picked"

    while (picked.length < count) {
        let best = -1, bestD = -Infinity;
        for (let i = 0; i < n; i++) if (minD[i] > bestD) { bestD = minD[i]; best = i; }
        if (best < 0 || bestD < 0) break;
        picked.push(best);
        minD[best] = -1;
        for (let i = 0; i < n; i++) {
            if (minD[i] < 0) continue;
            const d = 1 - _tanimoto(fps[i], fps[best]);
            if (d < minD[i]) minD[i] = d;
        }
    }
    return picked;
}

// Fixed-seed LCG so a "random" first pick still reproduces across runs.
function _seededIndex(n) {
    const s = (0x4d425350 * 1664525 + 1013904223) & 0x7fffffff;
    return s % n;
}

// ---- Filter ----

// With the filter off, unscorable rows are still in play, so both comparators
// have to treat a missing score as the worst possible value rather than as NaN.
function _better(a, b, dirHigher) {
    if (!Number.isFinite(a)) return false;
    if (!Number.isFinite(b)) return true;
    return dirHigher ? a > b : a < b;
}

function _cmpScore(a, b, dirHigher) {   // best first, unscorable last
    if (!Number.isFinite(a)) return Number.isFinite(b) ? 1 : 0;
    if (!Number.isFinite(b)) return -1;
    return dirHigher ? b - a : a - b;
}

function _pickOpts() {
    return {
        filterOn:  document.getElementById('pick-filter-toggle').checked,
        rankCol:   document.getElementById('pick-rank-col').value,
        dirHigher: document.getElementById('pick-direction').value === 'higher',
        mode:      document.getElementById('pick-filter-mode').value,
        value:     parseFloat(document.getElementById('pick-filter-value').value),
        advanced:  document.getElementById('pick-advanced-toggle').checked,
        targetN:   Math.max(1, parseInt(document.getElementById('pick-n').value, 10) || 24),
    };
}

// Survivors carry fpIdx (into fingerprints/validIndices), rowIdx and their score.
function _computeSurvivors(o) {
    const all = validIndices.map((rowIdx, fpIdx) =>
        ({ fpIdx, rowIdx, v: parseFloat(csvData[rowIdx][o.rankCol]) }));

    if (!o.filterOn) return all;

    const scored = all.filter(s => Number.isFinite(s.v));

    if (o.mode === 'absolute') {
        if (!Number.isFinite(o.value)) return scored;
        return scored.filter(s => (o.dirHigher ? s.v >= o.value : s.v <= o.value));
    }

    scored.sort((a, b) => _cmpScore(a.v, b.v, o.dirHigher));
    const pct = Number.isFinite(o.value) ? Math.min(100, Math.max(1, o.value)) : 10;
    return scored.slice(0, Math.max(2, Math.round((scored.length * pct) / 100)));
}

function _filterLabel(o) {
    if (!o.filterOn) return 'no filter';
    return o.mode === 'absolute'
        ? `${o.rankCol} ${o.dirHigher ? '≥' : '≤'} ${o.value}`
        : `top ${o.value}% by ${o.rankCol}`;
}

// ---- UI ----

function _initPickControls() {
    const sel  = document.getElementById('pick-rank-col');
    const cols = [columnMap.score1, columnMap.score2].filter(Boolean);
    sel.innerHTML = cols.map(c => `<option value="${_esc(c)}">${_esc(c)}</option>`).join('');
    sel.value = columnMap.score1;
    _updateSurvivorCount();
}

function _setPickMethod(method) {
    _pickMethod = method;
    const advanced = document.getElementById('pick-advanced-toggle').checked;

    document.getElementById('pick-tab-butina').classList.toggle('active', method === 'butina');
    document.getElementById('pick-tab-maxmin').classList.toggle('active', method === 'maxmin');
    document.getElementById('pick-explainer').innerHTML = PICK_EXPLAINERS[method];

    document.getElementById('pick-butina-advanced')
        .classList.toggle('hidden', !(advanced && method === 'butina'));
    document.getElementById('pick-maxmin-advanced')
        .classList.toggle('hidden', !(advanced && method === 'maxmin'));

    // In advanced Butina the Tanimoto cutoff decides how many clusters you get,
    // so an explicit pick count would be a lie.
    document.getElementById('pick-n-row')
        .classList.toggle('hidden', advanced && method === 'butina');

    _updateSurvivorCount();   // the "too thin" threshold is method-dependent
}

function _onFilterModeChange() {
    const absolute = document.getElementById('pick-filter-mode').value === 'absolute';
    const dirHigher = document.getElementById('pick-direction').value === 'higher';
    const input = document.getElementById('pick-filter-value');

    document.getElementById('pick-filter-value-label').textContent =
        absolute ? 'Keep score' : 'Keep top';
    document.getElementById('pick-filter-suffix').textContent =
        absolute ? (dirHigher ? '≥' : '≤') : '%';

    input.step  = absolute ? '0.05' : '1';
    input.value = absolute ? '0.5'  : '10';
    _updateSurvivorCount();
}

function _setFilterEnabled(on) {
    for (const id of ['pick-filter-mode', 'pick-filter-value'])
        document.getElementById(id).disabled = !on;
    document.getElementById('pick-filter-suffix').classList.toggle('muted', !on);
    _updateSurvivorCount();
}

function _updateSurvivorCount() {
    const el = document.getElementById('pick-survivors');
    if (!fingerprints.length) { el.textContent = '—'; el.classList.remove('warn'); return; }

    const o = _pickOpts();
    _onFilterSuffixRefresh(o);

    if (!o.filterOn) {
        el.innerHTML = `No filter — all <strong>${validIndices.length}</strong> molecules`;
        const tooMany = _pickMethod === 'butina' && validIndices.length > BUTINA_MAX;
        el.classList.toggle('warn', tooMany);
        if (tooMany) el.innerHTML += `<br>Too many for Butina (max ${BUTINA_MAX}) — filter, or use MaxMin.`;
        return;
    }

    const n = _computeSurvivors(o).length;
    el.innerHTML = `<strong>${n}</strong> of ${validIndices.length} molecules survive the filter`;

    // A pool that barely exceeds the pick size has nothing left to diversify over.
    const tooThin = n < 10 || (_pickMethod === 'maxmin' && n < o.targetN * 2);
    el.classList.toggle('warn', tooThin);
    if (tooThin) el.innerHTML += '<br>Too thin to diversify — loosen the cutoff.';
}

function _onFilterSuffixRefresh(o) {
    if (o.mode === 'absolute')
        document.getElementById('pick-filter-suffix').textContent = o.dirHigher ? '≥' : '≤';
}

function _setPickStatus(msg, type) {
    const el = document.getElementById('pick-status');
    el.textContent = msg;
    el.className = type ? `star-status ${type}` : 'star-status';
    el.classList.toggle('hidden', !msg);
}

function _copyPickedIds() {
    navigator.clipboard.writeText([...selectedIds].join('\n')).catch(() => {});
    const btn = document.getElementById('pick-copy-btn');
    btn.textContent = `Copied ${selectedIds.size} IDs`;
    setTimeout(() => { btn.textContent = 'Copy picked IDs'; }, 1500);
}

// ---- Driver ----

export async function runDiversityPick() {
    if (!fingerprints.length) return;

    const btn = document.getElementById('run-pick-btn');
    btn.disabled = true;

    try {
        const o    = _pickOpts();
        const surv = _computeSurvivors(o);

        if (surv.length < 2) {
            _setPickStatus('Filter left fewer than 2 molecules — loosen the cutoff.', 'error');
            return;
        }

        const fps = surv.map(s => fingerprints[s.fpIdx]);
        let pickedLocal, detail;

        if (_pickMethod === 'butina') {
            if (surv.length > BUTINA_MAX) {
                _setPickStatus(
                    `Butina needs all ${surv.length}² pairwise distances. Filter to ` +
                    `≤${BUTINA_MAX} molecules, or use MaxMin.`, 'error');
                return;
            }

            _setPickStatus(`Clustering ${surv.length} molecules…`, 'loading');
            await _tick();

            const D = _condensedDistances(fps);
            let clusters, sim;
            if (o.advanced) {
                sim = parseFloat(document.getElementById('pick-butina-cutoff').value) || 0.65;
                clusters = _butina(D, fps.length, 1 - sim);
            } else {
                ({ clusters, sim } = _butinaAutoCutoff(D, fps.length, o.targetN));
            }

            const repMode = o.advanced ? document.getElementById('pick-butina-rep').value : 'score';
            let reps = clusters.map(members =>
                repMode === 'centroid'
                    ? members[0]
                    : members.reduce((best, m) =>
                        _better(surv[m].v, surv[best].v, o.dirHigher) ? m : best, members[0]));

            reps.sort((a, b) => _cmpScore(surv[a].v, surv[b].v, o.dirHigher));

            // Simple mode promises exactly N; bisection can overshoot by a cluster
            // or two, so drop the weakest representatives to land on the number.
            const dropped = o.advanced ? 0 : Math.max(0, reps.length - o.targetN);
            if (dropped) reps = reps.slice(0, o.targetN);

            pickedLocal = reps;
            detail = `Butina Tc=${sim.toFixed(2)} · ${clusters.length} clusters → ${reps.length} picks`
                   + (dropped ? ` (${dropped} weakest reps dropped)` : '');
        } else {
            const seedMode = o.advanced ? document.getElementById('pick-maxmin-seed').value : 'top';
            const seed = seedMode === 'top'
                ? surv.reduce((b, s, i) => (_better(s.v, surv[b].v, o.dirHigher) ? i : b), 0)
                : _seededIndex(fps.length);

            const count = Math.min(o.targetN, fps.length);
            _setPickStatus(`Selecting ${count} of ${surv.length}…`, 'loading');
            await _tick();

            pickedLocal = _maxMin(fps, count, seed);
            detail = `MaxMin · first=${seedMode === 'top' ? 'top scorer' : 'seeded random'}`
                   + ` → ${pickedLocal.length} picks`;
        }

        // The pick becomes the plot selection, which also drives the card strip.
        selectedIds = new Set(pickedLocal.map(i => _rowId(surv[i].rowIdx)));
        renderPlots();
        renderCardStrip();

        document.getElementById('pick-copy-btn').classList.remove('hidden');
        _appendPickToMethods(`Pick: ${_filterLabel(o)} · ${detail}`);
        _setPickStatus(
            `${selectedIds.size} picked from ${surv.length} survivors ` +
            `(${validIndices.length} total)`, 'success');
    } catch (e) {
        _setPickStatus('Pick failed: ' + e.message, 'error');
    } finally {
        btn.disabled = false;
    }
}

// ----------------------------------------------------------------
// Helpers
// ----------------------------------------------------------------

function _rowId(rowIdx) {
    return String(csvData[rowIdx][columnMap.id] ?? rowIdx);
}

function _setUploadStatus(msg, type) {
    const el = document.getElementById('space-upload-status');
    el.textContent = msg;
    el.className = type ? `star-status ${type}` : 'star-status';
    el.classList.toggle('hidden', !msg);
}

function _fmtScore(val) {
    const n = parseFloat(val);
    return isNaN(n) ? '—' : n.toFixed(2);
}

function _esc(str) {
    return String(str)
        .replace(/&/g, '&amp;').replace(/</g, '&lt;')
        .replace(/>/g, '&gt;').replace(/"/g, '&quot;');
}

function _tick() {
    return new Promise(r => setTimeout(r, 0));
}
