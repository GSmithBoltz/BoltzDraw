/**
 * boltzplot.js
 * General-purpose scatter plot: upload CSV/TSV, pick axes, colour, label, SMILES.
 * Advanced: marginal histograms · linear regression · density overlay · correlation stats ·
 * classification stats (X = score, Y = assay value with an activity cutoff: ROC/PR AUC, top-k% precision).
 * Card strip: lasso-select points to compare structures.
 */

// ---- State ----
let _RDKit            = null;
let _data             = [];
let _cols             = [];
let _currentPts       = [];      // [{idx, x, y, yq, color, label, smiles}, …]  yq = '<' | '>' | ''
let _currentConfig    = null;    // {xCol, yCol, colorCol, labelCol, smilesCol}
let _highlightIndices = new Set();
let _initted          = false;
let _scriptsOk        = false;
let _ptsGen           = 0;       // bumped per Generate; keys the stats cache
let _statsCache       = { key: null, stats: null };
let _cutoffGuessCol   = null;    // Y column the activity cutoff was last guessed for

// ---- Constants ----
const COLORSCALE = [[0, '#f0ede8'], [0.5, '#67826b'], [1, '#35583d']];

const CAT_COLORS = [
    '#35583d','#5078D2','#e05c5c','#FCC400','#7b4daa',
    '#2eaaa0','#e07b2e','#4daa7b','#aa4d7b','#2e6ae0',
    '#aaa04d','#4daaaa','#e04d91','#8b5e3c',
];

const PLOTLY_CFG = {
    responsive: true, displayModeBar: true, displaylogo: false,
    modeBarButtonsToRemove: ['sendDataToCloud'],
    toImageButtonOptions: { format: 'png', filename: 'boltzplot', height: 900, width: 1400, scale: 2 },
    edits: { annotationPosition: true },   // drag the stats box off the points
};

const BOOT_MAX_N    = 5000;   // above this the CI is narrow and resampling gets slow
const KENDALL_MAX_N = 5000;   // τ is O(n²)
const TOP_FRACS     = [0.10, 0.05];

const CDN_SCRIPTS = [
    'https://cdn.jsdelivr.net/npm/papaparse@5.4.1/papaparse.min.js',
    'https://cdn.jsdelivr.net/npm/plotly.js-dist@2.27.0/plotly.min.js',
];

// ================================================================
// Init
// ================================================================

export async function initBoltzPlot(rdkit) {
    _RDKit = rdkit || null;

    if (!_scriptsOk) {
        _setStatus('Loading libraries…', 'loading');
        await Promise.all(CDN_SCRIPTS.map(_loadScript));
        _scriptsOk = true;
    }
    _wireListeners();
    _renderEmptyState();
    document.getElementById('bp-status').classList.add('hidden');
}

function _loadScript(src) {
    return new Promise((resolve, reject) => {
        if (document.querySelector(`script[src="${src}"]`)) { resolve(); return; }
        const s = document.createElement('script');
        s.src = src; s.onload = resolve;
        s.onerror = () => reject(new Error(`Failed to load ${src}`));
        document.head.appendChild(s);
    });
}

function _wireListeners() {
    document.getElementById('bp-file-input')
        .addEventListener('change', e => { if (e.target.files[0]) _handleFile(e.target.files[0]); });
    document.getElementById('bp-generate-btn')
        .addEventListener('click', _generatePlot);
    document.getElementById('bp-apply-highlight-btn')
        .addEventListener('click', _applyHighlight);
    document.getElementById('bp-advanced-toggle')
        .addEventListener('click', _toggleAdvanced);

    // Live-update toggles when a plot already exists
    for (const id of ['bp-opt-logy', 'bp-opt-marginals', 'bp-opt-regression', 'bp-opt-density', 'bp-opt-stats',
                      'bp-opt-eval', 'bp-eval-dir', 'bp-eval-thresh']) {
        document.getElementById(id)
            .addEventListener('change', () => { if (_initted) _redrawTraces(); });
    }
    document.getElementById('bp-opt-eval').addEventListener('change', e => {
        document.getElementById('bp-eval-config').classList.toggle('hidden', !e.target.checked);
    });
}

function _renderEmptyState() {
    document.getElementById('bp-empty-state').innerHTML = `
        <svg width="64" height="64" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1">
            <line x1="3" y1="20" x2="21" y2="20"/>
            <line x1="3" y1="3"  x2="3"  y2="20"/>
            <circle cx="7"  cy="14" r="1.5"/>
            <circle cx="11" cy="9"  r="1.5"/>
            <circle cx="15" cy="16" r="1.5"/>
            <circle cx="9"  cy="5"  r="1.5"/>
            <circle cx="17" cy="12" r="1.5"/>
            <circle cx="13" cy="7"  r="1.5"/>
        </svg>
        <p>Upload a CSV or TSV to get started</p>
    `;
}

// ================================================================
// File upload
// ================================================================

/** Load a File/Blob from another mode (e.g. a filtered subset from BoltzTable). */
export function loadCsvFile(file) {
    _handleFile(file);
}

function _handleFile(file) {
    _setStatus('Parsing…', 'loading');
    _data = []; _cols = [];
    document.getElementById('bp-column-config').classList.add('hidden');
    document.getElementById('bp-highlight-section').classList.add('hidden');

    const plotEl = document.getElementById('bp-plot');
    if (_initted) { try { Plotly.purge(plotEl); } catch {} _initted = false; }
    plotEl.classList.add('hidden');
    document.getElementById('bp-empty-state').classList.remove('hidden');
    document.getElementById('bp-card-strip').classList.add('hidden');

    Papa.parse(file, {
        header: true,
        skipEmptyLines: true,
        delimiter: file.name.toLowerCase().endsWith('.tsv') ? '\t' : '',
        complete(r) {
            _data = r.data;
            _cols = r.meta.fields || [];
            _populateSelects();
            _autoDetect();
            document.getElementById('bp-column-config').classList.remove('hidden');
            _setStatus(`${_data.length} rows loaded`, 'success');
        },
        error(err) { _setStatus('Parse error: ' + err.message, 'error'); },
    });
}

function _populateSelects() {
    const none = '<option value="">— none —</option>';
    const opts = _cols.map(c => `<option value="${_esc(c)}">${_esc(c)}</option>`).join('');
    document.getElementById('bp-x-col').innerHTML      = opts;
    document.getElementById('bp-y-col').innerHTML      = opts;
    document.getElementById('bp-color-col').innerHTML  = none + opts;
    document.getElementById('bp-label-col').innerHTML  = none + opts;
    document.getElementById('bp-smiles-col').innerHTML = none + opts;
}

function _autoDetect() {
    const numCols  = _cols.filter(_isNumeric);
    const colorKW  = ['score','value','class','type','category','activity','group','cluster'];
    const labelKW  = ['id','name','label','title','compound','molecule'];
    const smilesKW = ['smiles','smi','canonical_smiles','structure'];

    _sel('bp-x-col', numCols[0] || _cols[0] || '');
    _sel('bp-y-col', numCols[1] || _cols[1] || '');

    const colorHit = numCols[2] || _cols.find(c => colorKW.some(k => c.toLowerCase().includes(k)));
    if (colorHit) _sel('bp-color-col', colorHit);

    const labelHit = _cols.find(c => labelKW.some(k => c.toLowerCase() === k || c.toLowerCase().startsWith(k + '_')));
    _sel('bp-label-col', labelHit || _cols[0] || '');

    const smilesHit = _cols.find(c => smilesKW.some(k => c.toLowerCase().includes(k)));
    if (smilesHit) _sel('bp-smiles-col', smilesHit);
}

function _isNumeric(col) {
    const sample = _data.slice(0, 60).map(r => r[col]).filter(v => v !== '' && v != null);
    if (!sample.length) return false;
    return sample.filter(v => !isNaN(parseFloat(v))).length / sample.length > 0.75;
}

function _sel(id, val) {
    const el = document.getElementById(id);
    if (el && [...el.options].some(o => o.value === val)) el.value = val;
}

function _toggleAdvanced() {
    const body  = document.getElementById('bp-advanced-body');
    const arrow = document.querySelector('#bp-advanced-toggle .bp-adv-arrow');
    const willClose = !body.classList.contains('hidden');
    body.classList.toggle('hidden', willClose);
    if (arrow) arrow.textContent = willClose ? '▸' : '▾';
}

// ================================================================
// Plot generation
// ================================================================

function _generatePlot() {
    const xCol      = document.getElementById('bp-x-col').value;
    const yCol      = document.getElementById('bp-y-col').value;
    const colorCol  = document.getElementById('bp-color-col').value;
    const labelCol  = document.getElementById('bp-label-col').value;
    const smilesCol = document.getElementById('bp-smiles-col').value;

    if (!xCol || !yCol) { _setStatus('Select X and Y columns', 'error'); return; }

    const pts = [];
    for (const row of _data) {
        const x = parseFloat(row[xCol]);
        const { v: y, q: yq } = _parseCensored(row[yCol]);
        if (isNaN(x) || isNaN(y)) continue;
        pts.push({
            idx:    pts.length,
            x, y, yq,
            color:  colorCol  ? row[colorCol]                 : null,
            label:  labelCol  ? String(row[labelCol]  ?? '')  : '',
            smiles: smilesCol ? String(row[smilesCol] ?? '')  : '',
        });
    }

    if (!pts.length) { _setStatus('No numeric rows in X/Y columns', 'error'); return; }

    _currentPts    = pts;
    _currentConfig = { xCol, yCol, colorCol, labelCol, smilesCol };
    _highlightIndices = new Set();
    _ptsGen++;
    if (yCol !== _cutoffGuessCol) { _applyYHeaderDefaults(yCol); _cutoffGuessCol = yCol; }

    // A log axis would silently drop zeros and negatives, so offer it only when every Y is positive.
    const logOk = pts.every(p => p.y > 0);
    const logEl = document.getElementById('bp-opt-logy');
    logEl.disabled = !logOk;
    if (!logOk) logEl.checked = false;
    logEl.closest('label').title = logOk ? '' : 'Y has values ≤ 0';

    document.getElementById('bp-highlight-input').value = '';
    document.getElementById('bp-highlight-status').classList.add('hidden');
    document.getElementById('bp-empty-state').classList.add('hidden');
    document.getElementById('bp-plot').classList.remove('hidden');
    // Show the strip before drawing so Plotly sizes to the space left, not over the strip.
    document.getElementById('bp-card-strip').classList.remove('hidden');

    _redrawTraces();

    document.getElementById('bp-highlight-section').classList.remove('hidden');
    _showCardStripPlaceholder();

    const skipped  = _data.length - pts.length;
    const censored = pts.filter(p => p.yq).length;
    _setStatus(`${pts.length} points${censored ? ` · ${censored} censored` : ''}`
             + `${skipped ? ` · ${skipped} skipped` : ''}`, 'success');
}

// Assay values like ">10000" or "<= 0.5" → { v: number, q: '>' | '<' | '' }.
function _parseCensored(raw) {
    const s = String(raw ?? '').trim();
    const m = /^([<>≤≥])=?\s*/.exec(s);
    if (!m) return { v: parseFloat(s), q: '' };
    return { v: parseFloat(s.slice(m[0].length)), q: '<≤'.includes(m[1]) ? '<' : '>' };
}

// Defaults from the Y header: IC50 (nM) → ≤ 1000 on a log axis, (µM) → ≤ 1 on a log axis, pIC50 → ≥ 6.
function _applyYHeaderDefaults(yCol) {
    const h = yCol.toLowerCase();
    let guess = null;
    if      (/(^|[^a-z])p(ic50|ec50|ac50|xc50|ki|kd)/.test(h)) guess = ['ge', 6];
    else if (/(^|[^a-z])nm([^a-z]|$)/.test(h))                 guess = ['le', 1000];
    else if (/(^|[^a-z])[uµμ]m([^a-z]|$)/.test(h))             guess = ['le', 1];
    if (!guess) return;
    document.getElementById('bp-eval-dir').value    = guess[0];
    document.getElementById('bp-eval-thresh').value = guess[1];
    if (guess[0] === 'le') document.getElementById('bp-opt-logy').checked = true;
}

function _redrawTraces() {
    if (!_currentPts.length || !_currentConfig) return;

    const { colorCol } = _currentConfig;
    const numColor     = !!(colorCol && _isNumeric(colorCol));
    const xs           = _currentPts.map(p => p.x);
    const ys           = _currentPts.map(p => p.y);

    const showMarg  = document.getElementById('bp-opt-marginals').checked;
    const showReg   = document.getElementById('bp-opt-regression').checked;
    const showDens  = document.getElementById('bp-opt-density').checked;
    const showStats = document.getElementById('bp-opt-stats').checked;
    const showEval  = document.getElementById('bp-opt-eval').checked;
    const logY      = document.getElementById('bp-opt-logy').checked;   // only enabled when all Y > 0

    const traces = [];
    if (showDens)  traces.push(_buildDensityTrace(xs, ys, logY));
    traces.push(..._buildScatterTraces(numColor));
    if (showReg) {
        const reg = _computeRegression(xs, logY ? ys.map(Math.log10) : ys);
        if (reg) traces.push(_buildRegressionTrace(xs, reg, logY));
    }
    if (showMarg) {
        traces.push(_buildTopHistogram(xs));
        traces.push(logY ? _buildLogRightHistogram(ys) : _buildRightHistogram(ys));
    }

    const stats  = (showStats || showEval) ? _getStats(showStats, showEval, logY) : null;
    const layout = _buildLayout(numColor, showMarg, showReg, stats, logY);
    const plotEl = document.getElementById('bp-plot');

    if (_initted) {
        Plotly.react(plotEl, traces, layout, PLOTLY_CFG);
    } else {
        Plotly.newPlot(plotEl, traces, layout, PLOTLY_CFG);
        _attachPlotEvents();
        _initted = true;
    }
}

// ================================================================
// Trace builders
// ================================================================

function _buildScatterTraces(numColor) {
    const { colorCol } = _currentConfig;
    const normal = _currentPts.filter(p => !_highlightIndices.has(p.idx));
    const hilite = _currentPts.filter(p =>  _highlightIndices.has(p.idx));
    const traces = [];

    if (numColor || !colorCol) {
        traces.push(_buildSingleTrace(normal, numColor));
    } else {
        traces.push(..._buildCatTraces(normal));
    }
    if (hilite.length) traces.push(_buildHighlightTrace(hilite));
    return traces;
}

function _buildSingleTrace(pts, numColor) {
    const { colorCol } = _currentConfig;
    return {
        x: pts.map(p => p.x), y: pts.map(p => p.y),
        mode: 'markers', type: 'scatter',
        marker: {
            size: 7, opacity: 0.8,
            symbol:     pts.map(_markerSymbol),
            color:      numColor ? pts.map(p => parseFloat(p.color)) : '#35583d',
            colorscale: numColor ? COLORSCALE : undefined,
            showscale:  numColor,
            colorbar: numColor ? {
                thickness: 14, len: 0.55,
                title: { text: colorCol, font: { size: 10 }, side: 'right' },
            } : undefined,
            line: { width: 0.5, color: 'rgba(0,0,0,0.18)' },
        },
        text:          pts.map(p => _hoverText(p)),
        hovertemplate: '%{text}<extra></extra>',
        customdata:    pts.map(p => p.idx),
        showlegend:    false,
        name:          'data',
        xaxis: 'x', yaxis: 'y',
    };
}

function _buildCatTraces(pts) {
    const { colorCol } = _currentConfig;
    const uniq     = [...new Set(pts.map(p => String(p.color ?? '')))].sort();
    const colorMap = Object.fromEntries(
        uniq.map((v, i) => [v, CAT_COLORS[i % CAT_COLORS.length]]));

    return uniq.map(cat => {
        const sub = pts.filter(p => String(p.color ?? '') === cat);
        return {
            x: sub.map(p => p.x), y: sub.map(p => p.y),
            mode: 'markers', type: 'scatter',
            name: cat || '(blank)',
            marker: { size: 7, opacity: 0.8, color: colorMap[cat], symbol: sub.map(_markerSymbol),
                      line: { width: 0.5, color: 'rgba(0,0,0,0.18)' } },
            text:          sub.map(p => _hoverText(p)),
            hovertemplate: '%{text}<extra></extra>',
            customdata:    sub.map(p => p.idx),
            xaxis: 'x', yaxis: 'y',
        };
    });
}

// Censored Y plotted at its limit, pointing the way the true value lies.
function _markerSymbol(p) {
    return p.yq === '>' ? 'triangle-up' : p.yq === '<' ? 'triangle-down' : 'circle';
}

function _buildHighlightTrace(pts) {
    return {
        x: pts.map(p => p.x), y: pts.map(p => p.y),
        mode: 'markers', type: 'scatter', name: 'highlighted',
        marker: { size: 13, symbol: 'star', color: '#FCC400',
                  line: { color: '#1a1a1a', width: 1.5 }, opacity: 1 },
        text:          pts.map(p => _hoverText(p)),
        hovertemplate: '%{text}<extra></extra>',
        customdata:    pts.map(p => p.idx),
        xaxis: 'x', yaxis: 'y',
    };
}

// Plotly bins histograms linearly even on a log axis, so log Y bins log10(y) itself
// and contours the grid; the grid is evenly spaced on screen once mapped back with 10^.
function _buildDensityTrace(xs, ys, logY) {
    const grid = logY ? _grid2d(xs, ys.map(Math.log10)) : null;
    return {
        ...(grid ? { type: 'contour', x: grid.xc, y: grid.yc.map(v => 10 ** v), z: grid.z }
                 : { type: 'histogram2dcontour', x: xs, y: ys }),
        xaxis: 'x', yaxis: 'y',
        colorscale: [
            [0,   'rgba(0,0,0,0)'],
            [0.3, 'rgba(103,130,107,0.08)'],
            [1,   'rgba(53,88,61,0.35)'],
        ],
        showscale: false, reversescale: false,
        ncontours: 10,
        contours: { showlines: false, coloring: 'fill' },
        hoverinfo: 'skip', showlegend: false,
    };
}

function _grid2d(xs, ys) {
    const nb = Math.max(8, Math.min(40, Math.round(Math.sqrt(xs.length))));
    const x0 = Math.min(...xs), y0 = Math.min(...ys);
    const wx = (Math.max(...xs) - x0) / nb || 1, wy = (Math.max(...ys) - y0) / nb || 1;
    const z  = Array.from({ length: nb }, () => new Array(nb).fill(0));
    for (let i = 0; i < xs.length; i++) {
        z[Math.min(nb - 1, Math.floor((ys[i] - y0) / wy))][Math.min(nb - 1, Math.floor((xs[i] - x0) / wx))]++;
    }
    const centers = (lo, w) => Array.from({ length: nb }, (_, i) => lo + (i + 0.5) * w);
    return { xc: centers(x0, wx), yc: centers(y0, wy), z };
}

// On log Y the fit is log10(y) = a + b·x, a straight line on screen.
function _buildRegressionTrace(xs, reg, logY) {
    const { slope, intercept, r2 } = reg;
    const xMin = Math.min(...xs), xMax = Math.max(...xs);
    const yAt  = x => (logY ? 10 ** (slope * x + intercept) : slope * x + intercept);
    return {
        x: [xMin, xMax],
        y: [yAt(xMin), yAt(xMax)],
        mode: 'lines', type: 'scatter',
        name: `Trend  R²${logY ? ' (log₁₀ Y)' : ''} = ${r2.toFixed(3)}`,
        line: { color: '#e05c5c', width: 2, dash: 'dot' },
        hoverinfo: 'skip',
        showlegend: true,
        xaxis: 'x', yaxis: 'y',
    };
}

function _buildTopHistogram(xs) {
    return {
        type: 'histogram', x: xs,
        xaxis: 'x', yaxis: 'y2',
        marker: { color: '#67826b', opacity: 0.55 },
        showlegend: false, hoverinfo: 'skip',
        nbinsx: Math.min(50, Math.max(10, Math.round(xs.length / 8))),
    };
}

function _buildRightHistogram(ys) {
    return {
        type: 'histogram', y: ys,
        xaxis: 'x2', yaxis: 'y',
        marker: { color: '#67826b', opacity: 0.55 },
        showlegend: false, hoverinfo: 'skip',
        nbinsy: Math.min(50, Math.max(10, Math.round(ys.length / 8))),
    };
}

// Log-Y version of the right marginal: bin log10(y) and draw the filled step outline,
// since a histogram trace would bin linearly.
function _buildLogRightHistogram(ys) {
    const ls = ys.map(Math.log10);
    const nb = Math.min(50, Math.max(10, Math.round(ys.length / 8)));
    const lo = Math.min(...ls);
    const w  = (Math.max(...ls) - lo) / nb || 1;
    const counts = new Array(nb).fill(0);
    for (const l of ls) counts[Math.min(nb - 1, Math.floor((l - lo) / w))]++;

    const x = [0], y = [10 ** lo];
    counts.forEach((c, i) => { x.push(c, c); y.push(10 ** (lo + i * w), 10 ** (lo + (i + 1) * w)); });
    x.push(0); y.push(10 ** (lo + nb * w));
    return {
        type: 'scatter', mode: 'lines', x, y, fill: 'toself',
        xaxis: 'x2', yaxis: 'y',
        line: { color: '#67826b', width: 0.5 }, fillcolor: 'rgba(103,130,107,0.55)',
        showlegend: false, hoverinfo: 'skip',
    };
}

// ================================================================
// Layout
// ================================================================

function _buildLayout(numColor, showMarg, showReg, stats, logY) {
    const { xCol, yCol, colorCol } = _currentConfig;
    const showLegend = (!numColor && !!colorCol) || showReg;
    const rNoMarg    = numColor && colorCol ? 80 : 20;

    const shared = {
        annotations: stats ? [_buildStatsAnnotation(stats)] : [],
        paper_bgcolor: 'rgba(0,0,0,0)',
        plot_bgcolor:  '#f7f5f1',
        font:  { family: 'system-ui, sans-serif', color: '#1a1a1a', size: 12 },
        showlegend: showLegend,
        legend: { font: { size: 11 }, bgcolor: 'rgba(255,255,255,0.85)',
                  bordercolor: '#ddd', borderwidth: 1 },
        dragmode: 'zoom',
    };

    const axisStyle = { showgrid: true, gridcolor: '#ddd9d2', zeroline: false };
    const yType     = logY ? 'log' : 'linear';

    if (showMarg) {
        return {
            ...shared,
            xaxis:  { ...axisStyle, domain: [0, 0.78], title: { text: xCol } },
            yaxis:  { ...axisStyle, type: yType, domain: [0, 0.78], title: { text: yCol } },
            yaxis2: { domain: [0.82, 1.0], showticklabels: false, showgrid: false, zeroline: false },
            xaxis2: { domain: [0.82, 1.0], showticklabels: false, showgrid: false, zeroline: false },
            margin: { t: 8, r: 8, b: 56, l: 64 },
        };
    }

    return {
        ...shared,
        xaxis:  { ...axisStyle, title: { text: xCol } },
        yaxis:  { ...axisStyle, type: yType, title: { text: yCol } },
        margin: { t: 20, r: rNoMarg, b: 56, l: 64 },
    };
}

function _buildStatsAnnotation(stats) {
    const fmt   = v => (v == null ? 'n/a' : v.toFixed(3));
    const ci    = c => (c ? `  [${c[0].toFixed(2)}, ${c[1].toFixed(2)}]` : '');
    const pct   = v => `${Math.round(v * 100)}%`;
    const muted = s => `<span style="color:#9a9690">${s}</span>`;
    const lines = [];

    if ('spearman' in stats) {
        lines.push(`Pearson r${stats.pearsonLog ? ' (log₁₀ Y)' : ''} = ${fmt(stats.pearson)}`);
        lines.push(`Spearman ρ = ${fmt(stats.spearman)}${ci(stats.spearmanCI)}`);
        if (stats.kendall != null) lines.push(`Kendall τ = ${fmt(stats.kendall)}`);
    }
    lines.push(`n = ${stats.n}${stats.nCens ? `  (${stats.nCens} censored, at limit)` : ''}`);

    const c = stats.cls;
    if (c) {
        lines.push('');
        if (c.dir) {
            lines.push(`<b>Active: Y ${c.dir === 'le' ? '≤' : '≥'} ${c.thr}</b>  ·  ${c.n1}/${c.n} (${pct(c.base)})`);
            if (c.nAmbig) lines.push(muted(`${c.nAmbig} censored across the cutoff, excluded`));
        }
        if (c.error) {
            lines.push(muted(c.error));
        } else {
            lines.push(`ROC AUC = ${fmt(c.auc)}${ci(c.aucCI)}`);
            lines.push(`PR AUC = ${fmt(c.ap)}  ${muted(`(random ${c.base.toFixed(2)})`)}`);
            for (const t of c.top) {
                lines.push(`Top ${Math.round(t.f * 100)}%: ${t.hits}/${t.k} active (${pct(t.prec)})  ·  EF ${t.ef.toFixed(1)}×`);
            }
        }
    }
    if (stats.spearmanCI || stats.cls?.aucCI) lines.push(muted('[ ] 95% bootstrap CI'));

    return {
        xref: 'x domain', yref: 'y domain',
        x: stats.boxRight ? 0.98 : 0.02, y: 0.98,
        xanchor: stats.boxRight ? 'right' : 'left', yanchor: 'top',
        text: lines.join('<br>'),
        align: 'left', showarrow: false,
        font: { size: 11, color: '#1a1a1a' },
        bgcolor: 'rgba(255,255,255,0.85)',
        bordercolor: '#ddd', borderwidth: 1, borderpad: 5,
    };
}

// ================================================================
// Events + card strip
// ================================================================

function _attachPlotEvents() {
    const el = document.getElementById('bp-plot');

    el.on('plotly_selected', eventData => {
        if (!eventData?.points?.length) { _showCardStripPlaceholder(); return; }
        const indices = new Set(
            eventData.points.map(pt => pt.customdata).filter(i => i != null && i >= 0));
        const selected = _currentPts.filter(p => indices.has(p.idx));
        _renderCardStrip(selected);
    });

    el.on('plotly_deselect', () => _showCardStripPlaceholder());
}

function _showCardStripPlaceholder() {
    document.getElementById('bp-card-strip-inner').innerHTML =
        '<div class="bp-strip-placeholder">Use lasso or box select (plot toolbar) to compare points here</div>';
}

function _renderCardStrip(selected) {
    if (!selected.length) { _showCardStripPlaceholder(); return; }

    const capped = selected.slice(0, 50);
    const inner  = document.getElementById('bp-card-strip-inner');
    inner.innerHTML = capped.map(p => _buildCard(p)).join('');

    if (selected.length > 50) {
        inner.innerHTML += `<div class="bp-strip-more">+${selected.length - 50} more</div>`;
    }

    inner.querySelectorAll('.bp-copy-btn').forEach(btn =>
        btn.addEventListener('click', () =>
            navigator.clipboard.writeText(btn.dataset.smiles).catch(() => {})));
}

function _buildCard(p) {
    const { xCol, yCol, colorCol } = _currentConfig;

    let svg = '';
    if (p.smiles && _RDKit) {
        let mol;
        try {
            mol = _RDKit.get_mol(p.smiles);
            if (mol?.is_valid()) svg = mol.get_svg(160, 100);
        } catch {}
        finally { if (mol) mol.delete(); }
    }

    const label = p.label || p.smiles.slice(0, 20) || '—';

    return `<div class="bp-card">
        <div class="bp-card-header">
            <span class="bp-card-label" title="${_esc(p.label)}">${_esc(label)}</span>
            ${p.smiles ? `<button class="bp-copy-btn" data-smiles="${_esc(p.smiles)}" title="Copy SMILES">📋</button>` : ''}
        </div>
        ${svg ? `<div class="bp-card-structure">${svg}</div>` : ''}
        <div class="bp-card-scores">
            <div class="bp-score-item">
                <span class="bp-score-lbl">${_esc(xCol)}</span>
                <span class="bp-score-val">${p.x.toFixed(3)}</span>
            </div>
            <div class="bp-score-item">
                <span class="bp-score-lbl">${_esc(yCol)}</span>
                <span class="bp-score-val">${_esc(p.yq)}${p.y.toFixed(3)}</span>
            </div>
            ${colorCol ? `<div class="bp-score-item bp-score-wide">
                <span class="bp-score-lbl">${_esc(colorCol)}</span>
                <span class="bp-score-val">${_esc(String(p.color ?? '—'))}</span>
            </div>` : ''}
        </div>
    </div>`;
}

// ================================================================
// Highlight
// ================================================================

function _applyHighlight() {
    const lines = document.getElementById('bp-highlight-input').value
        .split('\n').map(l => l.trim()).filter(Boolean);

    _highlightIndices = new Set();
    const { labelCol, smilesCol } = _currentConfig || {};

    for (const line of lines) {
        const byLabel  = labelCol  ? _currentPts.find(p => p.label  === line) : null;
        const bySmiles = smilesCol ? _currentPts.find(p => p.smiles === line) : null;
        const match    = byLabel || bySmiles;
        if (match) _highlightIndices.add(match.idx);
    }

    _redrawTraces();

    const statusEl = document.getElementById('bp-highlight-status');
    const n        = _highlightIndices.size;
    statusEl.textContent = `Highlighted ${n} of ${lines.length}`;
    statusEl.className   = `star-status ${n > 0 ? 'success' : 'error'}`;
    statusEl.classList.remove('hidden');
}

// ================================================================
// Regression
// ================================================================

function _computeRegression(xs, ys) {
    const n = xs.length;
    if (n < 2) return null;
    const mx = xs.reduce((a, b) => a + b, 0) / n;
    const my = ys.reduce((a, b) => a + b, 0) / n;
    const num = xs.reduce((s, x, i) => s + (x - mx) * (ys[i] - my), 0);
    const den = xs.reduce((s, x)    => s + (x - mx) * (x - mx),     0);
    if (!den) return null;
    const slope     = num / den;
    const intercept = my - slope * mx;
    const ssRes     = ys.reduce((s, y, i) => s + (y - (slope * xs[i] + intercept)) ** 2, 0);
    const ssTot     = ys.reduce((s, y)    => s + (y - my) ** 2, 0);
    return { slope, intercept, r2: ssTot > 0 ? 1 - ssRes / ssTot : 0 };
}

// ================================================================
// Correlation
// ================================================================

// Bootstrap makes this slowish, so reuse it across toggles that don't change the inputs.
function _getStats(showStats, showEval, logY) {
    const ev  = showEval ? {
        dir: document.getElementById('bp-eval-dir').value,                 // 'le' | 'ge'
        thr: parseFloat(document.getElementById('bp-eval-thresh').value),
    } : null;
    const key = `${_ptsGen}|${showStats}|${logY}|${ev ? `${ev.dir}${ev.thr}` : ''}`;
    if (_statsCache.key !== key) _statsCache = { key, stats: _computeStats(showStats, ev, logY) };
    return _statsCache.stats;
}

function _computeStats(showStats, ev, logY) {
    const xs  = _currentPts.map(p => p.x);
    const ys  = _currentPts.map(p => p.y);
    const out = { n: xs.length, nCens: _currentPts.filter(p => p.yq).length };

    if (showStats) {
        // Pearson follows what's on screen (log Y), and raw IC50 spans orders of magnitude,
        // so a lower-is-better potency gets log10 too.
        out.pearsonLog = (logY || ev?.dir === 'le') && ys.every(y => y > 0);
        out.pearson    = _pearson(xs, out.pearsonLog ? ys.map(Math.log10) : ys);
        out.spearman   = _spearman(xs, ys);
        out.spearmanCI = _bootstrapCI(xs.length, idx => _spearman(_pick(xs, idx), _pick(ys, idx)));
        out.kendall    = xs.length <= KENDALL_MAX_N ? _kendallTauB(xs, ys) : null;
    }
    if (ev) out.cls = _computeClassification(ev);
    // Stats box goes in the corner the trend leaves empty (top-right for score vs IC50).
    out.boxRight = (out.spearman ?? _spearman(xs, ys) ?? 0) < 0;
    return out;
}

function _spearman(xs, ys) {
    return _pearson(_rank(xs), _rank(ys));
}

// τ-b, tie-corrected (matches scipy.stats.kendalltau).
function _kendallTauB(xs, ys) {
    let conc = 0, disc = 0, tieX = 0, tieY = 0;
    for (let i = 0; i < xs.length; i++) {
        for (let j = i + 1; j < xs.length; j++) {
            const dx = Math.sign(xs[i] - xs[j]), dy = Math.sign(ys[i] - ys[j]);
            if (!dx && !dy) continue;
            if (!dx) tieX++;
            else if (!dy) tieY++;
            else if (dx === dy) conc++;
            else disc++;
        }
    }
    const den = Math.sqrt((conc + disc + tieX) * (conc + disc + tieY));
    return den > 0 ? (conc - disc) / den : null;
}

// ================================================================
// Classification (X = score, higher = predicted active; Y = assay value)
// ================================================================

function _computeClassification({ dir, thr }) {
    if (!Number.isFinite(thr)) return { error: 'Set an activity cutoff for AUC / top-k%' };

    const xs = [], act = [];
    for (const p of _currentPts) {
        const a = _isActive(p.y, p.yq, dir, thr);
        if (a === null) continue;
        xs.push(p.x); act.push(a);
    }
    const n  = xs.length;
    const n1 = act.filter(Boolean).length;
    const res = { dir, thr, n, n1, nAmbig: _currentPts.length - n, base: n ? n1 / n : 0 };
    if (!n1)      return { ...res, error: 'No actives at this cutoff' };
    if (n1 === n) return { ...res, error: 'Every compound is active at this cutoff' };

    res.auc   = _rocAuc(xs, act);
    res.aucCI = _bootstrapCI(n, idx => _rocAuc(_pick(xs, idx), _pick(act, idx)));
    res.ap    = _avgPrecision(xs, act);
    res.top   = TOP_FRACS.map(f => _topK(xs, act, f, res.base));
    return res;
}

// true / false, or null when a censored value sits on the wrong side of the cutoff to tell.
function _isActive(y, q, dir, thr) {
    const le   = dir === 'le';
    const good = le ? y <= thr : y >= thr;
    if (!q) return good;
    if ((q === '>') === le) return (le ? y >= thr : y <= thr) ? false : null;   // bound on the inactive side
    return good ? true : null;
}

// Mann-Whitney form, tied scores count half (matches sklearn roc_auc_score).
function _rocAuc(xs, act) {
    const r = _rank(xs);
    let n1 = 0, rankSum = 0;
    for (let i = 0; i < xs.length; i++) if (act[i]) { n1++; rankSum += r[i]; }
    const n0 = xs.length - n1;
    if (!n1 || !n0) return null;
    return (rankSum - n1 * (n1 + 1) / 2) / (n1 * n0);
}

// Step-wise PR AUC; tied scores form one threshold (matches sklearn average_precision_score).
function _avgPrecision(xs, act) {
    const order = xs.map((_, i) => i).sort((a, b) => xs[b] - xs[a]);
    const n1    = act.filter(Boolean).length;
    let tp = 0, fp = 0, prevRecall = 0, ap = 0;
    for (let i = 0; i < order.length; i++) {
        if (act[order[i]]) tp++; else fp++;
        if (i + 1 < order.length && xs[order[i + 1]] === xs[order[i]]) continue;
        const recall = tp / n1;
        ap += (recall - prevRecall) * tp / (tp + fp);
        prevRecall = recall;
    }
    return ap;
}

// Precision in the top ⌈f·n⌉ scores; ties at the cut fall back to file order.
function _topK(xs, act, f, base) {
    const k     = Math.max(1, Math.ceil(f * xs.length));
    const order = xs.map((_, i) => i).sort((a, b) => xs[b] - xs[a] || a - b);
    const hits  = order.slice(0, k).filter(i => act[i]).length;
    return { f, k, hits, prec: hits / k, ef: hits / k / base };
}

// Percentile bootstrap 95% CI, seeded so the interval doesn't jitter between redraws.
function _bootstrapCI(n, stat) {
    if (n < 5 || n > BOOT_MAX_N) return null;
    const reps = Math.max(200, Math.min(1000, Math.floor(1e6 / n)));
    const rand = _mulberry32(1);
    const idx  = new Array(n);
    const vals = [];
    for (let b = 0; b < reps; b++) {
        for (let i = 0; i < n; i++) idx[i] = Math.floor(rand() * n);
        const v = stat(idx);
        if (v != null) vals.push(v);     // e.g. a resample with no actives
    }
    if (vals.length < reps / 2) return null;
    vals.sort((a, b) => a - b);
    const q = p => vals[Math.min(vals.length - 1, Math.floor(p * vals.length))];
    return [q(0.025), q(0.975)];
}

function _mulberry32(seed) {
    return () => {
        seed = (seed + 0x6D2B79F5) | 0;
        let t = Math.imul(seed ^ (seed >>> 15), 1 | seed);
        t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
        return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
    };
}

function _pick(arr, idx) {
    return idx.map(i => arr[i]);
}

function _pearson(xs, ys) {
    const n = xs.length;
    if (n < 2) return null;
    const mx = xs.reduce((a, b) => a + b, 0) / n;
    const my = ys.reduce((a, b) => a + b, 0) / n;
    let num = 0, dx = 0, dy = 0;
    for (let i = 0; i < n; i++) {
        const a = xs[i] - mx, b = ys[i] - my;
        num += a * b; dx += a * a; dy += b * b;
    }
    const den = Math.sqrt(dx * dy);
    return den > 0 ? num / den : null;   // null when a variable is constant
}

// Fractional ranks, ties averaged (so Spearman matches scipy's tie handling).
function _rank(vals) {
    const order  = vals.map((v, i) => i).sort((a, b) => vals[a] - vals[b]);
    const ranks  = new Array(vals.length);
    let i = 0;
    while (i < order.length) {
        let j = i;
        while (j + 1 < order.length && vals[order[j + 1]] === vals[order[i]]) j++;
        const avg = (i + j) / 2 + 1;                 // 1-based average rank
        for (let k = i; k <= j; k++) ranks[order[k]] = avg;
        i = j + 1;
    }
    return ranks;
}

// ================================================================
// Helpers
// ================================================================

function _hoverText(p) {
    const { xCol, yCol, colorCol } = _currentConfig;
    let t = '';
    if (p.label) t += `<b>${_esc(p.label)}</b><br>`;
    t += `${_esc(xCol)}: ${p.x}<br>${_esc(yCol)}: ${_esc(p.yq)}${p.y}`;
    if (colorCol && p.color != null) t += `<br>${_esc(colorCol)}: ${_esc(String(p.color))}`;
    return t;
}

function _setStatus(msg, type) {
    const el = document.getElementById('bp-status');
    el.textContent = msg;
    el.className   = type ? `star-status ${type}` : 'star-status';
    el.classList.toggle('hidden', !msg);
}

function _esc(str) {
    return String(str)
        .replace(/&/g, '&amp;').replace(/</g, '&lt;')
        .replace(/>/g, '&gt;').replace(/"/g, '&quot;');
}
