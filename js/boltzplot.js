/**
 * boltzplot.js
 * General-purpose scatter plot: upload CSV/TSV, pick axes, colour, label, SMILES.
 * Advanced: marginal histograms · linear regression · density overlay.
 * Card strip: lasso-select points to compare structures.
 */

// ---- State ----
let _RDKit            = null;
let _data             = [];
let _cols             = [];
let _currentPts       = [];      // [{idx, x, y, color, label, smiles}, …]
let _currentConfig    = null;    // {xCol, yCol, colorCol, labelCol, smilesCol}
let _highlightIndices = new Set();
let _initted          = false;
let _scriptsOk        = false;

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
};

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
    for (const id of ['bp-opt-marginals', 'bp-opt-regression', 'bp-opt-density']) {
        document.getElementById(id)
            .addEventListener('change', () => { if (_initted) _redrawTraces(); });
    }
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
        const y = parseFloat(row[yCol]);
        if (isNaN(x) || isNaN(y)) continue;
        pts.push({
            idx:    pts.length,
            x, y,
            color:  colorCol  ? row[colorCol]                 : null,
            label:  labelCol  ? String(row[labelCol]  ?? '')  : '',
            smiles: smilesCol ? String(row[smilesCol] ?? '')  : '',
        });
    }

    if (!pts.length) { _setStatus('No numeric rows in X/Y columns', 'error'); return; }

    _currentPts    = pts;
    _currentConfig = { xCol, yCol, colorCol, labelCol, smilesCol };
    _highlightIndices = new Set();

    document.getElementById('bp-highlight-input').value = '';
    document.getElementById('bp-highlight-status').classList.add('hidden');
    document.getElementById('bp-empty-state').classList.add('hidden');
    document.getElementById('bp-plot').classList.remove('hidden');

    _redrawTraces();

    document.getElementById('bp-highlight-section').classList.remove('hidden');
    document.getElementById('bp-card-strip').classList.remove('hidden');
    _showCardStripPlaceholder();

    const skipped = _data.length - pts.length;
    _setStatus(`${pts.length} points${skipped ? ` · ${skipped} skipped` : ''}`, 'success');
}

function _redrawTraces() {
    if (!_currentPts.length || !_currentConfig) return;

    const { colorCol } = _currentConfig;
    const numColor     = !!(colorCol && _isNumeric(colorCol));
    const xs           = _currentPts.map(p => p.x);
    const ys           = _currentPts.map(p => p.y);

    const showMarg = document.getElementById('bp-opt-marginals').checked;
    const showReg  = document.getElementById('bp-opt-regression').checked;
    const showDens = document.getElementById('bp-opt-density').checked;

    const traces = [];
    if (showDens)  traces.push(_buildDensityTrace(xs, ys));
    traces.push(..._buildScatterTraces(numColor));
    if (showReg) {
        const reg = _computeRegression(xs, ys);
        if (reg) traces.push(_buildRegressionTrace(xs, reg));
    }
    if (showMarg) {
        traces.push(_buildTopHistogram(xs));
        traces.push(_buildRightHistogram(ys));
    }

    const layout = _buildLayout(numColor, showMarg, showReg);
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
            marker: { size: 7, opacity: 0.8, color: colorMap[cat],
                      line: { width: 0.5, color: 'rgba(0,0,0,0.18)' } },
            text:          sub.map(p => _hoverText(p)),
            hovertemplate: '%{text}<extra></extra>',
            customdata:    sub.map(p => p.idx),
            xaxis: 'x', yaxis: 'y',
        };
    });
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

function _buildDensityTrace(xs, ys) {
    return {
        type: 'histogram2dcontour',
        x: xs, y: ys,
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

function _buildRegressionTrace(xs, reg) {
    const { slope, intercept, r2 } = reg;
    const xMin = Math.min(...xs), xMax = Math.max(...xs);
    const sign = intercept >= 0 ? '+' : '−';
    return {
        x: [xMin, xMax],
        y: [slope * xMin + intercept, slope * xMax + intercept],
        mode: 'lines', type: 'scatter',
        name: `Trend  R² = ${r2.toFixed(3)}`,
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

// ================================================================
// Layout
// ================================================================

function _buildLayout(numColor, showMarg, showReg) {
    const { xCol, yCol, colorCol } = _currentConfig;
    const showLegend = (!numColor && !!colorCol) || showReg;
    const rNoMarg    = numColor && colorCol ? 80 : 20;

    const shared = {
        paper_bgcolor: 'rgba(0,0,0,0)',
        plot_bgcolor:  '#f7f5f1',
        font:  { family: 'system-ui, sans-serif', color: '#1a1a1a', size: 12 },
        showlegend: showLegend,
        legend: { font: { size: 11 }, bgcolor: 'rgba(255,255,255,0.85)',
                  bordercolor: '#ddd', borderwidth: 1 },
        dragmode: 'zoom',
    };

    const axisStyle = { showgrid: true, gridcolor: '#ddd9d2', zeroline: false };

    if (showMarg) {
        return {
            ...shared,
            xaxis:  { ...axisStyle, domain: [0, 0.78], title: { text: xCol } },
            yaxis:  { ...axisStyle, domain: [0, 0.78], title: { text: yCol } },
            yaxis2: { domain: [0.82, 1.0], showticklabels: false, showgrid: false, zeroline: false },
            xaxis2: { domain: [0.82, 1.0], showticklabels: false, showgrid: false, zeroline: false },
            margin: { t: 8, r: 8, b: 56, l: 64 },
        };
    }

    return {
        ...shared,
        xaxis:  { ...axisStyle, title: { text: xCol } },
        yaxis:  { ...axisStyle, title: { text: yCol } },
        margin: { t: 20, r: rNoMarg, b: 56, l: 64 },
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
                <span class="bp-score-val">${p.y.toFixed(3)}</span>
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
// Helpers
// ================================================================

function _hoverText(p) {
    const { xCol, yCol, colorCol } = _currentConfig;
    let t = '';
    if (p.label) t += `<b>${_esc(p.label)}</b><br>`;
    t += `${_esc(xCol)}: ${p.x}<br>${_esc(yCol)}: ${p.y}`;
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
