/**
 * boltzplot.js
 * General-purpose scatter plot — upload CSV/TSV, pick X/Y/colour/label.
 */

let _data       = [];
let _cols       = [];
let _initted    = false;
let _scriptsOk  = false;

const COLORSCALE = [[0, '#f0ede8'], [0.5, '#67826b'], [1, '#35583d']];
const CAT_COLORS = [
    '#35583d','#5078D2','#e05c5c','#FCC400','#7b4daa',
    '#2eaaa0','#e07b2e','#4daa7b','#aa4d7b','#2e6ae0',
    '#aaa04d','#4daaaa','#e04d91','#8b5e3c',
];

const CDN_SCRIPTS = [
    'https://cdn.jsdelivr.net/npm/papaparse@5.4.1/papaparse.min.js',
    'https://cdn.jsdelivr.net/npm/plotly.js-dist@2.27.0/plotly.min.js',
];

// ----------------------------------------------------------------
// Init
// ----------------------------------------------------------------

export async function initBoltzPlot() {
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

// ----------------------------------------------------------------
// File upload
// ----------------------------------------------------------------

function _handleFile(file) {
    _setStatus('Parsing…', 'loading');
    _data = []; _cols = [];
    document.getElementById('bp-column-config').classList.add('hidden');

    const plotEl = document.getElementById('bp-plot');
    if (_initted) { try { Plotly.purge(plotEl); } catch {} _initted = false; }
    plotEl.classList.add('hidden');
    document.getElementById('bp-empty-state').classList.remove('hidden');

    const isTab = file.name.toLowerCase().endsWith('.tsv');

    Papa.parse(file, {
        header: true,
        skipEmptyLines: true,
        delimiter: isTab ? '\t' : '',
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
    document.getElementById('bp-x-col').innerHTML     = opts;
    document.getElementById('bp-y-col').innerHTML     = opts;
    document.getElementById('bp-color-col').innerHTML = none + opts;
    document.getElementById('bp-label-col').innerHTML = none + opts;
}

function _autoDetect() {
    const numCols = _cols.filter(_isNumeric);

    _sel('bp-x-col',     numCols[0] || _cols[0] || '');
    _sel('bp-y-col',     numCols[1] || _cols[1] || '');

    // Colour: 3rd numeric, or column matching score/activity/type/class
    const colorKW  = ['score','value','class','type','category','activity','group','cluster'];
    const colorHit = numCols[2] || _cols.find(c => colorKW.some(k => c.toLowerCase().includes(k)));
    if (colorHit) _sel('bp-color-col', colorHit);

    // Label: id/name column
    const labelKW  = ['id','name','label','title','compound','molecule','smiles'];
    const labelHit = _cols.find(c => labelKW.some(k => c.toLowerCase().includes(k)));
    _sel('bp-label-col', labelHit || _cols[0] || '');
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

// ----------------------------------------------------------------
// Plot generation
// ----------------------------------------------------------------

function _generatePlot() {
    const xCol     = document.getElementById('bp-x-col').value;
    const yCol     = document.getElementById('bp-y-col').value;
    const colorCol = document.getElementById('bp-color-col').value;
    const labelCol = document.getElementById('bp-label-col').value;

    if (!xCol || !yCol) { _setStatus('Select X and Y columns', 'error'); return; }

    // Build point arrays
    const pts = [];
    for (const row of _data) {
        const x = parseFloat(row[xCol]);
        const y = parseFloat(row[yCol]);
        if (isNaN(x) || isNaN(y)) continue;
        pts.push({ x, y,
            color:  colorCol ? row[colorCol] : null,
            label:  labelCol ? String(row[labelCol] ?? '') : '',
        });
    }

    if (!pts.length) { _setStatus('No numeric rows in X/Y columns', 'error'); return; }

    const numericColor = colorCol && _isNumeric(colorCol);
    const traces = numericColor || !colorCol
        ? _buildSingleTrace(pts, xCol, yCol, colorCol, numericColor)
        : _buildCatTraces(pts, colorCol);

    const layout = {
        paper_bgcolor: 'rgba(0,0,0,0)',
        plot_bgcolor:  '#f7f5f1',
        margin: { t: 20, r: colorCol && numericColor ? 80 : 20, b: 56, l: 64 },
        font:   { family: 'system-ui, sans-serif', color: '#1a1a1a', size: 12 },
        xaxis:  { title: { text: xCol }, showgrid: true, gridcolor: '#ddd9d2', zeroline: false },
        yaxis:  { title: { text: yCol }, showgrid: true, gridcolor: '#ddd9d2', zeroline: false },
        showlegend: !numericColor && !!colorCol,
        legend: { font: { size: 11 }, bgcolor: 'rgba(255,255,255,0.85)',
                  bordercolor: '#ddd', borderwidth: 1, tracegroupgap: 2 },
        dragmode: 'zoom',
    };

    const config = {
        responsive: true, displayModeBar: true, displaylogo: false,
        modeBarButtonsToRemove: ['sendDataToCloud'],
        toImageButtonOptions: { format: 'png', filename: 'boltzplot', height: 900, width: 1400, scale: 2 },
    };

    const plotEl = document.getElementById('bp-plot');
    document.getElementById('bp-empty-state').classList.add('hidden');
    plotEl.classList.remove('hidden');

    if (_initted) {
        Plotly.react(plotEl, traces, layout, config);
    } else {
        Plotly.newPlot(plotEl, traces, layout, config);
        _initted = true;
    }

    _setStatus(`${pts.length} points plotted`, 'success');
}

function _buildSingleTrace(pts, xCol, yCol, colorCol, numericColor) {
    const xs = pts.map(p => p.x);
    const ys = pts.map(p => p.y);
    const colVals = pts.map(p => numericColor ? parseFloat(p.color) : null);

    return [{
        x: xs, y: ys, mode: 'markers', type: 'scatter',
        marker: {
            size: 7, opacity: 0.8,
            color: numericColor ? colVals : '#35583d',
            ...(numericColor ? {
                colorscale: COLORSCALE, showscale: true,
                colorbar: { thickness: 14, len: 0.55,
                    title: { text: colorCol, font: { size: 10 }, side: 'right' } },
            } : {}),
            line: { width: 0.5, color: 'rgba(0,0,0,0.18)' },
        },
        text:          pts.map(p => _hoverText(p, xCol, yCol, colorCol)),
        hovertemplate: '%{text}<extra></extra>',
    }];
}

function _buildCatTraces(pts, colorCol) {
    const uniq     = [...new Set(pts.map(p => p.color))].sort();
    const colorMap = Object.fromEntries(uniq.map((v, i) => [v, CAT_COLORS[i % CAT_COLORS.length]]));

    return uniq.map(cat => {
        const sub = pts.filter(p => p.color === cat);
        return {
            x: sub.map(p => p.x), y: sub.map(p => p.y),
            mode: 'markers', type: 'scatter', name: String(cat),
            marker: { size: 7, opacity: 0.8, color: colorMap[cat],
                      line: { width: 0.5, color: 'rgba(0,0,0,0.18)' } },
            text:          sub.map(p => _hoverText(p, null, null, colorCol)),
            hovertemplate: '%{text}<extra></extra>',
        };
    });
}

function _hoverText(pt, xCol, yCol, colorCol) {
    let t = '';
    if (pt.label) t += `<b>${_esc(pt.label)}</b><br>`;
    if (xCol)     t += `${_esc(xCol)}: ${pt.x}<br>`;
    if (yCol)     t += `${_esc(yCol)}: ${pt.y}`;
    if (colorCol && pt.color != null) t += `<br>${_esc(colorCol)}: ${_esc(String(pt.color))}`;
    return t;
}

// ----------------------------------------------------------------
// Helpers
// ----------------------------------------------------------------

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
