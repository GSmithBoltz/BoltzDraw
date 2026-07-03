/**
 * adme.js
 *
 * ADME section of the Calculate tab. Lets the user tick molecules from the
 * canvas, paste a Boltz API key, and run them against the Boltz cloud ADME
 * model via the local Flask backend (POST /adme). Renders per-molecule
 * logD / permeability / solubility tiles.
 *
 * Backend availability gating (disabling the key input + Run button when the
 * local server isn't running) is owned by app.js (_updateAdmeGate); this
 * module only renders the molecule list and results.
 */

// ----------------------------------------------------------------
// Public API
// ----------------------------------------------------------------

/**
 * Render the checkbox list of molecules into #adme-mol-list.
 * Called whenever the Calculate view is shown or molecules change.
 *
 * @param {Array} molecules - current molecule array from app state
 */
export function updateAdmeMolecules(molecules) {
    const list   = document.getElementById('adme-mol-list');
    const runBtn = document.getElementById('run-adme-btn');

    if (!molecules.length) {
        list.innerHTML = '<div class="adme-empty">Draw a molecule in the editor</div>';
        runBtn.disabled = true;
        return;
    }

    list.innerHTML = molecules.map((mol, i) => {
        if (mol.error) return '';
        const id      = `Mol ${i + 1}`;
        const escaped = _escapeHtml(mol.smiles);
        const thumb   = mol.svg ? `<div class="adme-mol-thumb">${mol.svg}</div>` : '';
        return `
        <label class="adme-mol-row">
            <input type="checkbox" data-idx="${i}" checked>
            ${thumb}
            <div class="adme-mol-meta">
                <span class="adme-mol-id">${id}</span>
                <span class="adme-mol-smiles" title="${escaped}">${escaped}</span>
            </div>
        </label>`;
    }).join('');

    // Baseline enable; app.js _updateAdmeGate() refines this for backend state.
    runBtn.disabled = false;
}

/**
 * Run ADME on the ticked molecules.
 *
 * ADME predictions can take several minutes, so this submits the batch
 * (POST /adme/start) and then polls (POST /adme/poll) for status rather than
 * holding one long request open. A monotonically increasing run sequence lets
 * a fresh click supersede an in-flight poll loop.
 *
 * Wired to the Run ADME button in app.js.
 *
 * @param {Array}  molecules - current molecule array from app state
 * @param {string} apiKey    - Boltz API key pasted by the user
 */
const ADME_POLL_INTERVAL_MS = 4000;
const ADME_CLIENT_MAX_MS    = 30 * 60 * 1000;   // stop polling after 30 min

let _admeRunSeq = 0;   // bumped each runAdme(); guards against stale poll loops

export async function runAdme(molecules, apiKey) {
    const resultsDiv = document.getElementById('adme-results');
    const runBtn     = document.getElementById('run-adme-btn');
    const mySeq      = ++_admeRunSeq;

    // Collect ticked molecules (read live, at click time).
    const checked = [...document.querySelectorAll('#adme-mol-list input[type="checkbox"]:checked')];
    const selected = checked
        .map(cb => parseInt(cb.dataset.idx, 10))
        .filter(i => molecules[i] && molecules[i].smiles)
        .map(i => ({ id: `Mol ${i + 1}`, smiles: molecules[i].smiles }));

    if (!apiKey) {
        resultsDiv.innerHTML = '<div class="pka-error">Enter a Boltz API key.</div>';
        return;
    }
    if (!selected.length) {
        resultsDiv.innerHTML = '<div class="pka-error">Select at least one molecule.</div>';
        return;
    }

    runBtn.disabled = true;
    runBtn.textContent = 'Running…';
    resultsDiv.innerHTML = '<div class="pka-loading">Submitting…</div>';

    try {
        // 1) Submit the batch and get a prediction id.
        const startData = await _postJson('/adme/start', { api_key: apiKey, molecules: selected });
        const predictionId = startData.prediction_id;

        // 2) Poll for status until terminal (or the client cap is hit).
        const started = Date.now();
        while (true) {
            if (mySeq !== _admeRunSeq) return;   // superseded by a newer run

            const elapsed = Date.now() - started;
            resultsDiv.innerHTML =
                `<div class="pka-loading">Running ADME on ${selected.length} molecule${selected.length > 1 ? 's' : ''}… (${_mmss(elapsed)})</div>`;
            runBtn.textContent = `Running… ${_mmss(elapsed)}`;

            if (elapsed > ADME_CLIENT_MAX_MS) {
                resultsDiv.innerHTML =
                    `<div class="pka-error">Still running after ${_mmss(elapsed)} (prediction ${predictionId}). Try Run ADME again later.</div>`;
                return;
            }

            await _sleep(ADME_POLL_INTERVAL_MS);
            if (mySeq !== _admeRunSeq) return;

            const poll = await _postJson('/adme/poll', {
                api_key: apiKey, prediction_id: predictionId, molecules: selected,
            });

            if (poll.status === 'succeeded') {
                _displayAdmeResults(poll.results || []);
                return;
            }
            // pending | running → loop and poll again
        }
    } catch (err) {
        if (mySeq === _admeRunSeq) {
            resultsDiv.innerHTML = `<div class="pka-error">Error: ${_escapeHtml(err.message)}</div>`;
        }
    } finally {
        if (mySeq === _admeRunSeq) {
            runBtn.disabled = false;
            runBtn.textContent = 'Run ADME';
        }
    }
}

// POST JSON and return the parsed body, throwing the server's error message
// (or an HTTP status) on a non-2xx response.
async function _postJson(url, body) {
    const response = await fetch(url, {
        method:  'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(body),
    });
    if (!response.ok) {
        let msg = `Server error ${response.status}`;
        try { msg = (await response.json()).error || msg; } catch {}
        throw new Error(msg);
    }
    return response.json();
}

function _sleep(ms) {
    return new Promise(r => setTimeout(r, ms));
}

function _mmss(ms) {
    const total = Math.floor(ms / 1000);
    const m = Math.floor(total / 60);
    const s = total % 60;
    return `${m}:${String(s).padStart(2, '0')}`;
}

// ----------------------------------------------------------------
// Private helpers
// ----------------------------------------------------------------

function _displayAdmeResults(results) {
    const resultsDiv = document.getElementById('adme-results');

    if (!results.length) {
        resultsDiv.innerHTML = '<div class="pka-error">No results returned.</div>';
        return;
    }

    resultsDiv.innerHTML = results.map((r) => {
        const id = _escapeHtml(r.id || '');
        if (r.status !== 'ok') {
            return `
            <div class="adme-result-row error">
                <span class="adme-result-id">${id}</span>
                <span class="adme-result-err">${_escapeHtml(r.error || 'failed')}</span>
            </div>`;
        }
        return `
        <div class="adme-result-row">
            <span class="adme-result-id">${id}</span>
            <div class="adme-result-tiles">
                ${_tile('LogD', _fmt(r.logd))}
                ${_tile('Perm', _fmt(r.permeability))}
                ${_solTile(r.solubility)}
            </div>
        </div>`;
    }).join('');
}

function _tile(label, valueHtml, cls = '') {
    return `
    <div class="pka-result-item">
        <div class="pka-result-label">${label}</div>
        <div class="pka-result-value${cls ? ' ' + cls : ''}">${valueHtml}</div>
    </div>`;
}

// Solubility is categorical (high-confidence | medium-confidence | high-risk),
// not a number. Render a short label with a confidence colour.
const _SOL_MAP = {
    'high-confidence':   { label: 'High',   cls: 'sol-good' },
    'medium-confidence': { label: 'Medium', cls: 'sol-med'  },
    'high-risk':         { label: 'Risk',   cls: 'sol-risk' },
};

function _solTile(value) {
    const m = _SOL_MAP[value];
    if (!m) return _tile('Sol', _escapeHtml(_fmt(value)));
    return _tile('Sol', m.label, m.cls);
}

function _fmt(val) {
    if (val === null || val === undefined) return 'N/A';
    return typeof val === 'number' ? val.toFixed(2) : val;
}

function _escapeHtml(text) {
    const d = document.createElement('div');
    d.textContent = text;
    return d.innerHTML;
}
