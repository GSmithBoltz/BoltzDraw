/**
 * app.js
 *
 * Entry point. Owns global app state and coordinates all modules:
 *   - Boots RDKit
 *   - Initialises Ketcher (via ketcher.js)
 *   - Calculates properties on SMILES change (via properties.js)
 *   - Handles view toggle and panel collapse
 *   - Delegates pKa tab to pka.js
 */

import { initKetcher, loadSmiles }               from './ketcher.js';
import { calculateProperties, renderMolecules } from './properties.js';
import { updatePkaSmiles, calculatePka }         from './pka.js';
import { updateAdmeMolecules, runAdme }          from './adme.js';
import { initMolstar, loadByPdbId, loadFromFile, loadFromData, applyBoltzPreset, setSurfaceColors } from './molstar.js';
import { convertPeptideToSmiles, supportedCodes } from './peptide.js';
import { fetchPdbLigands, fetchLigandsByCompIds } from './pdb.js';
import { parsePdb, parseCif, buildMolFromPdbCoords, GENERIC_LIGAND_CODES } from './fileparse.js';

// ----------------------------------------------------------------
// App state
// ----------------------------------------------------------------
let RDKit          = null;
let isRDKitReady   = false;
let isKetcherReady = false;
let currentMolecules = [];
let currentView    = 'cards';   // 'cards' | 'pka'

// True only when the local Flask backend responds to /ping
let _backendAvailable = false;

const RDKIT_CDN_BASE = 'https://cdn.jsdelivr.net/npm/@rdkit/rdkit@2025.3.4-1.0.0/dist/';

// ----------------------------------------------------------------
// Boot sequence
// ----------------------------------------------------------------

async function boot() {
    // Run RDKit, Ketcher, and backend probe in parallel.
    _initKetcher();
    _checkBackend();       // fire-and-forget — doesn't block the loading overlay
    await _initRDKit();
}

async function _checkBackend() {
    try {
        const res = await fetch('/ping', {
            signal: AbortSignal.timeout(2000),
        });
        const json = await res.json();
        _backendAvailable = res.ok && json.backend === 'boltzstar';
    } catch {
        _backendAvailable = false;
    }
    // The probe is fire-and-forget, so refresh the ADME gate once it resolves
    // (the user may already be sitting on the Calculate tab).
    _updateAdmeGate();
}

boot();

// ----------------------------------------------------------------
// RDKit
// ----------------------------------------------------------------

async function _initRDKit() {
    try {
        _setLoadingStatus('Loading RDKit...');
        await _loadScript(RDKIT_CDN_BASE + 'RDKit_minimal.js');

        _setLoadingStatus('Initializing RDKit (this may take a moment)...');

        // Wait for the module to appear on window (WASM async init)
        let attempts = 0;
        while (!window.initRDKitModule && attempts < 100) {
            await _sleep(100);
            attempts++;
        }
        if (!window.initRDKitModule) throw new Error('RDKit module not found');

        RDKit = await window.initRDKitModule({
            locateFile: (file) => RDKIT_CDN_BASE + file,
        });

        console.log('RDKit ready:', RDKit.version());
        isRDKitReady = true;
        _setLoadingStatus('Loading editor...');
        _checkReady();
    } catch (err) {
        console.error('RDKit init failed:', err);
        _setLoadingStatus('RDKit failed: ' + err.message);
        _updateStatus('RDKit failed', false);
    }
}

function _loadScript(src) {
    return new Promise((resolve, reject) => {
        const s = document.createElement('script');
        s.src = src;
        s.onload = resolve;
        s.onerror = reject;
        document.head.appendChild(s);
    });
}

function _sleep(ms) {
    return new Promise(r => setTimeout(r, ms));
}

// ----------------------------------------------------------------
// Ketcher
// ----------------------------------------------------------------

function _initKetcher() {
    // Pass our SMILES-change handler to the Ketcher module
    initKetcher(_onSmilesChange);

    // Listen for the 'ketcherReady' event dispatched by ketcher.js
    window.addEventListener('ketcherReady', () => {
        isKetcherReady = true;
        _checkReady();
    });
}

/**
 * Called by ketcher.js whenever a structure is genuinely committed.
 * Recalculates properties and re-renders the active view.
 */
function _onSmilesChange(smiles) {
    currentMolecules = calculateProperties(smiles, RDKit);

    // Always re-render whichever view is active
    if (currentView === 'cards') {
        renderMolecules(currentMolecules);
    } else {
        updatePkaSmiles(currentMolecules);
        updateAdmeMolecules(currentMolecules);
        _updateAdmeGate();
    }

    // Update export tab info strip
    _updateExportInfo(currentMolecules);

    // Update molecule count in status bar
    const n = currentMolecules.length;
    document.getElementById('molecule-count').textContent =
        `${n} molecule${n !== 1 ? 's' : ''}`;
}

// ----------------------------------------------------------------
// Ready gate
// ----------------------------------------------------------------

function _checkReady() {
    if (isRDKitReady && isKetcherReady) {
        document.getElementById('loading').classList.add('hidden');
        _updateStatus('Ready', true);
    }
}

function _updateStatus(text, ready) {
    document.getElementById('status-text').textContent = text;
    document.getElementById('status-dot').classList.toggle('ready', ready);
}

function _setLoadingStatus(text) {
    document.getElementById('loading-status').textContent = text;
}

// ----------------------------------------------------------------
// Mode switching - BoltzDraw (2D) / BoltzStar (3D)
// ----------------------------------------------------------------

const MODES = {
    draw: {
        subtitle:    'Molecular Structure Editor',
        panelTitle:  'Properties',
        btnClass:    'active-draw',
        containerId: 'ketcher-container',
        btnId:       'btn-draw',
    },
    star: {
        subtitle:    '3D Structure Viewer',
        panelTitle:  'Viewer',
        btnClass:    'active-star',
        containerId: 'molstar-container',
        btnId:       'btn-star',
    },
    space: {
        subtitle:    'Chemical Space Explorer',
        panelTitle:  'Chemical Space',
        btnClass:    'active-space',
        containerId: 'boltzspace-container',
        btnId:       'btn-space',
    },
    plot: {
        subtitle:    'Custom Plot Tool',
        panelTitle:  'Custom Plot',
        btnClass:    'active-plot',
        containerId: 'boltzplot-container',
        btnId:       'btn-plot',
    },
    table: {
        subtitle:    'Structure Table Viewer',
        panelTitle:  'Table',
        btnClass:    'active-table',
        containerId: 'boltztable-container',
        btnId:       'btn-table',
    },
};

let currentMode       = 'draw';
let panelMode         = 'inspect'; // 'inspect' | 'build'
let buildView         = 'peptide'; // 'peptide' | 'pdb'
let molstarInitted    = false;
let boltzSpaceInitted = false;
let boltzPlotInitted  = false;
let boltzTableInitted = false;

async function switchMode(mode) {
    if (mode === currentMode) return;
    currentMode = mode;

    const cfg = MODES[mode];

    // Update header subtitle
    document.getElementById('app-subtitle').textContent = cfg.subtitle;

    // Swap button active classes and main containers — driven by MODES so
    // adding a new mode here never requires touching this loop.
    Object.entries(MODES).forEach(([key, c]) => {
        document.getElementById(c.btnId).className =
            'mode-btn' + (key === mode ? ` ${c.btnClass}` : '');
        document.getElementById(c.containerId).classList.toggle('hidden', key !== mode);
    });

    // Hide shared properties panel in modes that have their own right panel
    const hidePanel = mode === 'space' || mode === 'plot' || mode === 'table' || !!cfg.ownPanel;
    document.getElementById('properties-panel').classList.toggle('hidden', hidePanel);
    document.getElementById('panel-toggle').classList.toggle('hidden', hidePanel);

    // Swap panel content
    const isDraw = mode === 'draw';

    // Header: show Inspect/Build toggle in draw mode; show Viewer title in star mode
    document.getElementById('panel-mode-toggle').classList.toggle('hidden', !isDraw);
    document.getElementById('panel-header-star').classList.toggle('hidden',  isDraw);

    // Inspect sub-tabs only visible in draw+inspect mode
    document.getElementById('inspect-tabs').classList.toggle('hidden', !isDraw);

    document.getElementById('molecules-list').classList.toggle('hidden', !isDraw);
    document.getElementById('pka-container').classList.toggle('hidden',   true);   // always reset pKa
    document.getElementById('peptide-container').classList.remove('active');       // reset build tabs
    document.getElementById('pdb-container').classList.remove('active');
    document.getElementById('file-container').classList.remove('active');
    document.getElementById('export-container').classList.remove('active');        // reset export tab
    document.getElementById('build-tabs').classList.add('hidden');                 // reset build sub-tabs
    document.getElementById('star-panel').classList.toggle('hidden',     isDraw);

    // Reset to Inspect / Cards when returning to BoltzDraw
    if (isDraw) {
        _setPanelMode('inspect');
        _setDrawView('cards');
    }

    // Lazy-init BoltzSpace on first switch
    if (mode === 'space' && !boltzSpaceInitted) {
        boltzSpaceInitted = true;
        try {
            const { initBoltzSpace } = await import('./boltzspace.js');
            await initBoltzSpace(RDKit);
        } catch (err) {
            console.error('BoltzSpace init failed:', err);
        }
    }

    // Lazy-init BoltzPlot on first switch
    if (mode === 'plot' && !boltzPlotInitted) {
        boltzPlotInitted = true;
        try {
            const { initBoltzPlot } = await import('./boltzplot.js');
            await initBoltzPlot(RDKit);
        } catch (err) {
            console.error('BoltzPlot init failed:', err);
        }
    }

    // Lazy-init BoltzTable on first switch
    if (mode === 'table' && !boltzTableInitted) {
        boltzTableInitted = true;
        try {
            const { initBoltzTable } = await import('./boltztable.js');
            await initBoltzTable(RDKit, { onSendTo: sendCsvToMode });
        } catch (err) {
            console.error('BoltzTable init failed:', err);
        }
    }

    // Modes added through registerMode() run their own init on every switch
    if (cfg.onEnter) {
        try {
            await cfg.onEnter({ RDKit, molecules: currentMolecules, loadSmiles, switchMode });
        } catch (err) {
            console.error(`${mode} init failed:`, err);
        }
    }

    // Lazy-init Molstar on first switch to BoltzStar
    if (mode === 'star' && !molstarInitted) {
        molstarInitted = true;
        _updateStatus('Loading Molstar...', false);
        try {
            await initMolstar();
            _updateStatus('Ready', true);
        } catch (err) {
            console.error('Molstar init failed:', err);
            _updateStatus('Molstar failed to load', false);
        }
    }
}

/**
 * Hand a CSV string produced by one mode to another mode's uploader.
 * Switching first guarantees the target module is lazy-inited (and its CDN
 * scripts loaded) before we push data at it.
 */
async function sendCsvToMode(mode, csvText, filename = 'boltztable-subset.csv') {
    const file = new File([csvText], filename, { type: 'text/csv' });
    await switchMode(mode);
    if (mode === 'space') {
        const { handleCsvUpload } = await import('./boltzspace.js');
        handleCsvUpload(file);
    } else if (mode === 'plot') {
        const { loadCsvFile } = await import('./boltzplot.js');
        loadCsvFile(file);
    }
}

document.getElementById('btn-draw').addEventListener('click',  () => switchMode('draw'));
document.getElementById('btn-star').addEventListener('click',  () => switchMode('star'));
document.getElementById('btn-space').addEventListener('click', () => switchMode('space'));
document.getElementById('btn-plot').addEventListener('click',  () => switchMode('plot'));
document.getElementById('btn-table').addEventListener('click', () => switchMode('table'));

/**
 * Add a mode at runtime. Used by the optional, gitignored local/ folder, which
 * server.py loads when present. cfg takes the MODES fields plus `label`
 * ([brand, product] for the header button) and optionally `before` (id of the
 * button to insert ahead of), `ownPanel` and `onEnter(ctx)`.
 */
export function registerMode(key, cfg) {
    MODES[key] = cfg;
    const btn = document.createElement('button');
    btn.className = 'mode-btn';
    btn.id = cfg.btnId;
    btn.innerHTML = `<span class="btn-brand">${cfg.label[0]}</span><span class="btn-product">${cfg.label[1]}</span>`;
    btn.addEventListener('click', () => switchMode(key));
    const nav = document.querySelector('.app-switcher');
    nav.insertBefore(btn, (cfg.before && document.getElementById(cfg.before)) || null);
}


// ----------------------------------------------------------------
// View toggle (Cards / pKa) - BoltzDraw only
// ----------------------------------------------------------------

// ----------------------------------------------------------------
// Panel mode: Inspect (Cards/pKa/Export) ↔ Build (Peptide)
// ----------------------------------------------------------------

function _setPanelMode(mode) {
    panelMode = mode;
    const isInspect = mode === 'inspect';

    document.getElementById('mode-inspect').classList.toggle('active', isInspect);
    document.getElementById('mode-build').classList.toggle('active',  !isInspect);

    // Show/hide sub-tab strips
    document.getElementById('inspect-tabs').classList.toggle('hidden', !isInspect);
    document.getElementById('build-tabs').classList.toggle('hidden',    isInspect);

    if (isInspect) {
        // Hide all build content, then restore the last inspect tab
        document.getElementById('peptide-container').classList.remove('active');
        document.getElementById('pdb-container').classList.remove('active');
        document.getElementById('file-container').classList.remove('active');
        const safeView = ['cards', 'pka', 'export'].includes(currentView) ? currentView : 'cards';
        _setDrawView(safeView);
    } else {
        // Hide all inspect content, then show the current build sub-tab
        document.getElementById('molecules-list').classList.add('hidden');
        document.getElementById('pka-container').classList.remove('active');
        document.getElementById('export-container').classList.remove('active');
        _setBuildView(buildView);
    }
}

document.getElementById('mode-inspect').addEventListener('click', () => _setPanelMode('inspect'));
document.getElementById('mode-build').addEventListener('click',   () => _setPanelMode('build'));

// ----------------------------------------------------------------
// Inspect sub-tabs: Cards / pKa / Export
// ----------------------------------------------------------------

function _setDrawView(view) {
    const isCards  = view === 'cards';
    const isPka    = view === 'pka';
    const isExport = view === 'export';

    document.getElementById('view-cards').classList.toggle('active',  isCards);
    document.getElementById('view-pka').classList.toggle('active',    isPka);
    document.getElementById('view-export').classList.toggle('active', isExport);

    document.getElementById('molecules-list').classList.toggle('hidden', !isCards);
    document.getElementById('pka-container').classList.toggle('active',  isPka);
    document.getElementById('export-container').classList.toggle('active', isExport);

    if (isCards) {
        renderMolecules(currentMolecules);
    } else if (isPka) {
        updatePkaSmiles(currentMolecules);
        updateAdmeMolecules(currentMolecules);
        _updateAdmeGate();
    }

    currentView = view;
}

document.getElementById('view-cards').addEventListener('click',  () => _setDrawView('cards'));
document.getElementById('view-pka').addEventListener('click',    () => _setDrawView('pka'));
document.getElementById('view-export').addEventListener('click', () => _setDrawView('export'));

// ----------------------------------------------------------------
// Calculate tab: ADME (cloud) + pKa (local) buttons
// ----------------------------------------------------------------

// ADME needs the local backend (it proxies the pasted key to Boltz cloud),
// so gate the key input + Run button on backend availability. Mirrors the
// hydrate/PLIP gating pattern.
function _updateAdmeGate() {
    const ok = _backendAvailable;
    document.getElementById('adme-unavailable').classList.toggle('hidden', ok);
    document.getElementById('adme-api-key').disabled = !ok;
    document.getElementById('run-adme-btn').disabled = !ok || currentMolecules.length === 0;
}

document.getElementById('run-adme-btn').addEventListener('click', () => {
    const apiKey = document.getElementById('adme-api-key').value.trim();
    runAdme(currentMolecules, apiKey);
});

document.getElementById('calculate-pka-btn').addEventListener('click', () => {
    calculatePka(currentMolecules, _backendAvailable);
});

// ----------------------------------------------------------------
// Peptide tab
// ----------------------------------------------------------------

const VALID_CODES = supportedCodes();

const AA_CLASS_MAP = {
    A:'hphob', V:'hphob', I:'hphob', L:'hphob', M:'hphob', F:'hphob', W:'hphob', P:'hphob',
    a:'hphob', v:'hphob', i:'hphob', l:'hphob', m:'hphob', f:'hphob', w:'hphob', p:'hphob',
    G:'polar', S:'polar', T:'polar', C:'polar', Y:'polar', N:'polar', Q:'polar',
    g:'polar', s:'polar', t:'polar', c:'polar', y:'polar', n:'polar', q:'polar',
    R:'pos', H:'pos', K:'pos', r:'pos', h:'pos', k:'pos',
    D:'neg', E:'neg', d:'neg', e:'neg',
};

const pepSeqEl       = document.getElementById('peptide-sequence');
const pepPillsEl     = document.getElementById('peptide-pills');
const pepCountEl     = document.getElementById('peptide-count');
const pepConvertBtn  = document.getElementById('peptide-convert-btn');
const pepErrorEl     = document.getElementById('peptide-error');
const pepResultEl    = document.getElementById('peptide-result');
const pepSmilesOutEl = document.getElementById('peptide-smiles-output');
const pepCopyBtn     = document.getElementById('peptide-copy-btn');
const pepLoadBtn     = document.getElementById('peptide-load-btn');
const pepWarnEl      = document.getElementById('peptide-warning');

function _updatePeptidePills() {
    const seq      = pepSeqEl.value.replace(/\s/g, '');
    const residues = seq.split('').filter(Boolean);
    const validCount = residues.filter(r => VALID_CODES.has(r)).length;

    pepPillsEl.innerHTML = residues.map(r => {
        const cls = AA_CLASS_MAP[r];
        return `<span class="peptide-pill peptide-pill-${cls || 'bad'}" title="${r}">${r}</span>`;
    }).join('');

    pepCountEl.textContent = `${validCount} / ${residues.length} residues valid`;
    pepConvertBtn.disabled = residues.length === 0 || validCount !== residues.length;

    // Clear previous result on edit
    pepResultEl.classList.remove('active');
    pepErrorEl.classList.remove('active');
}

pepSeqEl.addEventListener('input', _updatePeptidePills);

pepConvertBtn.addEventListener('click', () => {
    const seq   = pepSeqEl.value.replace(/\s/g, '');
    const nterm = document.getElementById('peptide-nterm').value;
    const cterm = document.getElementById('peptide-cterm').value;

    pepErrorEl.classList.remove('active');
    pepResultEl.classList.remove('active');

    try {
        const { smiles, warnings } = convertPeptideToSmiles(seq, nterm, cterm);

        pepSmilesOutEl.textContent = smiles;
        pepResultEl.classList.add('active');

        if (warnings.length) {
            pepWarnEl.textContent = warnings.join(' ');
            pepWarnEl.classList.add('active');
        } else {
            pepWarnEl.classList.remove('active');
        }
    } catch (err) {
        pepErrorEl.textContent = err.message;
        pepErrorEl.classList.add('active');
    }
});

pepCopyBtn.addEventListener('click', () => {
    const smiles = pepSmilesOutEl.textContent;
    navigator.clipboard.writeText(smiles).then(() => {
        pepCopyBtn.textContent = 'Copied!';
        setTimeout(() => { pepCopyBtn.textContent = 'Copy SMILES'; }, 1800);
    });
});

pepLoadBtn.addEventListener('click', async () => {
    const smiles = pepSmilesOutEl.textContent;
    if (!smiles) return;
    pepLoadBtn.textContent = 'Loading…';
    pepLoadBtn.disabled = true;
    try {
        await loadSmiles(smiles);
        pepLoadBtn.textContent = '✓ Loaded';
        // Switch to Inspect → Cards so the user sees the structure
        _setPanelMode('inspect');
        _setDrawView('cards');
        setTimeout(() => {
            pepLoadBtn.textContent = 'Load in Editor →';
            pepLoadBtn.disabled = false;
        }, 2000);
    } catch (err) {
        pepLoadBtn.textContent = 'Failed';
        pepLoadBtn.disabled = false;
        console.error('loadSmiles failed:', err);
    }
});

// ----------------------------------------------------------------
// Build sub-tab switching: Peptide ↔ PDB
// ----------------------------------------------------------------

function _setBuildView(view) {
    buildView = view;
    ['peptide', 'pdb', 'file'].forEach(v => {
        document.getElementById(`build-tab-${v}`).classList.toggle('active', v === view);
        const containerId = v === 'peptide' ? 'peptide-container'
                          : v === 'pdb'     ? 'pdb-container'
                          :                   'file-container';
        document.getElementById(containerId).classList.toggle('active', v === view);
    });
}

document.getElementById('build-tab-peptide').addEventListener('click', () => _setBuildView('peptide'));
document.getElementById('build-tab-pdb').addEventListener('click',     () => _setBuildView('pdb'));
document.getElementById('build-tab-file').addEventListener('click',    () => _setBuildView('file'));

// ----------------------------------------------------------------
// PDB ligand importer
// ----------------------------------------------------------------

const pdbCodeInput = document.getElementById('pdb-code-input');
const pdbFetchBtn  = document.getElementById('pdb-fetch-btn');
const pdbStatusEl  = document.getElementById('pdb-status');
const pdbResultsEl = document.getElementById('pdb-results');

pdbFetchBtn.addEventListener('click', _doPdbFetch);
pdbCodeInput.addEventListener('keydown', (e) => {
    if (e.key === 'Enter') _doPdbFetch();
});

async function _doPdbFetch() {
    const code = pdbCodeInput.value.trim().toUpperCase();
    if (code.length < 3 || code.length > 4) return;

    pdbFetchBtn.disabled   = true;
    pdbResultsEl.innerHTML = '';
    pdbStatusEl.className  = 'pdb-status loading';
    pdbStatusEl.textContent = `Fetching ${code}…`;
    pdbStatusEl.classList.remove('hidden');

    try {
        const ligands = await fetchPdbLigands(code);

        if (!ligands.length) {
            pdbStatusEl.className   = 'pdb-status info';
            pdbStatusEl.textContent = `No small-molecule ligands found in ${code}.`;
            pdbFetchBtn.disabled = false;
            return;
        }

        pdbStatusEl.className   = 'pdb-status success';
        pdbStatusEl.textContent =
            `${ligands.length} ligand${ligands.length !== 1 ? 's' : ''} found in ${code}`;

        _renderLigandCards(ligands, pdbResultsEl);

    } catch (err) {
        pdbStatusEl.className   = 'pdb-status error';
        pdbStatusEl.textContent = err.message || 'Failed to fetch PDB entry.';
    }

    pdbFetchBtn.disabled = false;
}

// ----------------------------------------------------------------
// Shared ligand card renderer (used by PDB and File tabs)
// ----------------------------------------------------------------

function _renderLigandCards(ligands, resultsEl) {
    resultsEl.innerHTML = ligands.map(lig => `
        <div class="ligand-card">
            <img class="ligand-card-image"
                 src="${lig.imageUrl}"
                 alt="${lig.comp_id} structure"
                 loading="lazy"
                 onerror="this.style.display='none'">
            <div class="ligand-card-body">
                <div class="ligand-comp-id">${lig.comp_id}</div>
                <div class="ligand-name">${lig.name}</div>
                <div class="ligand-meta">
                    ${lig.formula ? `<span>${lig.formula}</span>` : ''}
                    ${lig.mw      ? `<span>${parseFloat(lig.mw).toFixed(1)} Da</span>` : ''}
                </div>
            </div>
            <button class="ligand-import-btn"
                    data-smiles="${lig.smiles.replace(/"/g, '&quot;')}">
                Import into Editor →
            </button>
        </div>
    `).join('');

    resultsEl.querySelectorAll('.ligand-import-btn').forEach(btn => {
        btn.addEventListener('click', async () => {
            const smiles = btn.dataset.smiles;
            if (!smiles) return;
            btn.textContent = 'Loading…';
            btn.disabled    = true;
            try {
                await loadSmiles(smiles);
                btn.textContent = '✓ Loaded';
                _setPanelMode('inspect');
                _setDrawView('cards');
                setTimeout(() => {
                    btn.textContent = 'Import into Editor →';
                    btn.disabled    = false;
                }, 2000);
            } catch (err) {
                console.error('Ligand import failed:', err);
                btn.textContent = 'Failed';
                btn.disabled    = false;
            }
        });
    });
}

// ----------------------------------------------------------------
// File upload ligand extractor
// ----------------------------------------------------------------

const ligFileInput  = document.getElementById('lig-file-input');
const fileUploadZone = document.getElementById('file-upload-zone');
const fileUploadText = document.getElementById('file-upload-text');
const fileStatusEl  = document.getElementById('file-status');
const fileResultsEl = document.getElementById('file-results');

// Click on zone → trigger file input
fileUploadZone.addEventListener('click', () => ligFileInput.click());

// Drag-and-drop styling
fileUploadZone.addEventListener('dragover',  (e) => { e.preventDefault(); fileUploadZone.classList.add('drag-over'); });
fileUploadZone.addEventListener('dragleave', ()  => fileUploadZone.classList.remove('drag-over'));
fileUploadZone.addEventListener('drop', (e) => {
    e.preventDefault();
    fileUploadZone.classList.remove('drag-over');
    const file = e.dataTransfer?.files?.[0];
    if (file) _processStructureFile(file);
});

ligFileInput.addEventListener('change', (e) => {
    const file = e.target.files?.[0];
    if (file) _processStructureFile(file);
    // Reset so re-uploading the same file still fires
    ligFileInput.value = '';
});

async function _processStructureFile(file) {
    const name  = file.name.toLowerCase();
    const isPdb = name.endsWith('.pdb');
    const isCif = name.endsWith('.cif') || name.endsWith('.mmcif');

    if (!isPdb && !isCif) {
        fileStatusEl.className   = 'pdb-status error';
        fileStatusEl.textContent = 'Unsupported file type. Please upload a .pdb or .cif file.';
        fileStatusEl.classList.remove('hidden');
        return;
    }

    fileUploadZone.classList.add('has-file');
    fileUploadText.textContent = file.name;

    fileStatusEl.className   = 'pdb-status loading';
    fileStatusEl.textContent = 'Parsing file…';
    fileStatusEl.classList.remove('hidden');
    fileResultsEl.innerHTML  = '';

    try {
        const text                = await file.text();
        const { compIds, smilesMap } = isPdb ? parsePdb(text) : parseCif(text);

        if (!compIds.length) {
            fileStatusEl.className   = 'pdb-status info';
            fileStatusEl.textContent = 'No small-molecule ligands found in this file.';
            return;
        }

        // Split: comp_ids whose SMILES we already have vs those that need the CCD API
        const inlineIds = compIds.filter(id =>  smilesMap[id]);
        const apiIds    = compIds.filter(id => !smilesMap[id]);

        fileStatusEl.textContent =
            apiIds.length
                ? `Found ${compIds.length} ligand${compIds.length !== 1 ? 's' : ''} — fetching data…`
                : `Found ${compIds.length} ligand${compIds.length !== 1 ? 's' : ''} — generating previews…`;

        // --- Build ligand objects from inline SMILES using RDKit for 2D SVG ---
        const inlineLigands = inlineIds.map(comp_id => {
            const smiles = smilesMap[comp_id];
            let svgDataUri = '';
            try {
                const mol = RDKit.get_mol(smiles);
                if (mol) {
                    const svgStr = mol.get_svg(220, 160);
                    svgDataUri = `data:image/svg+xml,${encodeURIComponent(svgStr)}`;
                    mol.delete();
                }
            } catch (e) { /* RDKit couldn't parse - SVG will be blank */ }
            return {
                comp_id,
                name:     comp_id,          // no name from file; CDD might add it
                formula:  '',
                mw:       null,
                smiles,
                imageUrl: svgDataUri || `https://cdn.rcsb.org/images/ccd/unlabeled/${comp_id[0].toLowerCase()}/${comp_id}.svg`,
            };
        });

        // --- Fetch remaining via RCSB CCD API ---
        const apiLigands = apiIds.length ? await fetchLigandsByCompIds(apiIds) : [];

        const allLigands = [...inlineLigands, ...apiLigands];

        if (!allLigands.length) {
            // Last resort: reconstruct bonds from 3D coordinates (e.g. generic 'LIG' codes)
            if (isPdb && compIds.length) {
                fileStatusEl.className   = 'pdb-status loading';
                fileStatusEl.textContent = 'No CCD data found — inferring structure from 3D coordinates…';

                const coordLigands = [];
                for (const compId of compIds) {
                    const molblock = buildMolFromPdbCoords(text, compId);
                    if (!molblock) continue;
                    try {
                        const mol = RDKit.get_mol(molblock);
                        if (mol) {
                            const smiles = mol.get_smiles();
                            const svgStr = mol.get_svg(220, 160);
                            mol.delete();
                            if (smiles) {
                                const label = GENERIC_LIGAND_CODES.has(compId)
                                    ? `${compId} (from coordinates)`
                                    : compId;
                                coordLigands.push({
                                    comp_id:  compId,
                                    name:     label,
                                    formula:  '',
                                    mw:       null,
                                    smiles,
                                    imageUrl: `data:image/svg+xml,${encodeURIComponent(svgStr)}`,
                                });
                            }
                        }
                    } catch (e) {
                        console.warn(`RDKit reconstruction failed for ${compId}:`, e);
                    }
                }

                if (coordLigands.length) {
                    fileStatusEl.className   = 'pdb-status success';
                    fileStatusEl.textContent =
                        `${coordLigands.length} ligand${coordLigands.length !== 1 ? 's' : ''} reconstructed from 3D coordinates`;
                    _renderLigandCards(coordLigands, fileResultsEl);
                    return;
                }
            }

            fileStatusEl.className   = 'pdb-status info';
            fileStatusEl.textContent = 'Ligands found but could not retrieve SMILES.';
            return;
        }

        fileStatusEl.className   = 'pdb-status success';
        fileStatusEl.textContent =
            `${allLigands.length} ligand${allLigands.length !== 1 ? 's' : ''} extracted from ${file.name}`;

        _renderLigandCards(allLigands, fileResultsEl);

    } catch (err) {
        console.error('File parse failed:', err);
        fileStatusEl.className   = 'pdb-status error';
        fileStatusEl.textContent = err.message || 'Failed to process file.';
    }
}

// ----------------------------------------------------------------
// Export tab
// ----------------------------------------------------------------

let selectedExportFormat = null;

function _updateExportInfo(molecules) {
    const infoEl = document.getElementById('export-molecule-info');
    if (!infoEl) return;
    if (!molecules.length) {
        infoEl.querySelector('.export-mol-name').textContent = '—';
        infoEl.querySelector('.export-mol-meta').textContent = 'Draw a molecule first';
        return;
    }
    const n   = molecules.length;
    const mol = molecules[0];
    infoEl.querySelector('.export-mol-name').textContent =
        n === 1 ? 'Mol 1' : `${n} molecules`;
    infoEl.querySelector('.export-mol-meta').textContent =
        `${mol.smiles.slice(0, 48)}${mol.smiles.length > 48 ? '…' : ''} · MW ${mol.mw?.toFixed(2) ?? '—'}`;
}

function _exportBtnLabel(format) {
    if (!format) return 'Select a format above';
    return format === 'smiles' ? 'List SMILES' : `Download ${format.toUpperCase()}`;
}

document.querySelectorAll('.export-format').forEach(card => {
    card.addEventListener('click', () => {
        document.querySelectorAll('.export-format').forEach(c => c.classList.remove('selected'));
        card.classList.add('selected');
        selectedExportFormat = card.dataset.format;

        // Collapse any open SMILES list when switching format
        document.getElementById('smiles-list-output').classList.add('hidden');

        const btn = document.getElementById('export-download-btn');
        btn.disabled = !currentMolecules.length;
        btn.textContent = currentMolecules.length
            ? _exportBtnLabel(selectedExportFormat)
            : 'Draw a molecule first';
    });
});

document.getElementById('export-download-btn').addEventListener('click', async () => {
    if (!selectedExportFormat || !currentMolecules.length) return;

    const btn     = document.getElementById('export-download-btn');
    const ketcher = document.getElementById('ketcher-frame').contentWindow?.ketcher;

    btn.disabled = true;

    try {
        switch (selectedExportFormat) {

            // ---- SMILES: inline list, no download ----
            case 'smiles': {
                btn.textContent = 'Listing…';
                const smiList = currentMolecules.map(m => m.smiles).join('\n');
                const ta = document.getElementById('smiles-list-textarea');
                ta.value = smiList;
                document.getElementById('smiles-list-output').classList.remove('hidden');
                btn.textContent = '✓ Listed';
                setTimeout(() => { btn.textContent = 'List SMILES'; btn.disabled = false; }, 2000);
                return; // no download, exit early
            }

            // ---- CSV: SMILES, ID, IUPAC ----
            case 'csv': {
                btn.textContent = 'Fetching IUPAC names…';
                const smilesList = currentMolecules.map(m => m.smiles);
                const iupacNames = await _fetchIupacNames(smilesList);
                const csvEsc     = s => `"${String(s).replace(/"/g, '""')}"`;
                const rows       = ['SMILES,ID,IUPAC'];
                smilesList.forEach((smi, i) =>
                    rows.push([csvEsc(smi), csvEsc(`Mol ${i + 1}`), csvEsc(iupacNames[i] || '')].join(','))
                );
                _downloadText(rows.join('\n'), 'molecules.csv', 'text/csv');
                break;
            }

            // ---- SDF: mol block + SMILES / ID / IUPAC data fields ----
            case 'sdf': {
                btn.textContent = 'Fetching IUPAC names…';
                const smilesList = currentMolecules.map(m => m.smiles);
                const iupacNames = await _fetchIupacNames(smilesList);
                const blocks     = smilesList.map((smi, i) => {
                    const mol      = RDKit.get_mol(smi);
                    const molblock = mol.get_molblock(); // ends with "M  END\n"
                    mol.delete();
                    return (
                        molblock +
                        `> <SMILES>\n${smi}\n\n` +
                        `> <ID>\nMol ${i + 1}\n\n` +
                        `> <IUPAC>\n${iupacNames[i] || ''}\n\n` +
                        `$$$$`
                    );
                });
                _downloadText(blocks.join('\n'), 'molecules.sdf', 'chemical/x-mdl-sdfile');
                break;
            }

            case 'png': {
                btn.textContent = 'Preparing…';
                const fullSmiles = currentMolecules.map(m => m.smiles).join('.');
                const blob = await ketcher.generateImage(fullSmiles, { outputFormat: 'png' });
                _downloadBlob(blob, 'structure.png');
                break;
            }

            case 'svg': {
                btn.textContent = 'Preparing…';
                const fullSmiles = currentMolecules.map(m => m.smiles).join('.');
                const blob = await ketcher.generateImage(fullSmiles, { outputFormat: 'svg' });
                _downloadBlob(blob, 'structure.svg');
                break;
            }

            default:
                throw new Error(`Unknown format: ${selectedExportFormat}`);
        }

        btn.textContent = '✓ Downloaded';
        setTimeout(() => {
            btn.textContent = _exportBtnLabel(selectedExportFormat);
            btn.disabled = false;
        }, 2000);

    } catch (err) {
        console.error('Export failed:', err);
        btn.textContent = 'Export failed — check console';
        btn.disabled = false;
    }
});

// Copy All button for the inline SMILES list
document.getElementById('smiles-copy-all-btn').addEventListener('click', () => {
    const text = document.getElementById('smiles-list-textarea').value;
    if (!text) return;
    navigator.clipboard.writeText(text).then(() => {
        const btn = document.getElementById('smiles-copy-all-btn');
        btn.textContent = 'Copied!';
        setTimeout(() => { btn.textContent = 'Copy All'; }, 1800);
    });
});

/**
 * Fetch IUPAC names from PubChem for each SMILES.
 * Returns empty string for any that fail (network error, unknown structure).
 */
async function _fetchIupacNames(smilesList) {
    return Promise.all(smilesList.map(async (smiles) => {
        try {
            const url = `https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/smiles/` +
                        `${encodeURIComponent(smiles)}/property/IUPACName/JSON`;
            const res  = await fetch(url);
            if (!res.ok) return '';
            const data = await res.json();
            return data?.PropertyTable?.Properties?.[0]?.IUPACName ?? '';
        } catch {
            return '';
        }
    }));
}

function _downloadText(text, filename, mime) {
    _downloadBlob(new Blob([text], { type: mime }), filename);
}

function _downloadBlob(blob, filename) {
    const url = URL.createObjectURL(blob);
    const a   = document.createElement('a');
    a.href     = url;
    a.download = filename;
    document.body.appendChild(a);
    a.click();
    document.body.removeChild(a);
    URL.revokeObjectURL(url);
}

// ----------------------------------------------------------------
// Panel collapse toggle
// ----------------------------------------------------------------

document.getElementById('panel-toggle').addEventListener('click', () => {
    const panel    = document.getElementById('properties-panel');
    const toggle   = document.getElementById('panel-toggle');
    const collapsed = panel.classList.toggle('collapsed');
    toggle.classList.toggle('collapsed', collapsed);
    toggle.querySelector('span').textContent = collapsed ? '▶' : '◀';
});

// ----------------------------------------------------------------
// BoltzStar - Load by PDB ID
// ----------------------------------------------------------------

document.getElementById('load-pdb-btn').addEventListener('click', () => {
    const id = document.getElementById('pdb-id-input').value.trim();
    loadByPdbId(id, _setStarStatus);
});

document.getElementById('pdb-id-input').addEventListener('keydown', (e) => {
    if (e.key === 'Enter') loadByPdbId(e.target.value.trim(), _setStarStatus);
});

// ----------------------------------------------------------------
// BoltzStar - Load from file
// ----------------------------------------------------------------

// Track the last-loaded file so the Hydrate buttons can re-send it
let _hydrateFileData    = null;
let _hydrateFileName    = null;
let _hydratedPdbData    = null;   // last successful hydration result
let _hydratedOutputName = null;   // suggested download filename

document.getElementById('structure-file-input').addEventListener('change', async (e) => {
    const file = e.target.files[0];
    if (!file) return;

    // Cache the raw text for the Hydrate pipeline
    _hydrateFileData = await file.text();
    _hydrateFileName = file.name;

    // Cache for PLIP (uses same file as hydrate)
    _plipFileData = _hydrateFileData;
    _plipFileName = _hydrateFileName;

    // Show Analysis tools only when the local backend is running
    if (_backendAvailable) {
        document.getElementById('hydrate-section').classList.remove('hidden');
        document.getElementById('quick-hydrate-btn').disabled   = false;
        document.getElementById('full-hydrate-btn').disabled    = false;
        document.getElementById('biotite-hydrate-btn').disabled = false;
        document.getElementById('hydrate-status').classList.add('hidden');

        document.getElementById('plip-section').classList.remove('hidden');
        document.getElementById('run-plip-btn').disabled = false;
        document.getElementById('plip-results').innerHTML = '';
        document.getElementById('plip-status').classList.add('hidden');
    }

    loadFromFile(file, _setStarStatus);
});

// ----------------------------------------------------------------
// BoltzStar - Quick Hydrate / Full Hydrate buttons
// ----------------------------------------------------------------

// Per-mode UI metadata for the three hydrate buttons.
const _HYDRATE_MODES = {
    quick:   { id: 'quick-hydrate-btn',   idle: '⚡ Quick',   running: '⚡ Running…', done: '✓ Quick',
               status: 'Running geometric water placement…', resultLabel: 'bridging' },
    full:    { id: 'full-hydrate-btn',    idle: '🔬 Full',    running: '🔬 Running…', done: '✓ Full',
               status: 'Running WaterKit pipeline (~40 s)…', resultLabel: 'energy-scored' },
    biotite: { id: 'biotite-hydrate-btn', idle: '🧪 Biotite', running: '🧪 Running…', done: '✓ Biotite',
               status: 'Running Biotite WaterKit (experimental, ~40 s)…', resultLabel: 'Biotite' },
};

function _setHydrateButtonsDisabled(disabled) {
    for (const m of Object.values(_HYDRATE_MODES)) {
        document.getElementById(m.id).disabled = disabled;
    }
}

function _resetHydrateButtons() {
    for (const m of Object.values(_HYDRATE_MODES)) {
        const btn = document.getElementById(m.id);
        btn.textContent = m.idle;
        btn.disabled = false;
    }
}

async function _doHydrate(mode) {
    if (!_hydrateFileData) return;

    const meta      = _HYDRATE_MODES[mode];
    const activeBtn = document.getElementById(meta.id);
    const statusEl  = document.getElementById('hydrate-status');

    _setHydrateButtonsDisabled(true);
    activeBtn.textContent = meta.running;
    statusEl.textContent  = meta.status;
    statusEl.className = 'star-status loading';
    statusEl.classList.remove('hidden');

    try {
        const response = await fetch(`/hydrate?mode=${mode}`, {
            method:  'POST',
            headers: {
                'Content-Type': 'text/plain; charset=utf-8',
                'X-Filename':   _hydrateFileName || 'structure.cif',
            },
            body: _hydrateFileData,
        });

        if (!response.ok) {
            let msg = `Server error ${response.status}`;
            try { msg = (await response.json()).error || msg; } catch {}
            throw new Error(msg);
        }

        const hydratedPdb = await response.text();
        const waterCount  = parseInt(response.headers.get('X-Water-Count') || '0', 10);

        await loadFromData(hydratedPdb, 'pdb', _setStarStatus);

        // Store result for download and PLIP
        _hydratedPdbData    = hydratedPdb;
        _hydratedOutputName = (_hydrateFileName || 'structure')
            .replace(/\.(cif|pdb|mmcif)$/i, '') + `_hydrated_${mode}.pdb`;
        document.getElementById('download-hydrated-btn').classList.remove('hidden');

        // PLIP now analyses the hydrated structure
        _plipFileData = hydratedPdb;
        _plipFileName = _hydratedOutputName;
        document.getElementById('plip-results').innerHTML = '';
        document.getElementById('plip-status').classList.add('hidden');

        statusEl.textContent =
            `${waterCount} ${meta.resultLabel} water${waterCount !== 1 ? 's' : ''} placed`;
        statusEl.className = 'star-status success';

        activeBtn.textContent = meta.done;
        setTimeout(_resetHydrateButtons, 3000);

    } catch (err) {
        console.error('Hydration failed:', err);
        statusEl.textContent = `Hydration failed: ${err.message}`;
        statusEl.className    = 'star-status error';
        _resetHydrateButtons();
    }
}

document.getElementById('quick-hydrate-btn').addEventListener('click',   () => _doHydrate('quick'));
document.getElementById('full-hydrate-btn').addEventListener('click',    () => _doHydrate('full'));
document.getElementById('biotite-hydrate-btn').addEventListener('click', () => _doHydrate('biotite'));

// ----------------------------------------------------------------
// BoltzStar - Analysis / Style tab switching
// ----------------------------------------------------------------

function _setStarTab(tab) {
    const isAnalysis = tab === 'analysis';
    document.getElementById('star-tab-analysis').classList.toggle('active',  isAnalysis);
    document.getElementById('star-tab-style').classList.toggle('active',    !isAnalysis);
    document.getElementById('star-panel-analysis').classList.toggle('hidden', !isAnalysis);
    document.getElementById('star-panel-style').classList.toggle('hidden',    isAnalysis);
}

document.getElementById('star-tab-analysis').addEventListener('click', () => _setStarTab('analysis'));
document.getElementById('star-tab-style').addEventListener('click',    () => _setStarTab('style'));

// ----------------------------------------------------------------
// BoltzStar - PLIP analysis
// ----------------------------------------------------------------

// Track structure data available for PLIP (set when file loaded or hydration done)
let _plipFileData = null;
let _plipFileName = null;

document.getElementById('download-hydrated-btn').addEventListener('click', () => {
    if (!_hydratedPdbData) return;
    const blob = new Blob([_hydratedPdbData], { type: 'chemical/x-pdb' });
    const url  = URL.createObjectURL(blob);
    const a    = document.createElement('a');
    a.href     = url;
    a.download = _hydratedOutputName || 'hydrated.pdb';
    document.body.appendChild(a);
    a.click();
    document.body.removeChild(a);
    URL.revokeObjectURL(url);
});

document.getElementById('run-plip-btn').addEventListener('click', async () => {
    if (!_plipFileData) return;

    const btn      = document.getElementById('run-plip-btn');
    const statusEl = document.getElementById('plip-status');
    const resultsEl = document.getElementById('plip-results');

    btn.disabled    = true;
    btn.textContent = 'Analysing…';
    statusEl.textContent = 'Running PLIP interaction analysis…';
    statusEl.className   = 'star-status loading';
    statusEl.classList.remove('hidden');
    resultsEl.innerHTML  = '';

    try {
        const response = await fetch('/plip', {
            method:  'POST',
            headers: {
                'Content-Type': 'text/plain; charset=utf-8',
                'X-Filename':   _plipFileName || 'structure.pdb',
            },
            body: _plipFileData,
        });

        const data = await response.json();

        if (!response.ok || data.error) {
            throw new Error(data.error || `Server error ${response.status}`);
        }

        if (!data.sites || data.sites.length === 0) {
            statusEl.textContent = 'No ligand binding sites detected.';
            statusEl.className   = 'star-status info';
            btn.textContent      = 'Run PLIP Analysis';
            btn.disabled         = false;
            return;
        }

        // Render results
        resultsEl.innerHTML = data.sites.map(site => _renderPlipSite(site, data.sites.length)).join('');
        _attachPlipToggles();

        const total = data.sites.reduce((s, site) => s + site.total, 0);
        statusEl.textContent = `${total} interaction${total !== 1 ? 's' : ''} found across ${data.sites.length} site${data.sites.length !== 1 ? 's' : ''}`;
        statusEl.className   = 'star-status success';

    } catch (err) {
        console.error('PLIP failed:', err);
        statusEl.textContent = `Analysis failed: ${err.message}`;
        statusEl.className   = 'star-status error';
    }

    btn.textContent = 'Run PLIP Analysis';
    btn.disabled    = false;
});

// ---- PLIP interaction type metadata ----
const PLIP_TYPES = {
    hbonds:         { label: 'Hydrogen Bonds',    color: '#2563eb' },
    hydrophobic:    { label: 'Hydrophobic',        color: '#525252' },
    pistacking:     { label: 'π-Stacking',         color: '#7c3aed' },
    pication:       { label: 'π-Cation',           color: '#b45309' },
    saltbridges:    { label: 'Salt Bridges',       color: '#dc2626' },
    waterbridges:   { label: 'Water Bridges',      color: '#0891b2' },
    halogenbonds:   { label: 'Halogen Bonds',      color: '#16a34a' },
    metalcomplexes: { label: 'Metal Complexes',    color: '#9d174d' },
};

// Column definitions per interaction type
const PLIP_COLS = {
    hbonds: [
        { key: 'resnr',     head: 'Res#' },
        { key: 'restype',   head: 'AA' },
        { key: 'reschain',  head: 'Chain' },
        { key: 'dist_h_a',  head: 'Dist H-A (Å)' },
        { key: 'dist_d_a',  head: 'Dist D-A (Å)' },
        { key: 'don_angle', head: 'Angle (°)' },
        { key: 'protisdon', head: 'Prot donor', bool: true },
        { key: 'sidechain', head: 'Side chain', bool: true },
        { key: 'donortype', head: 'Donor' },
        { key: 'acctype',   head: 'Acceptor' },
    ],
    hydrophobic: [
        { key: 'resnr',    head: 'Res#' },
        { key: 'restype',  head: 'AA' },
        { key: 'reschain', head: 'Chain' },
        { key: 'dist',     head: 'Dist (Å)' },
    ],
    pistacking: [
        { key: 'resnr',     head: 'Res#' },
        { key: 'restype',   head: 'AA' },
        { key: 'reschain',  head: 'Chain' },
        { key: 'cent_dist', head: 'Dist (Å)' },
        { key: 'angle',     head: 'Angle (°)' },
        { key: 'offset',    head: 'Offset (Å)' },
        { key: 'type',      head: 'Type' },
    ],
    pication: [
        { key: 'resnr',       head: 'Res#' },
        { key: 'restype',     head: 'AA' },
        { key: 'reschain',    head: 'Chain' },
        { key: 'dist',        head: 'Dist (Å)' },
        { key: 'offset',      head: 'Offset (Å)' },
        { key: 'protcharged', head: 'Prot charged', bool: true },
    ],
    saltbridges: [
        { key: 'resnr',    head: 'Res#' },
        { key: 'restype',  head: 'AA' },
        { key: 'reschain', head: 'Chain' },
        { key: 'dist',     head: 'Dist (Å)' },
        { key: 'protispos',head: 'Prot +ve', bool: true },
        { key: 'lig_group',head: 'Lig group' },
    ],
    waterbridges: [
        { key: 'resnr',       head: 'Res#' },
        { key: 'restype',     head: 'AA' },
        { key: 'reschain',    head: 'Chain' },
        { key: 'dist_a_w',    head: 'Dist A-W (Å)' },
        { key: 'dist_d_w',    head: 'Dist D-W (Å)' },
        { key: 'don_angle',   head: 'D angle (°)' },
        { key: 'water_angle', head: 'W angle (°)' },
        { key: 'protisdon',   head: 'Prot donor', bool: true },
        { key: 'donortype',   head: 'Donor' },
        { key: 'acctype',     head: 'Acceptor' },
    ],
    halogenbonds: [
        { key: 'resnr',     head: 'Res#' },
        { key: 'restype',   head: 'AA' },
        { key: 'reschain',  head: 'Chain' },
        { key: 'dist',      head: 'Dist (Å)' },
        { key: 'don_angle', head: 'D angle (°)' },
        { key: 'acc_angle', head: 'A angle (°)' },
        { key: 'sidechain', head: 'Side chain', bool: true },
        { key: 'donortype', head: 'Halogen' },
        { key: 'acctype',   head: 'Acceptor' },
    ],
    metalcomplexes: [
        { key: 'resnr',        head: 'Res#' },
        { key: 'restype',      head: 'AA' },
        { key: 'reschain',     head: 'Chain' },
        { key: 'metal_type',   head: 'Metal' },
        { key: 'target_type',  head: 'Target' },
        { key: 'dist',         head: 'Dist (Å)' },
        { key: 'coordination', head: 'Coord.' },
        { key: 'geometry',     head: 'Geometry' },
    ],
};

function _renderPlipSite(site, numSites) {
    const ixt = site.interactions;

    const siteLabel = numSites > 1
        ? `<div class="plip-site-id">${site.id}</div>` : '';

    const groups = Object.entries(PLIP_TYPES)
        .filter(([key]) => (ixt[key] || []).length > 0)   // hide empty types
        .map(([key, meta]) => {
            const items = ixt[key] || [];
            const cols  = PLIP_COLS[key] || [];

            const thead = cols.map(c => `<th>${c.head}</th>`).join('');
            const tbody = items.map((item, i) => {
                const cells = cols.map(c => {
                    const v = item[c.key];
                    if (c.bool) return `<td class="plip-bool">${v ? '✓' : '✗'}</td>`;
                    return `<td>${v ?? '—'}</td>`;
                }).join('');
                return `<tr><td class="plip-idx">${i + 1}</td>${cells}</tr>`;
            }).join('');

            return `<div class="plip-group open">
                <div class="plip-group-header" data-plip-toggle>
                    <span class="plip-type-label" style="border-color:${meta.color};color:${meta.color}">${meta.label}</span>
                    <span class="plip-group-count" style="background:${meta.color}">${items.length}</span>
                </div>
                <div class="plip-group-body open">
                    <div class="plip-table-wrap">
                        <table class="plip-table">
                            <thead><tr><th>#</th>${thead}</tr></thead>
                            <tbody>${tbody}</tbody>
                        </table>
                    </div>
                </div>
            </div>`;
        }).join('');

    if (!groups) {
        return siteLabel + '<p class="plip-none">No interactions detected for this site.</p>';
    }

    return siteLabel + groups;
}

function _attachPlipToggles() {
    document.querySelectorAll('[data-plip-toggle]').forEach(header => {
        header.addEventListener('click', () => {
            const body = header.nextElementSibling;
            if (body && body.classList.contains('plip-group-body')) {
                body.classList.toggle('open');
            }
        });
    });
}

// ----------------------------------------------------------------
// BoltzViewer - Apply Boltz Style preset
// ----------------------------------------------------------------

document.getElementById('apply-preset-btn').addEventListener('click', () => {
    applyBoltzPreset();
});

// ----------------------------------------------------------------
// BoltzViewer - Preset / Manual mode toggle
// ----------------------------------------------------------------

const _presetBtn   = document.getElementById('style-mode-preset');
const _manualBtn   = document.getElementById('style-mode-manual');
const _presetView  = document.getElementById('style-preset-view');
const _manualView  = document.getElementById('style-manual-view');
const _proteinPick = document.getElementById('protein-color-input');
const _ligandPick  = document.getElementById('ligand-color-input');

function _setStyleMode(mode) {
    const isPreset = mode === 'preset';
    _presetBtn.classList.toggle('active', isPreset);
    _manualBtn.classList.toggle('active', !isPreset);
    _presetView.classList.toggle('hidden', !isPreset);
    _manualView.classList.toggle('hidden',  isPreset);
}

_presetBtn.addEventListener('click', () => _setStyleMode('preset'));
_manualBtn.addEventListener('click', () => _setStyleMode('manual'));

// Hex string "#RRGGBB" → 0xRRGGBB integer for Molstar Color
function _hexToInt(hex) {
    return parseInt(hex.slice(1), 16);
}

function _applyManualColors() {
    setSurfaceColors({
        proteinColor: _hexToInt(_proteinPick.value),
        ligandColor:  _hexToInt(_ligandPick.value),
    });
}

// 'input' fires on every drag; 'change' on commit. Use 'input' for live feel.
_proteinPick.addEventListener('input', _applyManualColors);
_ligandPick.addEventListener('input',  _applyManualColors);

// ----------------------------------------------------------------
// Status helpers
// ----------------------------------------------------------------

function _setStarStatus(message, type) {
    const el = document.getElementById('star-load-status');
    el.textContent = message;
    el.className   = `star-status ${type}`;
    el.classList.remove('hidden');
}