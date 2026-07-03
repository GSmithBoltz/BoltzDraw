/**
 * pka.js
 *
 * pKa tab: calls the Railway-hosted UnipKa API, renders result tiles,
 * and draws the microstate population distribution chart with Chart.js.
 */

// Local Flask route (preferred when the backend is running) and the hosted
// Railway UnipKa service used as a fallback when it isn't.
const PKA_LOCAL_URL = '/pka';
const PKA_API_URL   = 'https://web-production-0ede.up.railway.app/pka';

// Chart.js instance - kept so we can destroy before redrawing
let pkaChart = null;

// ----------------------------------------------------------------
// Public API
// ----------------------------------------------------------------

/**
 * Update the SMILES display in the pKa tab.
 * Called whenever the view switches to pKa or molecules change.
 *
 * @param {Array} molecules - current molecule array from app state
 */
export function updatePkaSmiles(molecules) {
    const display = document.getElementById('pka-smiles');
    const btn     = document.getElementById('calculate-pka-btn');

    if (molecules.length > 0 && molecules[0].smiles) {
        display.textContent = molecules[0].smiles;
        btn.disabled = false;
    } else {
        display.textContent = 'Draw a molecule in the editor';
        btn.disabled = true;
    }
}

/**
 * Trigger pKa calculation for the first molecule.
 * Wired to the Calculate pKa button in app.js.
 *
 * @param {Array}   molecules        - current molecule array from app state
 * @param {boolean} backendAvailable - true when the local Flask backend is up;
 *                                      routes to /pka locally, else Railway.
 */
export async function calculatePka(molecules, backendAvailable = false) {
    const smiles = molecules[0]?.smiles;
    if (!smiles) return;

    const url = backendAvailable ? PKA_LOCAL_URL : PKA_API_URL;

    const resultsDiv     = document.getElementById('pka-results');
    const chartContainer = document.getElementById('pka-chart-container');
    const btn            = document.getElementById('calculate-pka-btn');

    // Loading state
    btn.disabled = true;
    btn.textContent = 'Calculating...';
    resultsDiv.innerHTML = '<div class="pka-loading">Calculating pKa properties...</div>';
    resultsDiv.classList.add('active');
    chartContainer.classList.remove('active');

    try {
        const response = await fetch(url, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ smiles, pH: 7.4 }),
        });

        const data = await response.json();

        if (data.success) {
            _displayResults(data);
        } else {
            resultsDiv.innerHTML = `<div class="pka-error">Error: ${data.error || 'Unknown error'}</div>`;
        }
    } catch (err) {
        resultsDiv.innerHTML = `<div class="pka-error">Network error: ${err.message}</div>`;
    } finally {
        btn.disabled = false;
        btn.textContent = 'Calculate pKa';
    }
}

// ----------------------------------------------------------------
// Private helpers
// ----------------------------------------------------------------

function _displayResults(data) {
    const resultsDiv     = document.getElementById('pka-results');
    const chartContainer = document.getElementById('pka-chart-container');

    const acidic   = data.pka?.acidic  != null ? data.pka.acidic.toFixed(2)  : 'N/A';
    const basic    = data.pka?.basic   != null ? data.pka.basic.toFixed(2)   : 'N/A';
    const logD     = data.logd         != null ? data.logd.toFixed(2)         : 'N/A';
    const dominant = data.dominant_microstate || 'N/A';

    resultsDiv.innerHTML = `
        <div class="pka-results-grid">
            <div class="pka-result-item">
                <div class="pka-result-label">Acidic pKa</div>
                <div class="pka-result-value">${acidic}</div>
            </div>
            <div class="pka-result-item">
                <div class="pka-result-label">Basic pKa</div>
                <div class="pka-result-value">${basic}</div>
            </div>
            <div class="pka-result-item">
                <div class="pka-result-label">LogD (pH 7.4)</div>
                <div class="pka-result-value">${logD}</div>
            </div>
            <div class="pka-result-item">
                <div class="pka-result-label">Microstates</div>
                <div class="pka-result-value">${data.microstates?.length || 0}</div>
            </div>
        </div>
        <div class="pka-label" style="margin-top:8px;">Dominant at pH 7.4</div>
        <div class="smiles-display" style="font-size:10px;">${dominant}</div>
    `;
    resultsDiv.classList.add('active');

    if (data.distribution_curve?.microstates?.length) {
        _drawChart(data.distribution_curve);
        chartContainer.classList.add('active');
    }
}

function _drawChart(curveData) {
    const ctx = document.getElementById('pka-chart').getContext('2d');

    if (pkaChart) pkaChart.destroy();

    // Boltz green palette for chart lines
    const colors = [
        'rgba(53,  88,  61, 1)',   // forest green
        'rgba(103, 130, 107, 1)',  // sage green
        'rgba(180, 83,  9,  1)',   // amber
        'rgba(59,  130, 246, 1)',  // blue
        'rgba(139, 92,  246, 1)',  // purple
    ];

    const datasets = curveData.microstates.map((ms, i) => ({
        label: `Charge ${ms.charge >= 0 ? '+' : ''}${ms.charge}`,
        data: ms.populations.map((pop, j) => ({ x: curveData.ph_values[j], y: pop })),
        borderColor: colors[i % colors.length],
        backgroundColor: 'transparent',
        borderWidth: 2,
        pointRadius: 0,
        tension: 0.4,
    }));

    pkaChart = new Chart(ctx, {
        type: 'line',
        data: { datasets },
        options: {
            responsive: true,
            maintainAspectRatio: true,
            scales: {
                x: {
                    type: 'linear',
                    title: { display: true, text: 'pH', color: '#1a1a1a' },
                    min: 0,
                    max: 14,
                    ticks: { color: '#5c5c5c' },
                    grid:  { color: 'rgba(0,0,0,0.06)' },
                },
                y: {
                    title: { display: true, text: 'Population', color: '#1a1a1a' },
                    min: 0,
                    max: 1,
                    ticks: { color: '#5c5c5c' },
                    grid:  { color: 'rgba(0,0,0,0.06)' },
                },
            },
            plugins: {
                legend: { labels: { color: '#1a1a1a', boxWidth: 12 } },
                tooltip: {
                    callbacks: {
                        label: (ctx) => `${ctx.dataset.label}: ${(ctx.parsed.y * 100).toFixed(1)}%`,
                    },
                },
            },
        },
    });
}
