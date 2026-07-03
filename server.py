"""
BoltzStar backend server.

Serves all static files (replaces SimpleHTTPRequestHandler) and exposes:
    POST /hydrate   — PDBFixer + geometric water placement
"""

import os
import traceback
from flask import Flask, request, jsonify, send_from_directory

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
app = Flask(__name__, static_folder=BASE_DIR)


# ----------------------------------------------------------------
# Static file serving
# ----------------------------------------------------------------

@app.route('/')
def index():
    return send_from_directory(BASE_DIR, 'index.html')

@app.route('/<path:path>')
def static_files(path):
    return send_from_directory(BASE_DIR, path)


# ----------------------------------------------------------------
# Backend availability probe (used by frontend to detect local server)
# ----------------------------------------------------------------

@app.route('/ping')
def ping():
    return jsonify({'status': 'ok', 'backend': 'boltzstar'})


# ----------------------------------------------------------------
# Hydration endpoint
# ----------------------------------------------------------------

@app.route('/hydrate', methods=['POST'])
def hydrate():
    from hydrate import run_pipeline

    # Accept raw body with X-Filename header
    input_data = request.get_data(as_text=True)
    filename   = request.headers.get('X-Filename', 'structure.cif')
    mode       = request.args.get('mode', 'full')   # 'quick' | 'full' | 'biotite'
    fmt        = 'pdb' if filename.lower().endswith('.pdb') else 'mmcif'

    if not input_data.strip():
        return jsonify({'error': 'Empty request body'}), 400

    if mode not in ('quick', 'full', 'biotite'):
        return jsonify({'error': f'Unknown mode "{mode}". Use quick, full or biotite.'}), 400

    try:
        result_pdb, water_count = run_pipeline(input_data, fmt, mode=mode)
        return result_pdb, 200, {
            'Content-Type': 'text/plain; charset=utf-8',
            'X-Water-Count': str(water_count),
            'X-Hydrate-Mode': mode,
        }
    except Exception as e:
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500


# ----------------------------------------------------------------
# PLIP analysis endpoint
# ----------------------------------------------------------------

@app.route('/plip', methods=['POST'])
def plip():
    from plip_analysis import run_plip

    input_data = request.get_data(as_text=True)
    filename   = request.headers.get('X-Filename', 'structure.cif')
    fmt        = 'pdb' if filename.lower().endswith('.pdb') else 'mmcif'

    if not input_data.strip():
        return jsonify({'error': 'Empty request body'}), 400

    try:
        result = run_plip(input_data, fmt)
        return jsonify(result)
    except Exception as e:
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500


# ----------------------------------------------------------------
# ADME endpoints (Boltz cloud, proxied) — async start + poll
# ----------------------------------------------------------------
#
# The browser cannot import the boltz_api Python SDK, so it POSTs the selected
# molecules + the user's pasted API key here. ADME predictions can run for
# several minutes, so we never block a single request waiting for completion:
#   POST /adme/start  -> submit, return the prediction id immediately
#   POST /adme/poll   -> retrieve current status; return results once done
# The frontend polls /adme/poll until status is "succeeded" or "failed".

@app.route('/adme/start', methods=['POST'])
def adme_start():
    body      = request.get_json(silent=True) or {}
    api_key   = (body.get('api_key') or '').strip()
    molecules = body.get('molecules') or []

    if not api_key:
        return jsonify({'error': 'Missing Boltz API key'}), 400
    if not molecules:
        return jsonify({'error': 'No molecules provided'}), 400

    try:
        prediction_id = _adme_start(molecules, api_key)
        return jsonify({'success': True, 'prediction_id': prediction_id})
    except Exception as e:
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500


@app.route('/adme/poll', methods=['POST'])
def adme_poll():
    body          = request.get_json(silent=True) or {}
    api_key       = (body.get('api_key') or '').strip()
    prediction_id = (body.get('prediction_id') or '').strip()
    molecules     = body.get('molecules') or []   # preserves order / fills gaps

    if not api_key or not prediction_id:
        return jsonify({'error': 'Missing api_key or prediction_id'}), 400

    try:
        status, results = _adme_poll(prediction_id, api_key, molecules)
        return jsonify({'success': True, 'status': status, 'results': results})
    except Exception as e:
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500


# ================================================================
# ALL boltz_api SDK coupling lives in the three helpers below.
# Confirmed against boltz_api 0.36.1. The API key is never logged.
#
# Set ADME_MOCK=1 to bypass the SDK and return deterministic fake results.
# ================================================================

def _adme_client(api_key):
    from boltz_api import Boltz
    return Boltz(api_key=api_key)


def _adme_start(molecules, api_key):
    """Submit an ADME batch and return the prediction id (no waiting)."""
    if os.environ.get('ADME_MOCK') == '1':
        return 'mock_pred_0001'

    adme_input = {
        'molecules': [
            {'smiles': m.get('smiles'), 'id': m.get('id')}
            for m in molecules
        ],
    }
    prediction = _adme_client(api_key).predictions.adme.start(
        model='adme-v1', input=adme_input)
    return prediction.id


def _adme_poll(prediction_id, api_key, molecules):
    """Retrieve one status snapshot. Returns (status, results) where results
    is None until the prediction succeeds. Raises on a failed prediction."""
    if os.environ.get('ADME_MOCK') == '1':
        return 'succeeded', _mock_adme(molecules)

    prediction = _adme_client(api_key).predictions.adme.retrieve(prediction_id)
    status = prediction.status

    if status in ('pending', 'running'):
        return status, None
    if status == 'failed':
        msg = prediction.error.message if prediction.error else 'prediction failed'
        raise RuntimeError(msg)

    return 'succeeded', _map_adme_output(prediction, molecules)


def _map_adme_output(prediction, molecules):
    """Map a succeeded prediction's per-molecule output to our result rows,
    keyed on the client id we sent (echoed back as external_id)."""
    by_id = {}
    output_molecules = prediction.output.molecules if prediction.output else []
    for m in output_molecules:
        key = m.external_id or m.id
        if m.status == 'succeeded':
            by_id[key] = {
                'id':           key,
                'status':       'ok',
                'logd':         m.adme.lipophilicity,   # internal LogD prediction
                'permeability': m.adme.permeability,
                'solubility':   m.adme.solubility,       # categorical: high-confidence|medium-confidence|high-risk
            }
        else:
            err = m.error.message if m.error else 'prediction failed'
            by_id[key] = {'id': key, 'status': 'error', 'error': err}

    # Preserve request order; surface a clear miss if a molecule came back blank.
    results = []
    for m in molecules:
        mid = m.get('id')
        results.append(by_id.get(mid, {
            'id': mid, 'status': 'error', 'error': 'no result returned',
        }))
    return results


def _mock_adme(molecules):
    """Deterministic fake ADME results for UI testing (ADME_MOCK=1).
    Mirrors the real response shape, including categorical solubility."""
    sol_buckets = ['high-confidence', 'medium-confidence', 'high-risk']
    out = []
    for i, m in enumerate(molecules):
        out.append({
            'id':           m.get('id'),
            'status':       'ok',
            'logd':         round(0.5 + 0.13 * i, 2),
            'permeability': round(-5.0 - 0.11 * i, 2),
            'solubility':   sol_buckets[i % len(sol_buckets)],
        })
    return out


# ----------------------------------------------------------------
# pKa endpoint (local, mirrors the hosted UnipKa service)
# ----------------------------------------------------------------

@app.route('/pka', methods=['POST'])
def pka():
    from pka_calc import run_pka

    body   = request.get_json(silent=True) or {}
    smiles = (body.get('smiles') or '').strip()
    ph     = body.get('pH', 7.4)

    if not smiles:
        return jsonify({'success': False, 'error': 'Empty SMILES'}), 400

    try:
        result = run_pka(smiles, ph)
        return jsonify(result)
    except Exception as e:
        traceback.print_exc()
        return jsonify({'success': False, 'error': str(e)}), 500


# ----------------------------------------------------------------
# Entry point
# ----------------------------------------------------------------

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 4400))
    print(f'BoltzStar server running on http://localhost:{port}', flush=True)
    app.run(host='0.0.0.0', port=port, debug=False)
