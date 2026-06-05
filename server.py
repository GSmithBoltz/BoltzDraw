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
    mode       = request.args.get('mode', 'full')   # 'quick' | 'full'
    fmt        = 'pdb' if filename.lower().endswith('.pdb') else 'mmcif'

    if not input_data.strip():
        return jsonify({'error': 'Empty request body'}), 400

    if mode not in ('quick', 'full'):
        return jsonify({'error': f'Unknown mode "{mode}". Use quick or full.'}), 400

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
# Entry point
# ----------------------------------------------------------------

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 4400))
    print(f'BoltzStar server running on http://localhost:{port}', flush=True)
    app.run(host='0.0.0.0', port=port, debug=False)
