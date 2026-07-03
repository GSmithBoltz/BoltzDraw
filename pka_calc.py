"""
pka_calc.py

Local pKa calculation, exposed by server.py's POST /pka route. Mirrors the
JSON shape that the hosted Railway UnipKa service returns, so the frontend
(js/pka.js) consumes it without any changes:

    {
        "success": true,
        "pka": { "acidic": <float|null>, "basic": <float|null> },
        "logd": <float|null>,
        "dominant_microstate": "<SMILES>",
        "microstates": [ ... ],
        "distribution_curve": {
            "ph_values": [0.0, 0.5, ... 14.0],
            "microstates": [ { "charge": <int>, "populations": [ ... ] }, ... ]
        }
    }

All UnipKa coupling is isolated in _run_unipka(); set PKA_MOCK=1 to bypass it
and return a deterministic fake result (no model download / inference) for
quick UI testing.
"""

import os
import math

# pH grid for the population-vs-pH distribution chart: 0.0 .. 14.0 step 0.5.
PH_GRID = [round(0.5 * i, 1) for i in range(29)]

# Lazily-created, reused UnipKa instance (model load is one-time per process).
_UNIPKA = None


def _get_unipka():
    global _UNIPKA
    if _UNIPKA is None:
        from unipka import UnipKa
        _UNIPKA = UnipKa()
    return _UNIPKA


def _nan_to_none(value, ndigits=2):
    """UnipKa returns NaN when a macro-pKa is undefined (e.g. no acidic group).
    JSON has no NaN, so map it to None and round real values."""
    try:
        f = float(value)
    except (TypeError, ValueError):
        return None
    return None if math.isnan(f) else round(f, ndigits)


def run_pka(smiles, ph=7.4):
    """Compute pKa / logD / microstate distribution for a single SMILES.

    Returns a dict in the shape js/pka.js expects (see module docstring).
    """
    if os.environ.get('PKA_MOCK') == '1':
        return _mock_pka(smiles, ph)
    return _run_unipka(smiles, ph)


# ----------------------------------------------------------------
# Real UnipKa integration
# ----------------------------------------------------------------

def _run_unipka(smiles, ph):
    """ALL unipka coupling lives here. Builds the documented JSON shape from
    the UnipKa class (confirmed against unipka 0.0.6). Lazy-imports so
    server.py boots without unipka installed.

    The population-vs-pH curve is built by aggregating per-microstate
    populations into per-charge macrostate populations across PH_GRID, using
    UnipKa's own get_distribution() at each pH (its internal thermodynamics,
    not a re-derivation)."""
    from rdkit import Chem

    u = _get_unipka()

    acidic = _nan_to_none(u.get_acidic_macro_pka(smiles))
    basic  = _nan_to_none(u.get_basic_macro_pka(smiles))
    logd   = _nan_to_none(u.get_logd(smiles, pH=ph))

    dom_mol  = u.get_dominant_microstate(smiles, pH=ph)
    dominant = Chem.MolToSmiles(dom_mol) if dom_mol is not None else smiles

    # Microstate ensemble (charge + SMILES) from a representative distribution.
    base = u.get_distribution(smiles, pH=ph)
    microstates = [
        {'charge': int(row['charge']), 'smiles': str(row['smiles'])}
        for _, row in base.iterrows()
    ]
    charges = sorted({int(c) for c in base['charge']})

    # Population vs pH, summed per charge (macrostate population).
    pops_by_charge = {c: [] for c in charges}
    for p in PH_GRID:
        agg = u.get_distribution(smiles, pH=p).groupby('charge')['population'].sum()
        for c in charges:
            pops_by_charge[c].append(round(float(agg.get(c, 0.0)), 4))

    curve = [{'charge': c, 'populations': pops_by_charge[c]} for c in charges]

    return {
        'success': True,
        'pka': {'acidic': acidic, 'basic': basic},
        'logd': logd,
        'dominant_microstate': dominant,
        'microstates': microstates,
        'distribution_curve': {'ph_values': PH_GRID, 'microstates': curve},
    }


# ----------------------------------------------------------------
# Mock (PKA_MOCK=1) — deterministic, no dependencies
# ----------------------------------------------------------------

def _mock_pka(smiles, ph):
    """Deterministic fake pKa result for UI testing.

    Produces two Henderson-Hasselbalch population curves (a neutral and a
    +1 charged microstate) so js/pka.js can render its tiles and chart.
    """
    acidic = 9.2
    basic  = 4.1

    ph_values = [round(0.5 * i, 1) for i in range(29)]  # 0.0 .. 14.0

    # Neutral (charge 0) vs protonated base (charge +1), governed by `basic`.
    neutral, cation = [], []
    for p in ph_values:
        frac_cation = 1.0 / (1.0 + 10 ** (p - basic))
        cation.append(round(frac_cation, 4))
        neutral.append(round(1.0 - frac_cation, 4))

    dominant = smiles  # neutral form dominates at physiological pH for this mock
    logd = round(2.3 - 0.0, 2)

    return {
        'success': True,
        'pka': {'acidic': acidic, 'basic': basic},
        'logd': logd,
        'dominant_microstate': dominant,
        'microstates': [
            {'charge': 0,  'smiles': smiles},
            {'charge': 1,  'smiles': smiles},
        ],
        'distribution_curve': {
            'ph_values': ph_values,
            'microstates': [
                {'charge': 0, 'populations': neutral},
                {'charge': 1, 'populations': cation},
            ],
        },
    }
