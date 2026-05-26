# BoltzDraw

A browser-based molecular toolkit — draw structures, explore chemical space, and import from PDB. No installation, no backend.

## Modes

| Mode | Description |
|------|-------------|
| **BoltzDraw** | 2D structure editor (Ketcher) with live property calculations |
| **BoltzStar** | 3D structure viewer (Mol* / Molstar) — load by PDB ID or upload CIF/PDB |
| **BoltzSpace** | Chemical space explorer — UMAP and t-SNE projections from a CSV of SMILES |

## Analysis panel (BoltzDraw)

**Cards** — per-molecule property grid, updated as you draw. Lipinski violations highlighted in amber/red.

| MW | cLogP | TPSA | HBA | HBD | Fsp3 | RotB | HAC | Hetero | ArRings | Stereo | Unspec |

**pKa** — estimated pKa values with a pH distribution chart.

**Export** — download the current canvas as PNG, SVG, CSV, or SDF. CSV and SDF include SMILES, molecule ID, and IUPAC name (fetched from PubChem). SMILES lists all structures inline with a copy-all button.

## Import panel (BoltzDraw)

**Peptide** — convert a one-letter amino acid sequence to SMILES and load it into the editor. Supports all 20 canonical L-amino acids (uppercase), their D-forms (lowercase), and N/C terminal modifications (free, acetyl, formyl, palmitoyl / free acid, amide, methyl ester).

**PDB** — enter a 4-character PDB accession code to fetch all small-molecule ligands from the RCSB. Each ligand is shown with its 2D structure, CCD code, name, and formula. Click **Import into Editor** to load it directly into Ketcher.

## Running locally

Requires a local HTTP server (needed for the Ketcher iframe and ES modules):

```bash
python3 -m http.server 4400
# then open http://localhost:4400
```

## Stack

- [Ketcher](https://github.com/epam/ketcher) — 2D structure editor (EPAM)
- [RDKit.js](https://github.com/rdkit/rdkit-js) — property calculations and mol blocks (WebAssembly)
- [Mol*](https://molstar.org/) — 3D structure viewer
- [RCSB PDB API](https://data.rcsb.org) — PDB ligand data
- [PubChem PUG REST](https://pubchem.ncbi.nlm.nih.gov/docs/pug-rest) — IUPAC names at export time

## License

MIT
