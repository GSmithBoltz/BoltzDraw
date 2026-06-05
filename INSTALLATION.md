# BoltzStar — Local Installation Guide

> **For Claude Code users:** Hand this file to your Claude Code instance with the message:
> *"Please follow INSTALLATION.md to set up BoltzStar on this machine."*
> Claude Code can run every command below automatically.

---

## What this installs

BoltzStar is a molecular structure viewer with a hydration pipeline. The browser-based
app (Molstar, RDKit, Ketcher) works on any static host. The **Hydrate** feature requires
a local Python backend with:

- **Miniconda** — Python environment manager
- **PDBFixer** (OpenMM) — structure preparation
- **AutoGrid4** — AutoDock energy grid calculation
- **WaterKit** — energy-scored water placement
- **OpenBabel** — molecular format conversion
- **Flask** — lightweight web server

---

## Step 1 — Check prerequisites

```bash
# Confirm macOS architecture (arm64 = Apple Silicon, x86_64 = Intel)
uname -m

# Check if Miniconda is already installed
conda --version 2>/dev/null || echo "Miniconda not found"
```

---

## Step 2 — Install Miniconda (skip if already installed)

**Apple Silicon (arm64):**
```bash
curl -fsSL https://repo.anaconda.com/miniconda/Miniconda3-latest-MacOSX-arm64.sh \
  -o /tmp/miniconda.sh
bash /tmp/miniconda.sh -b -p "$HOME/miniconda3"
~/miniconda3/bin/conda init zsh
source ~/.zshrc
```

**Intel (x86_64):**
```bash
curl -fsSL https://repo.anaconda.com/miniconda/Miniconda3-latest-MacOSX-x86_64.sh \
  -o /tmp/miniconda.sh
bash /tmp/miniconda.sh -b -p "$HOME/miniconda3"
~/miniconda3/bin/conda init zsh
source ~/.zshrc
```

Accept the Anaconda Terms of Service (required once):
```bash
~/miniconda3/bin/conda tos accept --override-channels \
  --channel https://repo.anaconda.com/pkgs/main
~/miniconda3/bin/conda tos accept --override-channels \
  --channel https://repo.anaconda.com/pkgs/r
```

---

## Step 3 — Create the boltzstar environment

From the project root directory (where `environment.yml` lives):

```bash
~/miniconda3/bin/conda env create -f environment.yml
```

This takes 5–10 minutes — it installs AmberTools, AutoGrid4, PDBFixer, RDKit,
OpenBabel, WaterKit, and Flask in one step.

---

## Step 4 — Verify the installation

```bash
~/miniconda3/bin/conda run -n boltzstar python -c "
import waterkit, pdbfixer, flask
import shutil
print('waterkit  OK')
print('pdbfixer  OK')
print('flask     OK')
print('autogrid4:', shutil.which('autogrid4') or 'not found — check conda install')
"
```

All four lines should print without errors. If `autogrid4` shows "not found", run:
```bash
~/miniconda3/bin/conda install -n boltzstar -c conda-forge autogrid -y
```

---

## Step 5 — Configure the launcher

Update `.claude/launch.json` so Claude Code uses the correct Python for this machine.
Find the Python path first:

```bash
~/miniconda3/bin/conda run -n boltzstar which python
```

Then set `.claude/launch.json` to:

```json
{
  "version": "0.0.1",
  "configurations": [
    {
      "name": "BoltzDraw",
      "runtimeExecutable": "<output of the which python command above>",
      "runtimeArgs": ["-u", "server.py"],
      "port": 4400,
      "autoPort": true
    }
  ]
}
```

For example, on a typical Mac the value will be:
`/Users/<your-username>/miniconda3/envs/boltzstar/bin/python`

---

## Step 6 — Run the app

```bash
~/miniconda3/bin/conda run -n boltzstar python -u server.py
```

Then open **http://localhost:4400** in your browser.

The app loads immediately. The **Hydrate** buttons in the BoltzStar viewer appear
automatically once the local server is detected — they are hidden on GitHub Pages
or any other static host where the backend is not running.

---

## Troubleshooting

**`ModuleNotFoundError: No module named 'waterkit'`**
```bash
~/miniconda3/bin/conda run -n boltzstar pip install \
  git+https://github.com/forlilab/waterkit.git
```

**`ModuleNotFoundError: No module named 'meeko'`**
```bash
~/miniconda3/bin/conda run -n boltzstar pip install meeko gemmi
```

**AutoGrid4 error: "No partial atomic charges found"**
This means OpenBabel conversion ran without charges. Ensure the `boltzstar`
environment is being used (not system Python). Re-run server.py via
`conda run -n boltzstar`.

**Port 4400 already in use**
```bash
lsof -ti :4400 | xargs kill -9
```

---

## What each tool does

| Tool | Role |
|---|---|
| PDBFixer | Adds missing atoms and hydrogens to Boltz-2 output |
| OpenBabel | Converts protein PDB → PDBQT with Gasteiger charges |
| meeko + RDKit | Converts ligand PDB → PDBQT with AutoDock atom types |
| AutoGrid4 | Computes OA/HD/SW/OW/HW energy maps around the binding site |
| WaterKit | Places TIP3P water molecules using AutoDock energy scoring |
| Flask | Serves the static app + exposes the `/hydrate` endpoint |
