# BoltzTriage — the workflow

**What this is:** a prototype triage tab in BoltzDraw. It is the PLIP tiering workflow that currently
lives in one notebook per target, reduced to two steps and one decision.

**Status:** prototype. The one demo screen — **CRBN allosteric** — ships **synthetic poses**; its pocket
definition is the real one, from `Internal Research/CRBN_allosteric/`. Nothing the demo prints is a
result. Real PLIP output loads through the same path (see [Loading real data](#loading-real-data)).

---

## The whole scheme

A pose is judged on two things.

| | **adds a new polar interaction** | **adds none** |
|---|---|---|
| **maintains ≥ 50%** of the standard's value | **Tier 1** — conserved + extended | **Tier 2** — conserved |
| **maintains < 50%** | **Tier 3** — partial + extended | **Tier 4** — partial |

Plus two things that are not tiers and cannot be:

- **Off-site** — the pose is in a different pocket. Wrong, not weak. A degron-pocket pose maintains 0%
  and adds ten new contacts, which the 2×2 would read as "partial but extended".
- **Not assessable** — the pocket is not in that pose's crop, so there is nothing to maintain.
  Unscored, not bottom-tier.

### Interaction value is read, not assigned

PLIP already says which contact is which, so the UI does not ask again:

| | value | |
|---|---|---|
| H-bond | 1.00 | |
| π-stacking | 1.00 | |
| π-cation | 0.90 | |
| salt bridge | 0.90 | cheap to form — flagged on the board |
| halogen bond | 0.80 | |
| water bridge | 0.60 | |
| hydrophobic | 0.35 | |

A directional polar contact is worth about three hydrophobic ones, because it is the contact a chemist
can build on and hydrophobic surface is what every greasy molecule makes everywhere.

Right residue, different chemistry earns **40%** — a pose that reaches Ser339 but packs against it
instead of donating an H-bond is in the right place with the wrong group, which is a medicinal-chemistry
lead, not a failure.

### The one decision: star the key interactions

A starred interaction is worth **3×**. That raises both the pose's score when it makes the interaction
and the bar when it misses it, so one star moves the whole board. In the CRBN demo, starring the Arg283
H-bond moves 18 poses into Tier 1 and 9 into Tier 2, and names the miss on every card that skips it.

Nothing else is configurable. Star nothing and the automatic weighting stands on its own.

---

## The two arms

### Arm A — a reference is bound

Something sits in the site in a structure you trust. PLIP it; its footprint is the standard. Each
interaction arrives with its geometry (both H-bond distances — see below), its value, and how often the
library makes the same interaction.

### Arm B — nothing is bound

The interactions the library makes most often *are* the site, so they are the standard. An interaction
joins it at **≥8% of poses when polar, ≥20% when hydrophobic**: a hydrophobic contact made by a third of
the library says little, while an H-bond made by a tenth of it is a real feature of the pocket.

On the CRBN demo this recovers **10 of the 11 real pocket residues** — the ones the notebook derived
from 9SFM — from frequency and interaction type alone, with no reference and no clustering. That is the
argument for the simplification: the earlier version got there by clustering poses into binding modes
behind three sliders, and the frequency cut does the same job in one line.

Read the result as a hypothesis, not a standard. Poses agreeing on something means the predictor agrees
with itself.

Both arms produce the same object — a list of (residue, interaction) pairs with a value — so everything
after step 1 is identical.

---

## Why "new" counts polar only

Total new contacts scale with molecular weight. The SS18-SSX audit found its composite score was
essentially a contact count (Spearman 0.95), which is how two real hits ended up mid-table. Counting all
new contacts would put the largest molecules in the library in Tier 1 by construction.

Counting only new *polar* interactions takes the left–right split off that axis without any size
correction, percentile or adjustment machinery. The board prints all three correlations when the top of
the list skews large, so you can see it working: on the CRBN demo, ρ(MW, maintained) ≈ 0.3,
ρ(MW, new polar) ≈ 0.2, and ρ(MW, all new contacts) would be ≈ 0.5.

---

## The board

Six lanes: Tier 1–4, Off-site, Not assessable. The 2×2 sits above them with live counts and doubles as
the tier filter. Three views — **lanes**, a dense **table**, and the **interaction map** (pose ×
standard interaction, where "right residue, wrong chemistry" and "not in the crop" are drawn differently
from "missed").

Every card carries its maintained percentage, its new-polar count, its MW, its Boltz probability and its
flags. Flags are filters, and all of them are automatic:

| Flag | What it means |
|---|---|
| Off-site | In a second pocket, none of the standard |
| Spans both sites | Touches both — check the pose before believing either |
| Misses *n* key | Starred interactions missed, by name (only for poses maintaining ≥25%) |
| Anion → Arg/Lys | Anionic ligand salt-bridging a cationic residue — cheap, weakly discriminating |
| Salt-bridge driven | Over 40% of the maintained value is salt bridges to Arg/Lys/His |
| Right residues, wrong chemistry | Reaches the standard's residues, makes a different interaction at every one |
| Low-confidence pose | Boltz probability under 0.35 in Tier 1 or 2 — "mechanism unknown", not "on-mechanism" |

The inspector states **why this tier** in words and itemises the maintained, partial, missed and new
interactions with their value contributions, so the polar-versus-hydrophobic arithmetic is visible per
pose. Decisions — pick / hold / reject with a reason — export with the full configuration in the CSV
header. Ranks print as ranges, because these scores tie in steps.

### Two numbers on the standard page worth reading

- **Residues the best pose reaches** — of 520 CRBN poses, the best reaches 9 of 11. If the best pose in
  a library of hundreds reached a third of the standard, the standard would not be satisfiable by one
  molecule, whether because it spans two pockets or because it came from a peptide.
- **Reached by the median pose** — 27% here. The gap between these two is the room the tiers work in.

(A co-occurrence graph over the standard's residues was the first attempt at detecting a two-pocket
standard. It stayed connected through one surface arginine that both pockets touch, and tuning the edge
threshold until it separated them would have been fitting the detector to the demo data. The reach
numbers are the honest version: one statistic, nothing to tune, and it covers the peptide case too.)

### One correction carried over from the notebooks

PLIP's `distance_ah` is the hydrogen–acceptor distance, not the donor–acceptor one. The shared notebook
helper prints it as `D-A`, which under-reports every H-bond by about 0.85 Å — that is how CRBN's 2.71 Å
Arg283 contact came to be quoted as 1.87 Å. The standard page prints both, labelled.

---

## Loading real data

`.jsonl`, one pose per line — the shape `plip_all/run_plip_all.py` already writes:

```json
{"id": "SM-7UMEP83T", "smiles": "...", "mw": 441.5, "prob": 0.41,
 "contacts": [{"chain": "U", "resnr": 44, "restype": "ILE", "itype": "hydrophobic", "dist": 3.8}]}
```

or a long CSV with `id,chain,resnr,restype,itype` and optional `distance`, `smiles`, `mw`, `prob`.
`smiles` draws the structure cards and `prob` enables the low-confidence flag. An optional `crop` array
(the standard's residues that the pose's crop contains, from the sequence alignment) is what separates
*not assessable* from Tier 4; without it the UI falls back to "contacted nothing in the standard".

An uploaded screen has no second pocket defined, so nothing is called off-site until one is supplied —
in the demo it comes from the reference structure's other ligand.

## Where the code is

| File | What |
|---|---|
| `js/triagescore.js` | The engine — interaction families, both standards, scoring, tiers, flags, import, export. No DOM; this is the part that would move server-side or back into a notebook unchanged. |
| `js/boltztriage.js` | The UI — two steps, the 2×2, the board, the inspector. |
| `js/triagedemo.js` | The CRBN demo screen and the synthetic pose generator. Delete when real data replaces it. |
| `css/boltztriage.css` | Styles. |

## What a real version would need

- PLIP on the server (`server.py` already has `/plip`) so a Boltz zip can be profiled
  in place instead of pre-processed in a notebook.
- The sequence alignment that maps a cropped, renumbered Boltz receptor onto reference numbering. Every
  notebook does this with `build_resnr_map`; the `crop` field is how that information arrives here.
- The 3D pose beside the card — Mol* is already in the app.
- Assay data alongside the tier. A tier is a mechanism annotation, not a potency ranking: against the
  real CRHR2 data the on-mechanism actives sat in the middle tiers, the "pocket touched, anchor missed"
  bin was the modal one and held actives by other means, and only the bottom was genuinely depleted.

## The notebooks this replaces

| Notebook | Target | What it did that the 2×2 now does automatically |
|---|---|---|
| `External/PLIPS/PLIP_tiering_jupyter.ipynb` | CRHR2 | KEY/WALL/REACH/DEEP residue sets, hand-assigned |
| `~/Downloads/GLP2R_tiering_colab.ipynb` | GLP-2R | three depth zones, gate on one anchor → star it instead |
| `Internal Research/CRBN_allosteric/` | CRBN | breadth of key contact + a degron off-site flag |
| `~/Downloads/STAT4_polar_pocket_jupyter.ipynb` | STAT4 | eleven polar residues typed in by hand → the ≥8% polar cut finds them |
| `Kadoch_ADME/5_hit_analysis/notebooks/01–03` | SS18-SSX | specificity weights, percentile cuts, MW deciles → new-polar-only |
