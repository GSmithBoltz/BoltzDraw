# Diversity Picking in BoltzSpace — Engineering Handoff

**What this covers:** the two diversity-selection methods now in the BoltzSpace tab, the science behind
them, and exactly which RDKit code they correspond to.

**The headline for engineers:** the RDKit *Python* library implements both methods. The RDKit *JavaScript*
build we load in the browser does **not** — it ships fingerprints only. So BoltzSpace reimplements both
algorithms in JS, validated line-for-line against the Python originals. See
[Why we reimplemented](#why-we-reimplemented-them) for the consequences.

---

## 1. What a diversity pick is, and why it exists

A virtual screen hands you a scored list — here, 1000 molecules with a Boltz binding probability. You can
only synthesise or order a few dozen. The naive move is to take the top 24 by score.

That's usually a bad pick. Generative and docking pipelines produce **families of near-identical
molecules**. The top 24 by score are frequently 24 variations on the same scaffold, differing by a methyl
here and a fluorine there. If that scaffold is a scoring-function artefact — and it often is — all 24 fail
together, and you've learned one thing from 24 experiments.

A diversity pick spends those 24 slots on 24 *different* chemotypes. If the model is right about some and
wrong about others, you find out. It converts a single bet into a portfolio.

### Order of operations: filter, then diversify

The cascade is **hard filters → score cut → diversity pick**, always in that order.

Diversity selection runs last because it's the only step whose output depends on the *composition* of the
set it's given. Run it across all 1000 molecules and it will faithfully sample the dead regions of chemical
space alongside the good ones — you'd be spending picks on chemistry the model already told you it doesn't
like. Diversity is a tiebreaker among molecules you already believe in, not a substitute for believing in
them.

The filter is optional in the UI (some users legitimately want to map the whole library), but it defaults
to on, and the on-path is the one to recommend.

### Cut by percentile, not by absolute score

The UI offers both, and defaults to percentile. This is deliberate.

Boltz-2's binding probability is a **classifier output, not a calibrated probability**, and its scale shifts
between targets. On the ROR1 set the values span 0.304–0.804 with a median of 0.356. A threshold of `≥ 0.7`
leaves 10 molecules — too few to diversify. `≥ 0.5` leaves 100, which is the top decile. Neither number
means anything on a different target. "Top 10%" transfers; "≥ 0.5" does not.

Top 10–15% is the conventional retained fraction from a virtual screen.

### Do not pick from the UMAP or t-SNE coordinates

This is the failure mode the UI invites, because the projection is sitting right there and 2-D Euclidean
distance *looks* like it should work.

It doesn't. **t-SNE does not preserve global distances at all** — the gap between two clusters on a t-SNE
plot carries essentially no information, and apparent cluster sizes are artefacts of the perplexity setting.
UMAP preserves local topology and somewhat more global structure, but its embedded distances still aren't a
metric you can select against. A pick made by grabbing widely-separated points off either plot is selecting
on projection artefacts, and it won't reproduce when you reseed the embedding.

Both pickers therefore run in the **original 2048-bit ECFP4 fingerprint space**, on Tanimoto distances. The
plots then colour the result. Visualisation for verification; fingerprint space for the decision.

---

## 2. The two methods

Both consume the same input: a set of molecules, their ECFP4 fingerprints, and a Tanimoto distance
`d(i,j) = 1 − Tanimoto(fp_i, fp_j)`.

### Butina (sphere exclusion) — the default

> Butina, D. *Unsupervised Database Clustering Based on Daylight's Fingerprint and Tanimoto Similarity: A
> Fast and Automated Way to Cluster Small and Large Data Sets.* J. Chem. Inf. Comput. Sci. **1999**, 39,
> 747–750.

A leader–follower clustering. Two molecules are "neighbours" if their Tanimoto similarity is at or above a
cutoff. The molecule with the most neighbours seeds the first cluster and absorbs all of them; the densest
remaining unassigned molecule seeds the next; repeat until everything is assigned. Molecules with no
neighbours end up as singleton clusters.

Then you take **one representative per cluster** — in BoltzSpace, the *best-scoring* member, not the
geometric centre. That last detail is what makes it a hit-picking method rather than a summarisation
method: you keep the strongest compound from each chemotype.

This is the standard industrial cherry-pick. It's the right default. Its quirk is that **you don't control
how many molecules you get** — the cutoff determines the cluster count, and the cluster count determines
your pick size. (BoltzSpace's simple mode papers over this; see §4.)

*Best for:* one representative per scaffold, when covering chemotypes matters more than hitting an exact
count.

### MaxMin (maximum dissimilarity)

> Ashton, M. et al. *Identification of Diverse Database Subsets using Property-Based and Fragment-Based
> Molecular Descriptions.* Quant. Struct.-Act. Relat. **2002**, 21, 598–604.

A greedy algorithm. Start from one seed molecule. Then repeatedly add whichever unpicked molecule has the
largest *minimum* distance to everything already picked — the molecule furthest from the current selection.
Stop at N.

You get **exactly N molecules**, guaranteed spread. The tradeoff is that it loves the edges of the space:
it will happily hand you singletons, oddballs, and whatever weird thing sits furthest from everything else.
On an unfiltered set that's a liability. On a pre-filtered set of things you already believe in, it's fine.

BoltzSpace seeds it with the **top-scoring survivor** by default, so your best compound is always pick #1.
RDKit's own default seeds randomly.

*Best for:* filling an exact number of wells or plate positions with maximum spread.

---

## 3. What code in RDKit does this

Verified against RDKit **2026.03.3**.

### Fingerprints

| Layer | Call |
|---|---|
| Python (modern) | `rdkit.Chem.rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048)` |
| Python (legacy, now deprecation-warns) | `rdkit.Chem.rdMolDescriptors.GetMorganFingerprintAsBitVect(mol, 2, 2048)` |
| JavaScript (MinimalLib) | `mol.get_morgan_fp(JSON.stringify({radius: 2, nBits: 2048}))` |

Radius 2 over 2048 bits is ECFP4. Tanimoto between two of these is
`rdkit.DataStructs.TanimotoSimilarity` (or `BulkTanimotoSimilarity` for one-against-many).

### Butina

```
rdkit.ML.Cluster.Butina.ClusterData(
    data, nPts, distThresh, isDistData=False, distFunc=EuclideanDist, reordering=False)
```

- **Source:** `rdkit/ML/Cluster/Butina.py` — this one is **pure Python**, not a C++ extension. You can read
  the whole algorithm in about 40 lines, which is why we could port it faithfully.
- `data` is the condensed lower-triangle distance list (`d(1,0), d(2,0), d(2,1), d(3,0), …`) when
  `isDistData=True`. `distThresh` is a **distance**, so Tanimoto similarity 0.65 is `distThresh=0.35`.
- `reordering=True` recomputes neighbour counts after each cluster forms, so the next seed is the molecule
  with the most *still-unassigned* neighbours. It's slower and gives different clusters. **BoltzSpace uses
  the default (`reordering=False`)**; if you turn this on server-side, the JS and Python will diverge.
- Returns a tuple of tuples. **The first element of each cluster is its centroid.**

### MaxMin

```
rdkit.SimDivFilters.rdSimDivPickers.MaxMinPicker()
    .LazyPick(distFunc, poolSize, pickSize, firstPicks=[...], seed=-1)
```

- **Source:** `rdkit/SimDivFilters/rdSimDivPickers.so` — a **compiled C++ extension** (`Code/SimDivPickers/`
  in the RDKit tree). You cannot read it from Python.
- `LazyPick` calls your `distFunc(i, j)` on demand and never materialises the full distance matrix — this is
  why MaxMin scales to very large pools. `LazyBitVectorPick` is the faster path when you hand it fingerprints
  directly rather than a callback.
- `firstPicks` seeds the selection. Pass `[index_of_top_scorer]` to reproduce BoltzSpace's default.
- Also on that module: `LeaderPicker` (a fast sphere-exclusion variant, C++, worth knowing as the scalable
  alternative to Butina) and `HierarchicalClusterPicker`.

### The exact reference calls we validated against

```python
from rdkit.ML.Cluster import Butina
from rdkit.SimDivFilters import rdSimDivPickers

clusters = Butina.ClusterData(condensed_dists, n, 0.35, isDistData=True)   # Tanimoto 0.65
picks    = rdSimDivPickers.MaxMinPicker().LazyPick(dist_fn, n, 24, firstPicks=[top_scorer])
```

---

## 4. Why we reimplemented them

The browser loads RDKit MinimalLib from
`@rdkit/rdkit@2025.3.4-1.0.0/dist/RDKit_minimal.{js,wasm}`.

I checked the 6.6 MB WASM binary directly. It contains `get_morgan_fp` and `get_svg`. It contains **no**
`Butina`, `ClusterData`, `MaxMinPicker`, or `LazyPick` symbols. MinimalLib exposes molecule parsing,
descriptors, fingerprints and drawing — the clustering and picking modules are not in the build.

So BoltzSpace computes fingerprints with RDKit-in-WASM and then runs both algorithms in hand-written JS, in
[`js/boltzspace.js`](js/boltzspace.js):

| Function | Line | Role |
|---|---|---|
| `_packFp` | 892 | 2048 bits → 64×`Uint32` words + popcount |
| `_tanimoto` | 901 | bitwise-AND popcount over the packed words |
| `_condensedDistances` | 917 | upper-triangle `Float32Array`, same layout as RDKit's condensed list |
| `_butina` | 927 | the port of `ClusterData` |
| `_butinaAutoCutoff` | 953 | bisects the cutoff to hit a requested pick count |
| `_maxMin` | 966 | the port of `LazyPick` |
| `runDiversityPick` | 1146 | filter → cluster/pick → paint the plot selection |

**If you ever move this server-side, delete the JS and call RDKit Python directly.** The port exists only
because the WASM build lacks the modules; it is not an improvement on RDKit and should not outlive the need.

### Two gotchas the port surfaced

**1. Butina's tie-break is not arbitrary.** RDKit sorts `(neighbour_count, index)` tuples with
`sort(reverse=True)`, which breaks ties on **descending index**. Our first port broke them ascending. On the
ROR1 set that single difference produced **82 clusters instead of 81**, with different membership and a
different representative list. If you reimplement this anywhere else, match the tie-break or your results
will silently disagree with everyone else's.

Note also that RDKit's neighbour lists include the molecule itself (`dist_matrix[i][i] = 0 ≤ threshold`), so
its neighbour counts are one higher than the intuitive value. It doesn't change the ordering, but it does
change the `> 1` guard that separates real clusters from singletons.

**2. Unscorable rows.** When the filter is on, rows whose score doesn't parse are dropped before the picker
sees them. With the filter *off* they reach the picker, and `NaN` poisons naive comparisons: `NaN > x` is
always false, and `a - b` returns `NaN`, which makes `Array.sort` behave unpredictably. Both comparators are
now NaN-safe — an unscorable molecule sorts last and never wins a comparison. Anyone porting this needs the
same guard; Python's `max()` has the identical trap.

### Auto-cutoff (simple mode)

Butina's cutoff, not the user, decides how many molecules you get. That's a bad first-run experience, so
simple mode inverts it: the user says "give me 24," and we **bisect the Tanimoto cutoff** until the cluster
count lands on 24. Cluster count rises monotonically with the similarity cutoff, so 18 bisection steps are
enough. If it overshoots by a cluster or two, the weakest-scoring representatives are dropped to hit the
number exactly, and the status line says so.

Advanced mode exposes the raw cutoff and lets the cluster count fall where it may.

**Watch the reported `Tc=` value.** It's in the methods bar after every run. If bisection had to push the
cutoff below about 0.3 to reach your pick count, the resulting "clusters" are not chemotypes in any
meaningful sense, and MaxMin is the more honest tool for that dataset.

---

## 5. Verification

The JS was cross-validated against RDKit Python on the real ROR1 candidate set (1000 molecules, filtered to
the top 10% by binding probability = 100 survivors), using identical fingerprints and cutoffs:

| Check | Result |
|---|---|
| Butina cluster count @ Tanimoto 0.65 | 81 vs RDKit's 81 |
| Butina cluster size distribution | exact match |
| Butina ranked representative list | exact match |
| MaxMin 24 picks, seeded on top scorer | exact match, in order |
| Auto-cutoff bisection (targets 12/24/50) | lands exactly on target |
| NaN-safe comparators | 11/11 assertions pass |

Performance envelope (main thread, no worker):

| Pool | Distance matrix | Butina auto-sweep | MaxMin (50 picks) |
|---|---|---|---|
| 1,000 | 32 ms | 15 ms | 4 ms |
| 3,000 | 265 ms | 116 ms | 9 ms |
| 20,000 | — | — | 65 ms |

Butina is capped at **3,000 molecules** (`BUTINA_MAX`, line 29) because it needs the full pairwise matrix.
MaxMin never builds one, so it stays usable far past that ceiling — above 3,000 the UI routes users to
MaxMin.

---

## 6. One empirical finding about this dataset

Worth knowing before anyone trusts a cluster count from a generated library.

The ROR1 top decile — 100 molecules — splits into **81 Butina clusters** at the conventional Tanimoto 0.65
cutoff. The largest cluster has 3 members. Nearly everything is a singleton.

This library is already close to maximally diverse. At the standard "same chemotype" threshold the
clustering does essentially no work: "one representative per cluster" would hand back 81 of the 100
molecules. To reach 24 clusters, bisection has to loosen the cutoff to **Tc ≈ 0.22**, which is a very
generous definition of "same chemotype."

Two implications:

1. On sets like this, **MaxMin is the more defensible choice** — it doesn't pretend chemotypes exist where
   they don't.
2. **24 may simply be too few.** Picking 40–60 would let you stay near a defensible cutoff instead of
   forcing the algorithm to invent clusters.

Whether this generalises to other generated libraries is an open question and worth checking on a second
target before treating it as a rule.

---

## 7. Known gaps / possible next steps

- **Two score columns, treated as one.** On the ROR1 file, Binding Probability and Optimization Score are
  nearly independent (Spearman ρ = 0.17). The UI makes you choose one to rank by. Taking the **Pareto front**
  over both would keep the tradeoff instead of discarding a column, and is cheap to compute.
- **No scaffold reporting.** Bemis–Murcko scaffold counts (`rdkit.Chem.Scaffolds.MurckoScaffold`) are coarse
  — one atom change flips a scaffold — but they're interpretable, and they're the number a medicinal chemist
  will ask for. Worth adding as a QC readout on the final pick, not as the picking axis.
- **No structural-alert filter.** The ROR1 CSV carries an `Alert Level` column (987 `None`, 13 `Minor`).
  Nothing in BoltzSpace reads it. A generic "drop rows where column X equals Y" pre-filter would cover it.
- **Butina's 3,000-molecule ceiling.** If this becomes a real constraint, RDKit's `LeaderPicker` is a fast
  C++ sphere-exclusion that scales much further, and is the natural upgrade path — but only once this runs
  server-side, since it isn't in the WASM build either.
