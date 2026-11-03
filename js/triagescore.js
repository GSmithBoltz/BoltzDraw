/**
 * triagescore.js
 *
 * The scoring engine behind BoltzTriage. Pure functions, no DOM — this is the
 * part that would move server-side (or back into a notebook) unchanged.
 *
 * THE WHOLE SCHEME
 * ----------------
 * A pose is judged on two things, and nothing else:
 *
 *   maintained   how much of the standard's interaction value it reproduces
 *   new polar    whether it makes a polar interaction the standard does not
 *
 *   Tier 1  ≥ half the standard  + new polar     conserved and extended
 *   Tier 2  ≥ half the standard,  no new polar   conserved
 *   Tier 3  < half the standard  + new polar     partial, extended elsewhere
 *   Tier 4  < half the standard,  no new polar
 *
 * Interaction value is read off the PLIP call, not assigned by hand: an H-bond,
 * a pi-stack or a pi-cation is worth about three hydrophobic contacts, because
 * directional polar contacts are what a medicinal chemist can build on and
 * hydrophobic surface is what every greasy molecule makes everywhere. The only
 * input is which interactions to star as **key**, which multiplies their value.
 *
 * "New" counts only *polar* interactions on purpose. Total new contacts scale
 * with molecular weight — the SS18-SSX audit found its composite score was
 * essentially a contact count (Spearman 0.95) — so rewarding new contacts
 * wholesale fills the top tier with the largest molecules. A new H-bond is an
 * opportunity; a new hydrophobic contact is usually just more surface.
 *
 * TWO THINGS THAT ARE NOT TIERS
 * -----------------------------
 *   off-site        the pose is in a different pocket. Wrong, not weak. A
 *                   degron-pocket pose scores 0% maintained with ten new
 *                   contacts, which the 2×2 would call "partial but extended".
 *   not assessable  the pocket is not in that pose's crop, so there is nothing
 *                   to maintain. Unscored, not bottom-tier.
 */

// ----------------------------------------------------------------
// Interaction families — the automatic part
// ----------------------------------------------------------------

export const FAMILY = {
    hbond:       { label: 'H-bond',       short: 'H-bond', polar: true,  value: 1.00 },
    pistack:     { label: 'π-stacking',   short: 'π-stack', polar: true, value: 1.00 },
    pication:    { label: 'π-cation',     short: 'π-cat',  polar: true,  value: 0.90 },
    saltbridge:  { label: 'salt bridge',  short: 'salt',   polar: true,  value: 0.90, cheap: true },
    halogen:     { label: 'halogen bond', short: 'halogen', polar: true, value: 0.80 },
    waterbridge: { label: 'water bridge', short: 'water',  polar: true,  value: 0.60 },
    hydrophobic: { label: 'hydrophobic',  short: 'phobic', polar: false, value: 0.35 },
};

export const KEY_MULTIPLIER = 3;
export const MAINTAINED_CUT = 0.5;      // half the standard's value — the T1/T2 ÷ T3/T4 line
const PARTIAL_CREDIT = 0.4;             // right residue, different chemistry

/** PLIP's interaction strings, normalised to a family. */
export function family(itype) {
    const t = String(itype).toLowerCase();
    if (t.startsWith('h-bond') || t.startsWith('hbond')) return 'hbond';
    if (t.startsWith('salt'))        return 'saltbridge';
    if (t.startsWith('pi-stack') || t.startsWith('pistack')) return 'pistack';
    if (t.startsWith('pi-cation') || t.startsWith('pication')) return 'pication';
    if (t.startsWith('halogen'))     return 'halogen';
    if (t.startsWith('water'))       return 'waterbridge';
    if (t.startsWith('hydrophobic')) return 'hydrophobic';
    return 'hydrophobic';               // an unrecognised call is treated as the cheap case
}

export const isPolar = (fam) => !!(FAMILY[fam] && FAMILY[fam].polar);
export const resKey  = (c) => `${c.chain}:${c.resnr}`;
const pairKey = (res, fam) => `${res}|${fam}`;

// ----------------------------------------------------------------
// Indexing — one pass over the screen, reused by every later step
// ----------------------------------------------------------------

export function indexScreen(screen) {
    const resFreq  = new Map();          // residue        -> poses contacting it
    const pairFreq = new Map();          // residue|family -> poses making it
    const restypes = { ...(screen.restypes || {}) };
    const famTally = new Map();

    const poses = screen.poses.map(p => {
        const resSet = new Set();
        const pairSet = new Set();
        const byRes = new Map();
        for (const c of p.contacts) {
            const k = resKey(c);
            const fam = family(c.itype);
            resSet.add(k);
            pairSet.add(pairKey(k, fam));
            if (!restypes[k]) restypes[k] = c.restype;
            if (!byRes.has(k)) byRes.set(k, []);
            byRes.get(k).push({ ...c, fam });
            famTally.set(fam, (famTally.get(fam) || 0) + 1);
        }
        for (const k of resSet)  resFreq.set(k, (resFreq.get(k) || 0) + 1);
        for (const k of pairSet) pairFreq.set(k, (pairFreq.get(k) || 0) + 1);
        return {
            ...p, resSet, pairSet, byRes,
            nContacts: p.contacts.length,
            crop: p.crop ? new Set(p.crop) : null,
        };
    });

    return {
        N: poses.length, poses, resFreq, pairFreq, restypes,
        families: [...famTally.entries()].sort((a, b) => b[1] - a[1]),
        residues: [...resFreq.keys()].sort(_resSort),
    };
}

function _resSort(a, b) {
    const [ca, na] = a.split(':'), [cb, nb] = b.split(':');
    return ca === cb ? (+na) - (+nb) : ca < cb ? -1 : 1;
}

// ----------------------------------------------------------------
// The standard
//
// A list of {res, fam} pairs with a value. Built one of two ways, and the
// result has the same shape either way, so everything downstream is identical.
// ----------------------------------------------------------------

/** From a reference's own PLIP contacts. */
export function standardFromReference(contacts, index) {
    const seen = new Map();
    for (const c of contacts) {
        const res = resKey(c), fam = family(c.itype);
        const k = pairKey(res, fam);
        if (!seen.has(k)) {
            seen.set(k, {
                res, fam, key: false,
                restype: c.restype || (index && index.restypes[res]) || '???',
                detail: c,
                libFrac: index && index.pairFreq.has(k) ? index.pairFreq.get(k) / index.N : 0,
            });
        }
    }
    return _orderStandard([...seen.values()]);
}

/**
 * From the library itself, when nothing is bound.
 *
 * The frequent interactions are the site, so they are the standard. Polar pairs
 * come in at a lower frequency bar than hydrophobic ones: a hydrophobic contact
 * made by a third of the library says little, while an H-bond made by a tenth of
 * it is a real feature of the pocket.
 */
export function standardFromLibrary(index, { polarMin = 0.08, phobicMin = 0.20, cap = 16 } = {}) {
    const rows = [...index.pairFreq.entries()].map(([k, n]) => {
        const [res, fam] = k.split('|');
        return { res, fam, n, frac: n / index.N, restype: index.restypes[res] || '???' };
    });
    const kept = rows
        .filter(r => r.frac >= (isPolar(r.fam) ? polarMin : phobicMin))
        .sort((a, b) => (isPolar(b.fam) - isPolar(a.fam)) || (b.frac - a.frac))
        .slice(0, cap)
        .map(r => ({ res: r.res, fam: r.fam, key: false, restype: r.restype, libFrac: r.frac, detail: null }));
    return _orderStandard(kept);
}

function _orderStandard(list) {
    return list.sort((a, b) =>
        (isPolar(b.fam) - isPolar(a.fam)) ||
        (FAMILY[b.fam].value - FAMILY[a.fam].value) ||
        _resSort(a.res, b.res));
}

export const pairValue = (p) => FAMILY[p.fam].value * (p.key ? KEY_MULTIPLIER : 1);
export const standardValue = (std) => std.reduce((a, p) => a + pairValue(p), 0);

/**
 * Can any one molecule actually satisfy this standard?
 *
 * The directly checkable version of "are these interactions all in one pocket?".
 * If the best pose in a library of hundreds reaches only a third of the
 * standard's residues, the standard is not reachable by a single molecule —
 * whether because it spans two sites or because it came from something much
 * larger, like a peptide. One number, nothing to tune, and it answers both.
 *
 * (A co-occurrence graph over the standard's residues was the first attempt. It
 * stayed connected through one surface arginine that both pockets touch, and
 * tuning the edge threshold until it separated them would have been fitting the
 * detector to the demo data. This is the honest version.)
 */
export function reachCheck(std, index) {
    const residues = [...new Set(std.map(p => p.res))];
    if (!residues.length || !index.N) return null;
    let best = 0, bestId = null;
    const fracs = [];
    for (const p of index.poses) {
        const n = residues.reduce((a, r) => a + (p.resSet.has(r) ? 1 : 0), 0);
        fracs.push(n / residues.length);
        if (n > best) { best = n; bestId = p.id; }
    }
    fracs.sort((a, b) => a - b);
    return {
        nResidues: residues.length,
        bestCount: best,
        bestFrac: best / residues.length,
        bestId,
        medianFrac: fracs[Math.floor(fracs.length / 2)],
    };
}

// ----------------------------------------------------------------
// Survey — what the library touches, before any standard exists
// ----------------------------------------------------------------

export function survey(index) {
    const rows = [...index.pairFreq.entries()].map(([k, n]) => {
        const [res, fam] = k.split('|');
        return { key: k, res, fam, restype: index.restypes[res] || '???', n, frac: n / index.N };
    }).sort((a, b) => (isPolar(b.fam) - isPolar(a.fam)) || (b.frac - a.frac));

    const mw = index.poses.map(p => p.mw).filter(v => v != null);
    const nres = index.poses.map(p => p.resSet.size);
    const prob = index.poses.map(p => p.prob).filter(v => v != null);
    const polarPerPose = index.poses.map(p =>
        [...p.pairSet].filter(k => isPolar(k.split('|')[1])).length);

    return {
        rows,
        polarRows: rows.filter(r => isPolar(r.fam)),
        sizeConfound: mw.length === index.N ? spearman(mw, nres) : null,
        meanContacts: nres.reduce((a, b) => a + b, 0) / Math.max(nres.length, 1),
        meanPolar: polarPerPose.reduce((a, b) => a + b, 0) / Math.max(polarPerPose.length, 1),
        mwRange: mw.length ? [Math.min(...mw), Math.max(...mw)] : null,
        probRange: prob.length ? [Math.min(...prob), Math.max(...prob)] : null,
        families: index.families,
    };
}

export function spearman(x, y) {
    const n = x.length;
    if (n < 3 || y.length !== n) return null;
    const rank = (v) => {
        const idx = v.map((val, i) => [val, i]).sort((a, b) => a[0] - b[0]);
        const r = new Array(n);
        let i = 0;
        while (i < n) {
            let j = i;
            while (j + 1 < n && idx[j + 1][0] === idx[i][0]) j++;
            const avg = (i + j) / 2 + 1;
            for (let k = i; k <= j; k++) r[idx[k][1]] = avg;
            i = j + 1;
        }
        return r;
    };
    const rx = rank(x), ry = rank(y);
    const mx = rx.reduce((a, b) => a + b, 0) / n, my = ry.reduce((a, b) => a + b, 0) / n;
    let num = 0, dx = 0, dy = 0;
    for (let i = 0; i < n; i++) {
        const a = rx[i] - mx, b = ry[i] - my;
        num += a * b; dx += a * a; dy += b * b;
    }
    return dx && dy ? num / Math.sqrt(dx * dy) : 0;
}

// ----------------------------------------------------------------
// Scoring
// ----------------------------------------------------------------

const CATIONIC = new Set(['ARG', 'LYS', 'HIS']);

export function scoreScreen(screen, index, std, offSiteResidues = []) {
    const stdRes = new Set(std.map(p => p.res));
    const offSite = new Set(offSiteResidues.filter(r => !stdRes.has(r)));
    const total = standardValue(std) || 1;

    const rows = index.poses.map(p => {
        let maintained = 0;
        const hitExact = [], hitPartial = [], missed = [];

        for (const sp of std) {
            const cs = p.byRes.get(sp.res);
            if (!cs) { missed.push(sp); continue; }
            if (cs.some(c => c.fam === sp.fam)) {
                maintained += pairValue(sp);
                hitExact.push(sp);
            } else {
                maintained += pairValue(sp) * PARTIAL_CREDIT;
                hitPartial.push({ ...sp, insteadOf: cs[0].fam });
            }
        }

        // New interactions: residues outside the standard. Polar ones are the
        // ones that count; the rest are reported.
        const newPolar = [], newPhobic = [], offHit = [];
        for (const [res, cs] of p.byRes) {
            if (stdRes.has(res)) continue;
            if (offSite.has(res)) { offHit.push(res); continue; }
            const fams = [...new Set(cs.map(c => c.fam))];
            if (fams.some(isPolar)) newPolar.push({ res, fams: fams.filter(isPolar) });
            else newPhobic.push({ res, fams });
        }

        const frac = maintained / total;
        const inCrop = p.crop ? [...stdRes].filter(r => p.crop.has(r)).length : null;
        const assessable = inCrop == null ? (hitExact.length + hitPartial.length > 0 || offHit.length > 0) : inCrop > 0;
        const isOffSite = offHit.length > 0 && hitExact.length === 0;

        // Artefact read-outs. Automatic, not configurable — a salt bridge to a
        // surface arginine is cheap, and a score resting on one should say so.
        let saltValue = 0;
        for (const sp of hitExact) {
            if (sp.fam === 'saltbridge' && CATIONIC.has((index.restypes[sp.res] || '').toUpperCase())) {
                saltValue += pairValue(sp);
            }
        }
        const anionSalt = !!p.anion && [...p.byRes.entries()].some(([res, cs]) =>
            cs.some(c => c.fam === 'saltbridge') && CATIONIC.has((index.restypes[res] || '').toUpperCase()));

        let tier, why;
        if (!assessable) {
            tier = 'NA';
            why = 'the pocket is not in this pose\'s crop, so there is nothing to maintain';
        } else if (isOffSite) {
            tier = 'OFF';
            why = `in a different pocket — ${offHit.length} off-site residue(s), none of the standard`;
        } else if (frac >= MAINTAINED_CUT) {
            tier = newPolar.length ? 1 : 2;
            why = newPolar.length
                ? `maintains ${Math.round(100 * frac)}% of the standard and adds ${newPolar.length} new polar interaction(s)`
                : `maintains ${Math.round(100 * frac)}% of the standard, adds no new polar interaction`;
        } else {
            tier = newPolar.length ? 3 : 4;
            why = newPolar.length
                ? `maintains only ${Math.round(100 * frac)}% of the standard, but adds ${newPolar.length} new polar interaction(s) elsewhere`
                : `maintains only ${Math.round(100 * frac)}% of the standard and adds no new polar interaction`;
        }

        return {
            id: p.id, smiles: p.smiles, mw: p.mw, prob: p.prob, anion: p.anion,
            maintained, maintainedFrac: frac, standardTotal: total,
            hitExact, hitPartial, missed,
            nExact: hitExact.length, nPartial: hitPartial.length, nStd: std.length,
            nKeyMissed: missed.filter(m => m.key).length,
            newPolar, newPhobic, offHit,
            nNewPolar: newPolar.length, nNewPhobic: newPhobic.length,
            nContacts: p.nContacts, nRes: p.resSet.size,
            saltShare: maintained > 0 ? saltValue / maintained : 0,
            anionSalt, inCrop, assessable, isOffSite,
            tier, why,
            _pose: p,
        };
    });

    _applyRanks(rows);
    _applyFlags(rows, screen, std);
    return { rows, std, total, stdRes, offSite };
}

const TIER_ORDER = { 1: 1, 2: 2, 3: 3, 4: 4, OFF: 5, NA: 6 };

function _applyRanks(rows) {
    const vals = rows.map(r => r.maintainedFrac).sort((a, b) => b - a);
    for (const r of rows) {
        // Tie-aware: these scores step, so a rank is a range, not a position.
        r.rankLo = 1 + vals.filter(v => v > r.maintainedFrac).length;
        r.rankHi = vals.filter(v => v >= r.maintainedFrac).length;
    }
    rows.sort((a, b) => TIER_ORDER[a.tier] - TIER_ORDER[b.tier]
        || b.maintainedFrac - a.maintainedFrac
        || b.nNewPolar - a.nNewPolar
        || a.id.localeCompare(b.id));
}

function _applyFlags(rows, screen, std) {
    const anionNote = screen.guards && screen.guards.anionNote;
    const keyPairs = std.filter(p => p.key).length;
    for (const r of rows) {
        const f = [];
        if (!r.assessable) f.push({ id: 'not-assessable', label: 'Not assessable', level: 'info',
            text: `No standard residue in this pose's crop${r.inCrop != null ? ` (0 of ${r.nStd})` : ''}. A tier here would say nothing about the compound.` });
        if (r.isOffSite) f.push({ id: 'off-site', label: 'Off-site', level: 'warn',
            text: `Contacts ${r.offHit.length} residue(s) of a second pocket and none of the standard. Wrong site, not weak binding.` });
        else if (r.offHit.length) f.push({ id: 'mixed-site', label: 'Spans both sites', level: 'warn',
            text: `Touches ${r.offHit.length} off-site residue(s) as well as the standard — check the pose before believing either.` });
        // Only worth saying for a pose that is otherwise in the pocket: "missed the
        // key interaction" is not news about a pose that maintains nothing.
        if (keyPairs && r.nKeyMissed && r.maintainedFrac >= 0.25) f.push({ id: 'key-missed', label: `Misses ${r.nKeyMissed} key`, level: 'warn',
            text: `Misses ${r.nKeyMissed} of the ${keyPairs} interaction(s) you starred as key: ${r.missed.filter(m => m.key).map(m => `${m.res} ${FAMILY[m.fam].label}`).join(', ')}.` });
        if (r.anionSalt) f.push({ id: 'anion-anchor', label: 'Anion → Arg/Lys', level: 'alert',
            text: anionNote || 'An anionic ligand salt-bridging a cationic residue is cheap to form and weakly discriminating.' });
        if (r.saltShare > 0.4) f.push({ id: 'salt-driven', label: 'Salt-bridge driven', level: 'warn',
            text: `${Math.round(100 * r.saltShare)}% of this pose's maintained value is salt bridges to Arg/Lys/His.` });
        if (r.nPartial && r.nExact === 0) f.push({ id: 'wrong-chemistry', label: 'Right residues, wrong chemistry', level: 'info',
            text: `Reaches ${r.nPartial} standard residue(s) but makes a different kind of interaction at every one.` });
        if (r.prob != null && r.prob < 0.35 && (r.tier === 1 || r.tier === 2)) f.push({ id: 'pale-pose', label: 'Low-confidence pose', level: 'info',
            text: `Boltz probability ${r.prob.toFixed(2)}. Read this as "mechanism unknown" rather than "on-mechanism" — a contact on a rim residue flips on a small pose error.` });
        r.flags = f;
    }
}

// ----------------------------------------------------------------
// Summaries
// ----------------------------------------------------------------

export function tierSummary(rows) {
    const t = { 1: 0, 2: 0, 3: 0, 4: 0, OFF: 0, NA: 0 };
    for (const r of rows) t[r.tier] = (t[r.tier] || 0) + 1;
    return t;
}

export function flagSummary(rows) {
    const t = new Map();
    for (const r of rows) for (const f of r.flags) {
        if (!t.has(f.id)) t.set(f.id, { id: f.id, label: f.label, level: f.level, n: 0 });
        t.get(f.id).n++;
    }
    return [...t.values()].sort((a, b) => b.n - a.n);
}

/**
 * Is the top of the list just the biggest molecules? Counting only new *polar*
 * interactions is what keeps this honest, so it is worth printing.
 */
export function sizeCheck(rows) {
    const withMw = rows.filter(r => r.mw != null);
    if (withMw.length < 40) return null;
    const med = (a) => { const s = [...a].sort((x, y) => x - y); return s[Math.floor(s.length / 2)]; };
    const top = [...withMw].sort((a, b) => b.maintainedFrac - a.maintainedFrac);
    const k = Math.max(5, Math.round(withMw.length * 0.05));
    return {
        libraryMedianMw: med(withMw.map(r => r.mw)),
        topMedianMw: med(top.slice(0, k).map(r => r.mw)),
        k,
        rhoMaintained: spearman(withMw.map(r => r.mw), withMw.map(r => r.maintainedFrac)),
        rhoNewPolar:   spearman(withMw.map(r => r.mw), withMw.map(r => r.nNewPolar)),
        rhoNewAll:     spearman(withMw.map(r => r.mw), withMw.map(r => r.nNewPolar + r.nNewPhobic)),
    };
}

/** Pose × standard-pair grid. 2 = made, 1 = right residue wrong chemistry, 0 = missed, -1 = not in the crop. */
export function standardHeat(rows, std, limit = 70) {
    return {
        pairs: std,
        rows: rows.slice(0, limit).map(r => {
            const exact = new Set(r.hitExact.map(p => pairKey(p.res, p.fam)));
            const partial = new Set(r.hitPartial.map(p => p.res));
            return {
                id: r.id, tier: r.tier,
                cells: std.map(sp => {
                    if (exact.has(pairKey(sp.res, sp.fam))) return 2;
                    if (partial.has(sp.res)) return 1;
                    if (r._pose.crop && !r._pose.crop.has(sp.res)) return -1;
                    return 0;
                }),
            };
        }),
    };
}

// ----------------------------------------------------------------
// Import — PLIP results from a file
//
// Accepts what the project's own pipelines emit:
//   JSONL  one pose per line: {id, smiles?, mw?, prob?, contacts:[{chain,resnr,restype,itype,...}]}
//   CSV    long format: id,chain,resnr,restype,itype[,distance][,smiles][,mw][,prob]
// ----------------------------------------------------------------

export function parsePlipFile(text, filename = '') {
    const trimmed = text.trim();
    if (!trimmed) throw new Error('the file is empty');
    if (trimmed[0] === '{' || trimmed[0] === '[') return _parseJson(trimmed);
    if (/\.jsonl?$/i.test(filename)) return _parseJson(trimmed);
    return _parseCsv(trimmed);
}

function _parseJson(text) {
    let records;
    if (text[0] === '[') {
        records = JSON.parse(text);
    } else {
        records = text.split('\n').map(l => l.trim()).filter(Boolean).map(l => JSON.parse(l));
        if (records.length === 1 && Array.isArray(records[0].poses)) records = records[0].poses;
    }
    return { poses: records.map((r, i) => _normalisePose(r, i)), source: 'jsonl' };
}

function _normalisePose(r, i) {
    const raw = r.contacts || r.interactions || [];
    const contacts = raw.map(c => ({
        chain:   String(c.chain ?? c.reschain ?? c.c ?? 'A'),
        resnr:   +(c.resnr ?? c.res ?? c.residue ?? c.n),
        restype: String(c.restype ?? c.restype3 ?? c.aa ?? '???').toUpperCase(),
        itype:   String(c.itype ?? c.type ?? c.interaction ?? 'contact'),
        dist:    c.dist != null ? +c.dist : (c.distance != null ? +c.distance : undefined),
        dist_ad: c.dist_ad != null ? +c.dist_ad : (c.distance_ad != null ? +c.distance_ad : undefined),
        dist_ah: c.dist_ah != null ? +c.dist_ah : (c.distance_ah != null ? +c.distance_ah : undefined),
    })).filter(c => Number.isFinite(c.resnr));
    return {
        id: r.id || r.pose_id || r.compound || r.name || `pose-${i + 1}`,
        smiles: r.smiles || r.SMILES || null,
        mw: r.mw != null ? +r.mw : (r.MW != null ? +r.MW : null),
        prob: r.prob != null ? +r.prob : (r.probability != null ? +r.probability : null),
        anion: r.anion || '',
        crop: r.crop || null,
        contacts,
    };
}

function _splitCsvLine(line) {
    const out = []; let cur = '', q = false;
    for (let i = 0; i < line.length; i++) {
        const ch = line[i];
        if (q) {
            if (ch === '"') { if (line[i + 1] === '"') { cur += '"'; i++; } else q = false; }
            else cur += ch;
        } else if (ch === '"') q = true;
        else if (ch === ',') { out.push(cur); cur = ''; }
        else cur += ch;
    }
    out.push(cur);
    return out.map(s => s.trim());
}

function _parseCsv(text) {
    const lines = text.split(/\r?\n/).filter(l => l.trim());
    const head = _splitCsvLine(lines[0]).map(h => h.toLowerCase().replace(/[^a-z0-9]/g, ''));
    const at = (names) => { for (const n of names) { const i = head.indexOf(n); if (i >= 0) return i; } return -1; };
    const ci = {
        id: at(['id', 'poseid', 'compound', 'name', 'moleculeid']),
        chain: at(['chain', 'reschain']),
        resnr: at(['resnr', 'res', 'residue', 'resi', 'residuenumber']),
        restype: at(['restype', 'aa', 'resname', 'residuetype']),
        itype: at(['itype', 'type', 'interaction', 'interactiontype']),
        dist: at(['dist', 'distance']),
        distad: at(['distad', 'distancead']),
        distah: at(['distah', 'distanceah']),
        smiles: at(['smiles']),
        mw: at(['mw', 'molecularweight', 'amw']),
        prob: at(['prob', 'probability', 'bindingprobability', 'affinityprobability']),
    };
    if (ci.id < 0 || ci.resnr < 0) {
        throw new Error('CSV needs at least an "id" and a "resnr" column (long format: one contact per row)');
    }
    const byId = new Map();
    for (let i = 1; i < lines.length; i++) {
        const f = _splitCsvLine(lines[i]);
        const id = f[ci.id];
        if (!id) continue;
        if (!byId.has(id)) byId.set(id, {
            id,
            smiles: ci.smiles >= 0 ? f[ci.smiles] || null : null,
            mw: ci.mw >= 0 && f[ci.mw] ? +f[ci.mw] : null,
            prob: ci.prob >= 0 && f[ci.prob] ? +f[ci.prob] : null,
            contacts: [],
        });
        const resnr = +f[ci.resnr];
        if (!Number.isFinite(resnr)) continue;
        byId.get(id).contacts.push({
            chain: ci.chain >= 0 ? (f[ci.chain] || 'A') : 'A',
            resnr,
            restype: (ci.restype >= 0 ? f[ci.restype] : '???').toUpperCase(),
            itype: ci.itype >= 0 ? (f[ci.itype] || 'contact') : 'contact',
            dist: ci.dist >= 0 && f[ci.dist] ? +f[ci.dist] : undefined,
            dist_ad: ci.distad >= 0 && f[ci.distad] ? +f[ci.distad] : undefined,
            dist_ah: ci.distah >= 0 && f[ci.distah] ? +f[ci.distah] : undefined,
        });
    }
    return { poses: [...byId.values()], source: 'csv' };
}

// ----------------------------------------------------------------
// Export
// ----------------------------------------------------------------

export function rowsToCsv(rows, decisions) {
    const cols = ['rank_lo', 'rank_hi', 'id', 'tier', 'decision', 'decision_note', 'why',
        'maintained_pct', 'maintained_value', 'standard_value', 'exact', 'partial', 'missed',
        'key_missed', 'new_polar', 'new_hydrophobic', 'mw', 'boltz_prob', 'anion', 'salt_share',
        'in_crop', 'assessable', 'off_site', 'flags', 'maintained_interactions',
        'missed_interactions', 'new_polar_residues'];
    const esc = (v) => {
        const s = v == null ? '' : String(v);
        return /[",\n]/.test(s) ? `"${s.replace(/"/g, '""')}"` : s;
    };
    const pairStr = (p) => `${p.res} ${FAMILY[p.fam].short}${p.key ? '*' : ''}`;
    const out = [cols.join(',')];
    for (const r of rows) {
        const d = decisions.get(r.id) || {};
        out.push([
            r.rankLo, r.rankHi, r.id, r.tier, d.verdict || '', d.note || '', r.why,
            (100 * r.maintainedFrac).toFixed(1), r.maintained.toFixed(2), r.standardTotal.toFixed(2),
            r.nExact, r.nPartial, r.missed.length, r.nKeyMissed,
            r.nNewPolar, r.nNewPhobic, r.mw ?? '', r.prob ?? '', r.anion || '',
            r.saltShare.toFixed(3), r.inCrop ?? '', r.assessable, r.isOffSite,
            r.flags.map(f => f.id).join(';'),
            r.hitExact.map(pairStr).join(';'),
            r.missed.map(pairStr).join(';'),
            r.newPolar.map(n => `${n.res} ${n.fams.map(f => FAMILY[f].short).join('/')}`).join(';'),
        ].map(esc).join(','));
    }
    return out.join('\n');
}
