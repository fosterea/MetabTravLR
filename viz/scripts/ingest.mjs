#!/usr/bin/env node
/**
 * Ingest adapter: harreman `easy_download` outputs -> app data contract JSON.
 *
 * The app never parses raw CSV. This script is the ONE place that knows the
 * on-disk harreman schema (see viz/docs/05_data_contract.md and the parent
 * project's DataForClaude/documentation/05_harreman_reference.md, produced by
 * metab_processing/). When new source formats appear (e.g. SpaceTravLR betadata),
 * add a sibling adapter that emits the SAME contract — do not change the app.
 *
 * Usage:
 *   node scripts/ingest.mjs <path> [--out public/data] [--id <id>] [--name <name>]
 *
 * <path> may be any of (auto-detected):
 *   - a harreman_outputs/ directory
 *   - an easy_download/ directory containing harreman_outputs/
 *   - a ROOT of datasets:          <root>/<datasetName>/easy_download/harreman_outputs/
 *   - a ROOT of PROJECTS:          <root>/<project>/<datasetName>/easy_download/harreman_outputs/
 *     (the `Results/<project>/<dataset>` layout — every <datasetName> becomes one dataset,
 *      tagged with its <project>)
 *
 * A dataset whose run never finished (network JSON present but no tier outputs) is NOT a hard
 * error: it is recorded in the manifest as `available: false` with a reason, so the app can show
 * it greyed-out instead of silently dropping it. Any other per-dataset failure is caught and
 * degraded the same way, so one bad dataset can never abort the whole ingest.
 *
 * Output tree:
 *   <out>/manifest.json
 *   <out>/<id>/dataset.json
 *   <out>/<id>/edges/<Tier>.<entityKind>.json
 *   <out>/<id>/nbhd/<Tier>.json            (neighborhood scores; omitted if the run lacks them)
 *   <out>/<id>/beta/<Tier>.json            (SpaceTravLR coefficients; omitted if the run lacks them)
 */
import { readFileSync, writeFileSync, mkdirSync, existsSync, readdirSync, statSync } from 'node:fs';
import { join, basename, resolve } from 'node:path';
import Papa from 'papaparse';

const SCHEMA_VERSION = 3; // 3: +beta bundles (SpaceTravLR gene-pair coefficients) / hasBeta
// 2: manifest datasets carry project/available/hasNbhd; +nbhd bundles

// ---------- tiny helpers ----------
const isDir = (p) => existsSync(p) && statSync(p).isDirectory();
const num = (v) => {
  if (v === '' || v === null || v === undefined) return null;
  const n = Number(v);
  return Number.isNaN(n) ? null : n;
};
const bool = (v) => String(v).trim().toLowerCase() === 'true';
const readCsv = (p) => {
  const text = readFileSync(p, 'utf8');
  const { data, errors } = Papa.parse(text, { header: true, skipEmptyLines: 'greedy' });
  if (errors.length) {
    // Papa reports per-row issues; surface the first so ingest stays honest.
    console.warn(
      `  ! CSV parse warnings in ${basename(p)}: ${errors[0].message} (row ${errors[0].row})`,
    );
  }
  return data;
};
const readJson = (p) => JSON.parse(readFileSync(p, 'utf8'));

// harreman writes bracketed filenames; keep them in one place.
const F = {
  network: 'harreman_network.json',
  globalM: '[ccc_results][cell_com_df_m].csv',
  tierM: '[ct_ccc_results][cell_com_df_m].csv',
  tierGp: '[ct_ccc_results][cell_com_df_gp_sig].csv',
  metabSummary: join('summary', 'metabolite_summary.csv'),
  // Neighborhood scores (harreman wrapper `nbhd_scores.py`). Per (cell type, entity) summary of
  // the per-cell interacting-cell scores. NOT the ct interface statistic — see buildNbhd().
  nbhdM: '[nbhd_scores][summary_m].csv',
  nbhdGp: '[nbhd_scores][summary_gp].csv',
  // SpaceTravLR (MetabTravLR) learned coefficients, a SIBLING of harreman_outputs/ inside
  // easy_download/. Per (target gene, directed transporter pair, cell type) mean beta.
  betaDir: 'metabtravlr_outputs',
  betaGp: 'gene_pairs.csv',
};

const gpId = (g1, g2) => `${g1}__${g2}`;
/**
 * Order-INDEPENDENT key for the beta tables. The harreman `gp` list is not sorted and may hold a
 * pair in both orders, so a directed beta row (export,import) cannot be attributed to one `gpId`
 * unambiguously. Betas are therefore keyed on the sorted pair, and the app looks them up the same
 * way — the direction survives inside each row as `env`/`cell`, where it actually means something.
 */
const betaKey = (g1, g2) => (g1 <= g2 ? `${g1}__${g2}` : `${g2}__${g1}`);

/** "Human_Prostate_Adenocarcinoma" -> "Human Prostate Adenocarcinoma". */
const prettyName = (s) => s.replace(/[_-]+/g, ' ').trim();

// ---------- locate the harreman_outputs dir(s) ----------
function findHarremanRoot(dir) {
  // A harreman_outputs/ CHILD wins over a network JSON sitting in `dir` itself: some runs leave a
  // stray copy of harreman_network.json one level up (in easy_download/), and matching that would
  // pick a directory with no tier tables and look like an incomplete dataset.
  if (isDir(join(dir, 'harreman_outputs'))) return join(dir, 'harreman_outputs');
  // dir itself is a harreman_outputs dir?
  if (existsSync(join(dir, F.network))) return dir;
  return null;
}

/** Datasets directly under `dir`: <dir>/<name>/[easy_download/]harreman_outputs. */
function datasetsUnder(dir, project) {
  const found = [];
  for (const name of readdirSync(dir)) {
    const child = join(dir, name);
    if (!isDir(child)) continue;
    const root = findHarremanRoot(child) || findHarremanRoot(join(child, 'easy_download'));
    if (root) found.push({ id: name, root, project });
  }
  return found;
}

function discoverDatasets(inputPath) {
  const p = resolve(inputPath);
  if (!isDir(p)) throw new Error(`Not a directory: ${p}`);

  const direct = findHarremanRoot(p);
  if (direct) return [{ id: basename(resolve(p, '..')) || basename(p), root: direct, project: null }];

  // <root>/<datasetName>/easy_download/harreman_outputs
  const flat = datasetsUnder(p, null);
  if (flat.length) return flat;

  // <root>/<project>/<datasetName>/easy_download/harreman_outputs  (the Results/ layout).
  // Only one level deeper — enough for Results/<project>/<dataset>, and it can't wander.
  const nested = [];
  for (const name of readdirSync(p)) {
    const child = join(p, name);
    if (isDir(child)) nested.push(...datasetsUnder(child, name));
  }
  if (nested.length) return nested;

  throw new Error(`No harreman outputs found under ${p} (looked for ${F.network}).`);
}

// ---------- tier discovery ----------
function discoverTiers(root) {
  const tierDirs = readdirSync(root)
    .filter((n) => /^Tier\w+$/i.test(n) && isDir(join(root, n)))
    .sort(); // Tier1 < Tier2 < Tier3 lexicographically for single digits
  return tierDirs;
}

/** A tier dir only counts once it has the per-cell-type metabolite table — a run that was
 *  interrupted can leave an empty (or metadata-only) tier dir behind. */
function tierIsUsable(root, t) {
  return existsSync(join(root, t, F.tierM));
}

// ---------- build one dataset ----------
function buildDataset(id, root, nameOverride, project) {
  console.log(`\n• dataset "${id}"  <- ${root}`);
  const network = readJson(join(root, F.network));
  const tierIds = discoverTiers(root).filter((t) => tierIsUsable(root, t));
  console.log(`  tiers: ${tierIds.join(', ') || '(none)'}`);
  // Graceful degradation: the harreman network JSON is written early, the tier tables last. A
  // dataset with no usable tier has nothing to draw a graph from — report it as unavailable
  // rather than emitting an empty dataset the app would render as a blank canvas.
  if (!tierIds.length) {
    return { unavailable: `no tier outputs — the harreman run for "${id}" is incomplete` };
  }

  // ---- metabolite entities from network + summary + global table ----
  const gpPerMetab = network.gp_per_metabolite || {};
  const metabSummary = existsSync(join(root, F.metabSummary))
    ? indexBy(readCsv(join(root, F.metabSummary)), 'metabolite')
    : {};
  const globalM = existsSync(join(root, F.globalM))
    ? indexBy(readCsv(join(root, F.globalM)), 'Metabolite')
    : {};

  const metaboliteEntities = Object.keys(gpPerMetab).map((name) => {
    const pairs = (gpPerMetab[name].gene_pair || []).map(([a, b]) => [a, b]);
    const sum = metabSummary[name] || {};
    const g = globalM[name] || {};
    const perTier = {};
    for (const t of tierIds) {
      perTier[t] = {
        nSigPairs: num(sum[`${t}_n_sig_pairs`]),
        tcellInvolved: sum[`${t}_tcell_involved`] != null ? bool(sum[`${t}_tcell_involved`]) : null,
        withinTcell: emptyToNull(sum[`${t}_within_Tcell`]),
        tcellInterfaces: emptyToNull(sum[`${t}_Tcell_interfaces`]),
        interactions: emptyToNull(sum[`${t}_interactions`]),
      };
    }
    return {
      id: name,
      name,
      kind: 'metabolite',
      nGenePairs: pairs.length || num(sum.n_gene_pairs) || 0,
      genePairs: pairs,
      globalSignificant:
        sum.global_significant != null ? bool(sum.global_significant) : bool(g.selected),
      globalFDR: num(sum.global_FDR_np) ?? num(g.FDR_np),
      nSigGenePairsGlobal: num(sum.n_sig_gene_pairs_global),
      perTier,
    };
  });

  // ---- gene-pair entities from network gp list + gp<->metabolite (many-to-many) ----
  const pairToMetabs = {};
  for (const [metab, obj] of Object.entries(gpPerMetab)) {
    for (const [a, b] of obj.gene_pair || []) {
      (pairToMetabs[gpId(a, b)] ||= new Set()).add(metab);
    }
  }
  const genePairEntities = (network.gp || []).map(([a, b]) => ({
    id: gpId(a, b),
    genes: [a, b],
    kind: 'gene_pair',
    metabolites: [...(pairToMetabs[gpId(a, b)] || [])].sort(),
  }));

  // gp id lookup: the nbhd tables key gene pairs as "GENE1_GENE2" (single underscore), which is
  // ambiguous to split. Resolve via the network's own pair list instead of guessing.
  const gpBySingleUnderscore = {};
  for (const [a, b] of network.gp || []) gpBySingleUnderscore[`${a}_${b}`] = gpId(a, b);

  // Every unordered pair harreman knows about, so a beta row for a pair outside the network can
  // be dropped rather than silently creating an entity the rest of the app has never heard of.
  const knownBetaKeys = new Set((network.gp || []).map(([a, b]) => betaKey(a, b)));

  // ---- edges per (tier, entityKind) ----
  const betaRoot = findBetaRoot(root);
  const tiers = [];
  const edgeBundles = []; // {filename, bundle}
  const nbhdBundles = []; // {filename, bundle}
  const betaBundles = []; // {filename, bundle}
  tierIds.forEach((t, i) => {
    const tdir = join(root, t);
    const mRows = existsSync(join(tdir, F.tierM)) ? readCsv(join(tdir, F.tierM)) : [];
    const gpRows = existsSync(join(tdir, F.tierGp)) ? readCsv(join(tdir, F.tierGp)) : [];

    const cellTypes = uniq([
      ...mRows.flatMap((r) => [r['Cell Type 1'], r['Cell Type 2']]),
      ...gpRows.flatMap((r) => [r['Cell Type 1'], r['Cell Type 2']]),
    ]).filter(Boolean);

    tiers.push({
      id: t,
      label: t,
      cellTypes,
      parentTier: i > 0 ? tierIds[i - 1] : null, // coarse->fine ordering assumption
      cellTypeParents: null, // no annotation crosswalk yet (see docs TODO)
    });

    edgeBundles.push({
      file: `${t}.metabolite.json`,
      bundle: bundleEdges(t, 'metabolite', cellTypes, mRows, (r) => r.metabolite),
    });
    edgeBundles.push({
      file: `${t}.gene_pair.json`,
      bundle: bundleEdges(t, 'gene_pair', cellTypes, gpRows, (r) => gpId(r['Gene 1'], r['Gene 2'])),
    });

    const nbhd = buildNbhd(t, tdir, gpBySingleUnderscore);
    if (nbhd) nbhdBundles.push({ file: `${t}.json`, bundle: nbhd });

    const beta = betaRoot ? buildBeta(t, join(betaRoot, t), knownBetaKeys) : null;
    if (beta) betaBundles.push({ file: `${t}.json`, bundle: beta });
  });

  const dataset = {
    id,
    name: nameOverride || prettyName(id),
    project: project ? prettyName(project) : null,
    source: 'harreman',
    entityKinds: ['metabolite', 'gene_pair'],
    hasNbhd: nbhdBundles.length > 0,
    hasBeta: betaBundles.length > 0,
    tiers,
    entities: {
      metabolite: metaboliteEntities,
      gene_pair: genePairEntities,
    },
  };

  return { dataset, edgeBundles, nbhdBundles, betaBundles };
}

/** `metabtravlr_outputs/` sits beside `harreman_outputs/` inside `easy_download/`. */
function findBetaRoot(harremanRoot) {
  for (const cand of [
    join(harremanRoot, '..', F.betaDir), // root = <easy_download>/harreman_outputs
    join(harremanRoot, F.betaDir), // root = <easy_download> (network JSON sat directly there)
  ]) {
    if (isDir(cand)) return cand;
  }
  return null;
}

/**
 * SpaceTravLR learned coefficients for one tier (`metabtravlr_outputs/<Tier>/gene_pairs.csv`).
 *
 * A row is one (target gene, DIRECTED transporter pair, cell type) mean beta: how much the
 * modelled interaction between an environment-expressed transporter and a cell-expressed
 * transporter moves that target gene, averaged over the cells of that type.
 *
 * Direction is real here, unlike everything else in this app: `export` is the gene expressed by
 * the ENVIRONMENT (neighboring cells), `import` the gene expressed by THE CELL itself. So
 * `A -> B` and `B -> A` are two different coefficients and must never be merged. They are stored
 * as sibling rows under one order-independent `betaKey`, each keeping its own `env`/`cell`.
 *
 * Returns null when this dataset/tier has no SpaceTravLR run.
 */
function buildBeta(tier, tdir, knownBetaKeys) {
  const path = join(tdir, F.betaGp);
  if (!existsSync(path)) return null;

  const byPair = {};
  let dropped = 0;
  for (const r of readCsv(path)) {
    const env = r.export;
    const cell = r.import;
    const gene = r.gene;
    const ct = r.cell_type;
    if (!env || !cell || !gene || !ct) continue;
    const key = betaKey(env, cell);
    // A pair the harreman network doesn't list has no entity in the app to hang off.
    if (!knownBetaKeys.has(key)) {
      dropped++;
      continue;
    }
    (byPair[key] ||= []).push({
      env,
      cell,
      gene,
      cellType: ct,
      mean: num(r.mean),
      std: num(r.std),
      n: num(r.n),
    });
  }
  if (!Object.keys(byPair).length) return null;

  const all = Object.values(byPair).flat();
  // Strongest |beta| first, so the app's default read is the biggest effect.
  for (const rows of Object.values(byPair)) {
    rows.sort((a, b) => Math.abs(b.mean ?? 0) - Math.abs(a.mean ?? 0));
  }
  if (dropped) console.warn(`  ! ${tier}: dropped ${dropped} beta row(s) for pairs not in the network`);

  return {
    tier,
    cellTypes: uniq(all.map((r) => r.cellType)),
    targetGenes: uniq(all.map((r) => r.gene)).sort(),
    byPair,
  };
}

/**
 * Neighborhood scores for one tier: per (cell type, entity), how much that cell type's OWN cells
 * sit in high-scoring neighborhoods for the entity.
 *
 * ⚠️ This is deliberately NOT the cell-type-interface statistic the graph draws. Each cell's
 * score is bucketed by that cell's own label, so a row means "cells of type X score high on
 * metabolite M", never "X talks to Y". The app must label it as such and never mix it into edges.
 * `log2_enrichment` is unstable for tiny labels — `nCells` travels with every row so the UI can
 * de-emphasize thin ones (see parent doc 05 §5a).
 *
 * Returns null when the run predates the nbhd wrapper (both files missing).
 */
function buildNbhd(tier, tdir, gpBySingleUnderscore) {
  const mPath = join(tdir, F.nbhdM);
  const gpPath = join(tdir, F.nbhdGp);
  const hasM = existsSync(mPath);
  const hasGp = existsSync(gpPath);
  if (!hasM && !hasGp) return null;

  const rowsOf = (path, keyCol, toId) => {
    const byEntity = {};
    for (const r of readCsv(path)) {
      const raw = r[keyCol];
      const ct = r.cell_type;
      if (!raw || !ct) continue;
      const id = toId(raw);
      if (!id) continue;
      (byEntity[id] ||= []).push({
        cellType: ct,
        nCells: num(r.n_cells),
        fracSig: num(r.frac_sig),
        meanCs: num(r.mean_cs),
        meanCsSig: num(r.mean_cs_sig),
        meanNegLog10P: num(r.mean_neg_log10_pval),
        log2Enrichment: num(r.log2_enrichment),
      });
    }
    // Strongest neighborhood first — the UI's default read.
    for (const rows of Object.values(byEntity)) rows.sort((a, b) => (b.fracSig ?? 0) - (a.fracSig ?? 0));
    return byEntity;
  };

  const metabolite = hasM ? rowsOf(mPath, 'metabolite', (v) => v) : {};
  // Unknown pairs (present in the nbhd table but not in the network gp list) are dropped rather
  // than guessed at, so a nbhd row can never be attributed to the wrong pair.
  const gene_pair = hasGp
    ? rowsOf(gpPath, 'gene_pair', (v) => gpBySingleUnderscore[v] ?? null)
    : {};

  const cellTypes = uniq(
    [...Object.values(metabolite), ...Object.values(gene_pair)].flat().map((r) => r.cellType),
  );
  return { tier, cellTypes, byEntity: { metabolite, gene_pair } };
}

function bundleEdges(tier, entityKind, cellTypes, rows, keyFn) {
  const byEntity = {};
  for (const r of rows) {
    const key = keyFn(r);
    if (!key || r['Cell Type 1'] == null || r['Cell Type 2'] == null) continue;
    (byEntity[key] ||= []).push({
      source: r['Cell Type 1'],
      target: r['Cell Type 2'],
      scores: {
        C_p: num(r.C_p),
        Z: num(r.Z),
        Z_FDR: num(r.Z_FDR),
        C_np: num(r.C_np),
        FDR_np: num(r.FDR_np),
        selected: bool(r.selected),
      },
    });
  }
  return { tier, entityKind, cellTypes, byEntity };
}

// ---------- misc ----------
function indexBy(rows, key) {
  const out = {};
  for (const r of rows) if (r[key] != null) out[r[key]] = r;
  return out;
}
const uniq = (a) => [...new Set(a)];
const emptyToNull = (v) => (v === '' || v == null ? null : v);

// ---------- write ----------
function writeDataset(outDir, { dataset, edgeBundles, nbhdBundles, betaBundles }) {
  const dsDir = join(outDir, dataset.id);
  mkdirSync(join(dsDir, 'edges'), { recursive: true });
  writeFileSync(join(dsDir, 'dataset.json'), JSON.stringify(dataset));
  for (const { file, bundle } of edgeBundles) {
    writeFileSync(join(dsDir, 'edges', file), JSON.stringify(bundle));
  }
  if (nbhdBundles.length) {
    mkdirSync(join(dsDir, 'nbhd'), { recursive: true });
    for (const { file, bundle } of nbhdBundles) {
      writeFileSync(join(dsDir, 'nbhd', file), JSON.stringify(bundle));
    }
  }
  if (betaBundles.length) {
    mkdirSync(join(dsDir, 'beta'), { recursive: true });
    for (const { file, bundle } of betaBundles) {
      writeFileSync(join(dsDir, 'beta', file), JSON.stringify(bundle));
    }
  }
  const nMetab = dataset.entities.metabolite?.length ?? 0;
  const nGp = dataset.entities.gene_pair?.length ?? 0;
  console.log(
    `  wrote dataset.json (${nMetab} metabolites, ${nGp} gene pairs) + ${edgeBundles.length} edge files` +
      (nbhdBundles.length ? ` + ${nbhdBundles.length} nbhd files` : ' (no nbhd scores)') +
      (betaBundles.length ? ` + ${betaBundles.length} beta files` : ''),
  );
}

// ---------- main ----------
function main() {
  const args = process.argv.slice(2);
  if (!args.length || args[0].startsWith('-')) {
    console.error(
      'Usage: node scripts/ingest.mjs <path> [--out public/data] [--id <id>] [--name <name>]',
    );
    process.exit(1);
  }
  const inputPath = args[0];
  const getFlag = (f) => {
    const i = args.indexOf(f);
    return i >= 0 ? args[i + 1] : undefined;
  };
  const outDir = resolve(getFlag('--out') || 'public/data');
  const idOverride = getFlag('--id');
  const nameOverride = getFlag('--name');

  const datasets = discoverDatasets(inputPath);
  if (idOverride && datasets.length === 1) datasets[0].id = idOverride;

  mkdirSync(outDir, { recursive: true });
  const refs = [];
  for (const { id, root, project } of datasets) {
    // One unfinished/corrupt dataset must never abort the run — degrade it to an unavailable
    // manifest entry so the app can list it greyed-out with the reason.
    let built;
    try {
      built = buildDataset(id, root, datasets.length === 1 ? nameOverride : undefined, project);
    } catch (e) {
      built = { unavailable: e.message };
    }
    if (built.unavailable) {
      console.warn(`  ! skipping "${id}": ${built.unavailable}`);
      refs.push({
        id,
        name: prettyName(id),
        project: project ? prettyName(project) : null,
        source: 'harreman',
        entityKinds: [],
        tierIds: [],
        hasNbhd: false,
        hasBeta: false,
        available: false,
        unavailableReason: built.unavailable,
      });
      continue;
    }
    writeDataset(outDir, built);
    refs.push({
      id: built.dataset.id,
      name: built.dataset.name,
      project: built.dataset.project,
      source: built.dataset.source,
      entityKinds: built.dataset.entityKinds,
      tierIds: built.dataset.tiers.map((t) => t.id),
      hasNbhd: built.dataset.hasNbhd,
      hasBeta: built.dataset.hasBeta,
      available: true,
    });
  }

  // Available datasets first (so the app's default selection is always a usable one), then by name.
  refs.sort((a, b) => Number(b.available) - Number(a.available) || a.name.localeCompare(b.name));

  const manifest = { generatedAt: null, schemaVersion: SCHEMA_VERSION, datasets: refs };
  writeFileSync(join(outDir, 'manifest.json'), JSON.stringify(manifest, null, 2));
  const nOk = refs.filter((r) => r.available).length;
  console.log(
    `\n✓ manifest.json written with ${nOk} dataset(s)` +
      (refs.length - nOk ? ` (+${refs.length - nOk} unavailable)` : '') +
      ` -> ${outDir}`,
  );
}

main();
