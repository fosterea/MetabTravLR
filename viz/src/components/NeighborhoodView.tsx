/**
 * Canvas view: **neighborhood scores** for the selected entity — which cell types sit in
 * high-scoring neighborhoods for this metabolite / gene pair.
 *
 * ⚠️ This is deliberately a SEPARATE view from the graph, not another edge encoding. The graph
 * draws harreman's cell-type *interface* statistic (CT1↔CT2). These scores bucket each cell's
 * own score by that cell's own label, so they carry no interface and no direction — mixing them
 * into the graph would misrepresent them (parent doc 05 §5a).
 *
 * Form: horizontal bars, one row per cell type, sorted by the chosen metric. One series ⇒ one
 * hue and no legend; enrichment is the one signed metric, so it gets a diverging pair around a
 * neutral zero line. Every number is directly labelled in the row, so the chart doubles as its
 * own table view.
 */
import { useMemo, useState } from 'react';
import { useVizStore, selectCurrentTier } from '@/store/useVizStore';
import { entityLabel } from '@/data/ranking';
import type { NbhdRow } from '@/data/types';
import styles from './NeighborhoodView.module.css';

type Metric = 'fracSig' | 'meanCs' | 'log2Enrichment';

const METRICS: { id: Metric; label: string; help: string }[] = [
  {
    id: 'fracSig',
    label: 'Significant share',
    help: 'Fraction of this cell type’s cells with a significant score for this entity.',
  },
  {
    id: 'meanCs',
    label: 'Mean score',
    help: 'Mean interacting-cell score across all cells of this type (significant or not).',
  },
  {
    id: 'log2Enrichment',
    label: 'Enrichment (log₂)',
    help: 'log₂(observed / expected) significant share. Signed: above 0 is enriched, below is depleted.',
  },
];

/** Below this many significant cells, ratio metrics (especially log2 enrichment) are noise —
 *  the parent docs cite a Tier3 label whose top enrichment came from 3 significant cells. */
const MIN_SIG_CELLS = 25;

const pct = (v: number | null) => (v == null ? '—' : `${(v * 100).toFixed(v < 0.01 ? 2 : 1)}%`);
const dec = (v: number | null, d = 2) => (v == null ? '—' : v.toFixed(d));
const int = (v: number | null) => (v == null ? '—' : v.toLocaleString());
const nSigCells = (r: NbhdRow) =>
  r.nCells != null && r.fracSig != null ? Math.round(r.nCells * r.fracSig) : null;

export default function NeighborhoodView() {
  const dataset = useVizStore((s) => s.dataset);
  const tier = useVizStore(selectCurrentTier);
  const entityKind = useVizStore((s) => s.entityKind);
  const entityId = useVizStore((s) => s.entityId);
  const nbhdBundle = useVizStore((s) => s.nbhdBundle);
  const bundleLoading = useVizStore((s) => s.bundleLoading);
  const [metric, setMetric] = useState<Metric>('fracSig');

  const entity = useMemo(() => {
    if (!dataset || !entityId) return undefined;
    const list =
      entityKind === 'metabolite' ? dataset.entities.metabolite : dataset.entities.gene_pair;
    return list?.find((e) => e.id === entityId);
  }, [dataset, entityId, entityKind]);

  const rows = useMemo(() => {
    const all = (entityId && nbhdBundle?.byEntity[entityKind]?.[entityId]) || [];
    return [...all].sort((a, b) => (b[metric] ?? -Infinity) - (a[metric] ?? -Infinity));
  }, [nbhdBundle, entityKind, entityId, metric]);

  // Bar geometry. Non-negative metrics grow from a left baseline; enrichment is signed, so its
  // axis sits at the middle and bars grow either way from a neutral zero line.
  const diverging = metric === 'log2Enrichment';
  const extent = useMemo(() => {
    const vals = rows.map((r) => r[metric]).filter((v): v is number => v != null);
    if (!vals.length) return 1;
    return diverging
      ? Math.max(...vals.map(Math.abs), 0.001)
      : Math.max(...vals, 0.001);
  }, [rows, metric, diverging]);

  const metricInfo = METRICS.find((m) => m.id === metric)!;

  if (bundleLoading) return <div className="empty">Loading…</div>;
  if (!dataset?.hasNbhd) {
    return (
      <div className="empty">
        This dataset has no neighborhood scores — its harreman run predates them.
      </div>
    );
  }
  if (!entity) return <div className="empty">Select an entity to see its neighborhood scores.</div>;

  return (
    <div className={styles.wrap}>
      <div className={styles.head}>
        <div>
          <h2 className={styles.title}>Neighborhood scores</h2>
          <div className={styles.subtitle}>
            {entityLabel(entity)} · {tier?.label}
          </div>
        </div>
        <div className={styles.metrics} role="group" aria-label="Metric">
          {METRICS.map((m) => (
            <button
              key={m.id}
              aria-pressed={metric === m.id}
              className={`${styles.metricBtn} ${metric === m.id ? styles.metricOn : ''}`}
              onClick={() => setMetric(m.id)}
              title={m.help}
            >
              {m.label}
            </button>
          ))}
        </div>
      </div>

      <p className={styles.caveat}>
        <b>Not an interface statistic.</b> Each cell’s score is counted under its own cell type, so
        a row reads “cells of this type sit in high-scoring neighborhoods for {entityLabel(entity)}”
        — never “this type exchanges it with that one”. Use the graph view for interfaces.
      </p>

      {rows.length === 0 ? (
        <div className={styles.none}>
          No neighborhood scores for this {entityKind === 'metabolite' ? 'metabolite' : 'gene pair'}{' '}
          at {tier?.label}.
        </div>
      ) : (
        <>
          <div className={`${styles.row} ${styles.colHead}`} aria-hidden>
            <span>Cell type</span>
            <span className={styles.plotHead}>
              {metricInfo.label}
              {diverging && <span className={styles.zeroTick}>0</span>}
            </span>
            <span className={styles.numHead}>value</span>
            <span className={styles.numHead}>sig cells</span>
            <span className={styles.numHead}>cells</span>
          </div>

          <ul className={styles.list}>
            {rows.map((r) => {
              const v = r[metric];
              const nSig = nSigCells(r);
              // A thin cell type can't support a ratio; flag it instead of silently ranking it.
              const thin = nSig != null && nSig < MIN_SIG_CELLS;
              const frac = v == null ? 0 : Math.min(1, Math.abs(v) / extent);
              return (
                <li
                  key={r.cellType}
                  className={`${styles.row} ${thin ? styles.thin : ''}`}
                  title={[
                    r.cellType,
                    `significant share ${pct(r.fracSig)}`,
                    `mean score ${dec(r.meanCs, 3)}`,
                    `mean score (significant cells only) ${dec(r.meanCsSig, 2)}`,
                    `mean −log₁₀ p ${dec(r.meanNegLog10P, 2)}`,
                    `log₂ enrichment ${dec(r.log2Enrichment)}`,
                    `${int(nSig)} significant of ${int(r.nCells)} cells`,
                  ].join('\n')}
                >
                  <span className={styles.name}>
                    {r.cellType}
                    {thin && (
                      <span className={styles.thinTag} title="Too few significant cells to trust a ratio">
                        thin
                      </span>
                    )}
                  </span>

                  <span className={styles.plot}>
                    {diverging ? (
                      <span className={styles.axis}>
                        <span className={styles.zeroLine} />
                        <span
                          className={`${styles.bar} ${(v ?? 0) < 0 ? styles.barNeg : styles.barPos}`}
                          style={{ width: `${(frac * 100) / 2}%` }}
                        />
                      </span>
                    ) : (
                      <span className={styles.axis}>
                        <span className={styles.bar} style={{ width: `${frac * 100}%` }} />
                      </span>
                    )}
                  </span>

                  <span className={styles.num}>
                    {metric === 'fracSig' ? pct(v) : dec(v, metric === 'meanCs' ? 3 : 2)}
                  </span>
                  <span className={styles.num}>{int(nSig)}</span>
                  <span className={styles.num}>{int(r.nCells)}</span>
                </li>
              );
            })}
          </ul>

          <p className={styles.foot}>
            {metricInfo.help} Rows marked <b>thin</b> have fewer than {MIN_SIG_CELLS} significant
            cells — their ratios are unstable. Hover a row for the full statistics.
          </p>
        </>
      )}
    </div>
  );
}
