/** The graph canvas. Renders cell-type nodes for the current tier and undirected entity
 *  edges for the selected entity. Cytoscape lives entirely behind this component. */
import { useEffect, useMemo, useRef } from 'react';
import cytoscape from 'cytoscape';
import { useVizStore, selectCurrentTier } from '@/store/useVizStore';
import { classifyCellType } from '@/data/cellTypes';
import { makeEdgeWidthScale, isSelfEdge } from '@/data/scales';
import type { EntityEdge } from '@/data/types';
import { buildStylesheet } from './stylesheet';

export default function GraphView() {
  const containerRef = useRef<HTMLDivElement>(null);
  const cyRef = useRef<cytoscape.Core | null>(null);

  const tier = useVizStore(selectCurrentTier);
  const edgeBundle = useVizStore((s) => s.edgeBundle);
  const entityId = useVizStore((s) => s.entityId);
  const showNonSignificant = useVizStore((s) => s.showNonSignificant);
  const bundleLoading = useVizStore((s) => s.bundleLoading);

  // Edges to display for the current selection. Defensively drop any edge whose endpoints
  // aren't in the current tier — a stale bundle can linger for a frame after a tier/kind
  // switch and would otherwise make cy.add throw "nonexistant source".
  const edges: EntityEdge[] = useMemo(() => {
    if (!entityId || !edgeBundle || !tier) return [];
    const all = edgeBundle.byEntity[entityId] ?? [];
    const present = new Set(tier.cellTypes);
    const inTier = all.filter((e) => present.has(e.source) && present.has(e.target));
    return showNonSignificant ? inTier : inTier.filter((e) => e.scores.selected);
  }, [entityId, edgeBundle, showNonSignificant, tier]);

  // Create the cy instance once.
  useEffect(() => {
    if (!containerRef.current) return;
    const cy = cytoscape({
      container: containerRef.current,
      style: buildStylesheet(),
      minZoom: 0.2,
      maxZoom: 3,
    });
    cyRef.current = cy;
    return () => {
      cy.destroy();
      cyRef.current = null;
    };
  }, []);

  // Rebuild elements when tier / edges change.
  useEffect(() => {
    const cy = cyRef.current;
    if (!cy || !tier) return;

    const widthScale = makeEdgeWidthScale(edges);
    const nodes = tier.cellTypes.map((ct) => ({
      data: { id: ct, label: ct },
      classes: `fam-${classifyCellType(ct)}`,
    }));
    const els: cytoscape.ElementDefinition[] = [
      ...nodes,
      ...edges.map((e, i) => ({
        data: {
          id: `e${i}:${e.source}->${e.target}`,
          source: e.source,
          target: e.target,
          w: widthScale(e.scores.C_np),
        },
        classes: [e.scores.selected ? 'sig' : 'nonsig', isSelfEdge(e) ? 'self' : ''].join(' '),
      })),
    ];

    cy.batch(() => {
      cy.elements().remove();
      cy.add(els);
    });
    cy.style(buildStylesheet());
    cy.layout({ name: 'circle', padding: 40, animate: false }).run();
    cy.fit(undefined, 48);
  }, [tier, edges]);

  // Loading takes precedence over the empty state so the two never overlap.
  const showEmpty = !bundleLoading && (!entityId || edges.length === 0);
  return (
    <>
      <div className="graph" ref={containerRef} data-testid="graph" />
      {bundleLoading && <div className="empty">Loading…</div>}
      {showEmpty && (
        <div className="empty">
          {!entityId
            ? 'Select an entity to see which cell-type interfaces exchange it.'
            : `No ${showNonSignificant ? '' : 'significant '}interfaces for this selection at ${tier?.label}.`}
        </div>
      )}
    </>
  );
}
