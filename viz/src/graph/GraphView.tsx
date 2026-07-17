/** The graph canvas. Renders cell-type nodes for the current tier and undirected entity
 *  edges for the selected entity. Cytoscape lives entirely behind this component.
 *  Interactions: click an edge to pick it (details panel); hover an edge for its strength. */
import { useEffect, useMemo, useRef } from 'react';
import cytoscape from 'cytoscape';
import { useVizStore, selectCurrentTier } from '@/store/useVizStore';
import { classifyCellType } from '@/data/cellTypes';
import { makeEdgeWidthScale, isSelfEdge, sameInterface } from '@/data/scales';
import { formatStrength } from '@/data/format';
import type { EntityEdge } from '@/data/types';
import { buildStylesheet } from './stylesheet';

export default function GraphView() {
  const containerRef = useRef<HTMLDivElement>(null);
  const tooltipRef = useRef<HTMLDivElement>(null);
  const cyRef = useRef<cytoscape.Core | null>(null);

  const tier = useVizStore(selectCurrentTier);
  const edgeBundle = useVizStore((s) => s.edgeBundle);
  const entityId = useVizStore((s) => s.entityId);
  const showNonSignificant = useVizStore((s) => s.showNonSignificant);
  const bundleLoading = useVizStore((s) => s.bundleLoading);
  const selectedEdge = useVizStore((s) => s.selectedEdge);

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

  // Create the cy instance once and wire the interaction handlers.
  useEffect(() => {
    if (!containerRef.current) return;
    const cy = cytoscape({
      container: containerRef.current,
      style: buildStylesheet(),
      minZoom: 0.2,
      maxZoom: 3,
      // We drive edge highlight from the store via a `.picked` class, so disable
      // Cytoscape's own tap-selection to keep a single source of truth.
      autounselectify: true,
      boxSelectionEnabled: false,
    });
    cyRef.current = cy;
    // Dev-only test handle: Cytoscape renders to <canvas> (no DOM to probe), so expose the
    // instance for the playwright MCP to assert node/label/edge geometry. Stripped in prod.
    if (import.meta.env.DEV) (window as unknown as { __cy?: cytoscape.Core }).__cy = cy;

    // Click an edge → pick it; click empty canvas → clear the pick.
    cy.on('tap', 'edge', (evt) => {
      const e = evt.target;
      useVizStore.getState().selectEdge({ source: e.data('source'), target: e.data('target') });
    });
    cy.on('tap', (evt) => {
      if (evt.target === cy) useVizStore.getState().selectEdge(undefined);
    });

    // Hover an edge → floating tooltip with the numeric strength. We mutate the tooltip
    // DOM directly (not React state) so mousemove doesn't thrash re-renders.
    const showTip = (evt: cytoscape.EventObject) => {
      const tip = tooltipRef.current;
      if (!tip) return;
      const e = evt.target;
      const self = e.data('source') === e.data('target');
      const label = self
        ? `within ${e.data('source')}`
        : `${e.data('source')} ↔ ${e.data('target')}`;
      tip.innerHTML = `<b>${escapeHtml(label)}</b><br/>strength C<sub>np</sub> ${escapeHtml(
        formatStrength(e.data('cnp')),
      )}`;
      tip.style.display = 'block';
      moveTip(evt);
    };
    const moveTip = (evt: cytoscape.EventObject) => {
      const tip = tooltipRef.current;
      if (!tip || tip.style.display === 'none' || !evt.renderedPosition) return;
      const { x, y } = evt.renderedPosition;
      tip.style.left = `${x + 12}px`;
      tip.style.top = `${y + 12}px`;
    };
    const hideTip = () => {
      if (tooltipRef.current) tooltipRef.current.style.display = 'none';
    };
    cy.on('mouseover', 'edge', showTip);
    cy.on('mousemove', 'edge', moveTip);
    cy.on('mouseout', 'edge', hideTip);

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
          cnp: e.scores.C_np, // carried for the hover tooltip
        },
        classes: [e.scores.selected ? 'sig' : 'nonsig', isSelfEdge(e) ? 'self' : ''].join(' '),
      })),
    ];

    cy.batch(() => {
      cy.elements().remove();
      cy.add(els);
    });
    if (tooltipRef.current) tooltipRef.current.style.display = 'none';
    cy.style(buildStylesheet());
    cy.layout({ name: 'circle', padding: 60, animate: false }).run();
    // Extra fit padding leaves room for the below-node labels so they aren't clipped.
    cy.fit(undefined, 72);
  }, [tier, edges]);

  // Reflect the picked edge as a `.picked` class (highlight), order-agnostically.
  useEffect(() => {
    const cy = cyRef.current;
    if (!cy) return;
    cy.edges('.picked').removeClass('picked');
    if (!selectedEdge) return;
    cy.edges()
      .filter((e) => sameInterface({ source: e.data('source'), target: e.data('target') }, selectedEdge))
      .addClass('picked');
  }, [selectedEdge, edges]);

  // Loading takes precedence over the empty state so the two never overlap.
  const showEmpty = !bundleLoading && (!entityId || edges.length === 0);
  return (
    <>
      <div className="graph" ref={containerRef} data-testid="graph" />
      <div className="edge-tip" ref={tooltipRef} role="tooltip" aria-hidden style={{ display: 'none' }} />
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

/** Minimal HTML escape for the tooltip (cell-type names/metabolites are trusted, but keep
 *  the innerHTML injection safe if a name ever contains angle brackets). */
function escapeHtml(s: string): string {
  return s.replace(/[&<>"]/g, (c) => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;' })[c]!);
}
