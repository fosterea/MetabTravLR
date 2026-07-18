/** Cytoscape stylesheet, built from theme tokens. See docs/02_style_and_conventions.md.
 *  Edge WIDTH encodes strength (data 'w'); edge COLOR/dash encodes significance. */
import type cytoscape from 'cytoscape';
import { cssVar } from './cssVars';

export function buildStylesheet(): cytoscape.StylesheetJson {
  const textPrimary = cssVar('--text-primary');
  const tcell = cssVar('--cell-tcell');
  const other = cssVar('--cell-other');
  const edgeSig = cssVar('--edge-sig');
  const edgeNonsig = cssVar('--edge-nonsig');
  const accent = cssVar('--accent');
  const border = cssVar('--border');
  const canvasBg = cssVar('--bg-canvas');

  return [
    {
      selector: 'node',
      style: {
        label: 'data(label)',
        'font-size': 12,
        // Cytoscape draws to canvas and rejects CSS keyword/quoted font stacks
        // (e.g. ui-sans-serif, 'Segoe UI'); use a plain family list.
        'font-family': 'Helvetica Neue, Arial, sans-serif',
        color: textPrimary,
        // Label sits BELOW the node so long cell-type names ("Proliferating CD8 T
        // cell") never overflow the circle. A canvas-colored halo keeps it legible
        // where it crosses edges.
        'text-valign': 'bottom',
        'text-halign': 'center',
        'text-margin-y': 6,
        'text-wrap': 'wrap',
        'text-max-width': '130px',
        'text-background-color': canvasBg,
        'text-background-opacity': 0.85,
        'text-background-shape': 'roundrectangle',
        'text-background-padding': '3px',
        // Larger than the max edge width (15px) so strong edges anchor cleanly instead of
        // blobbing into a small node.
        width: 56,
        height: 56,
        'border-width': 2,
        'border-color': border,
      },
    },
    {
      selector: 'node.fam-tcell',
      style: { 'background-color': tcell },
    },
    {
      selector: 'node.fam-other',
      style: { 'background-color': other },
    },
    {
      selector: 'edge',
      style: {
        width: 'data(w)',
        'curve-style': 'bezier',
        // Wider control-point step spreads parallel edges (e.g. gene-pair fan-outs) apart
        // instead of bundling them into one clump.
        'control-point-step-size': 55,
        'line-color': edgeNonsig,
        opacity: 0.9,
        // Self-loops arc up and away from the below-node label; a wide sweep reads as a clear
        // arc rather than a blob on top of the node.
        'loop-direction': '-90deg',
        'loop-sweep': '80deg',
      },
    },
    {
      selector: 'edge.sig',
      style: { 'line-color': edgeSig, opacity: 0.92 },
    },
    {
      selector: 'edge.nonsig',
      style: { 'line-color': edgeNonsig, 'line-style': 'dashed', width: 1.5, opacity: 0.3 },
    },
    {
      // Fan-out transporter-gene-pair sub-edges (metabolite "graph" expand mode). Each of an
      // interface's parallel sub-edges gets an explicit control point (data 'cpd') so they bow
      // apart into distinct curves instead of overlapping into one line.
      selector: 'edge.gp',
      style: {
        'line-color': edgeSig,
        opacity: 0.9,
        'curve-style': 'unbundled-bezier',
        'control-point-distances': 'data(cpd)',
        'control-point-weights': 0.5,
      },
    },
    {
      // Fan-out self-loops: fan them around the node by per-edge direction (data 'loopDir')
      // with a tight sweep, so many within-cell-type pairs read as separate petals.
      selector: 'edge.gp.self',
      style: {
        'curve-style': 'bezier',
        'loop-direction': 'data(loopDir)',
        'loop-sweep': '26deg',
      },
    },
    {
      selector: 'edge.self',
      style: { 'line-cap': 'round' },
    },
    {
      // The one gene pair currently highlighted from the details panel (hover) or by clicking
      // its sub-edge. Its siblings recede so the pair is findable inside a dense fan; the
      // highlight itself is an accent halo, so significance color is untouched.
      selector: 'edge.gp-dim',
      style: { opacity: 0.16 },
    },
    {
      selector: 'edge.gp-focus',
      style: {
        'overlay-color': accent,
        'overlay-opacity': 0.25,
        'overlay-padding': 4,
        opacity: 1,
        'z-index': 12,
      },
    },
    {
      // Picked (clicked) edge: an accent halo via overlay, keeping the base line-color so
      // significance is still visible. Selection is store-driven (see GraphView `.picked`).
      selector: 'edge.picked',
      style: {
        'overlay-color': accent,
        'overlay-opacity': 0.3,
        'overlay-padding': 5,
        opacity: 1,
        'z-index': 10,
      },
    },
  ] as unknown as cytoscape.StylesheetJson;
}
