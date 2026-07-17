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
        width: 40,
        height: 40,
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
        'line-color': edgeNonsig,
        opacity: 0.9,
        'loop-direction': '-45deg',
        'loop-sweep': '-30deg',
      },
    },
    {
      selector: 'edge.sig',
      style: { 'line-color': edgeSig, opacity: 1 },
    },
    {
      selector: 'edge.nonsig',
      style: { 'line-color': edgeNonsig, 'line-style': 'dashed', opacity: 0.5 },
    },
    {
      selector: 'edge.self',
      style: { 'line-cap': 'round' },
    },
    {
      selector: ':selected',
      style: { 'border-color': accent, 'border-width': 4, 'line-color': accent },
    },
  ] as unknown as cytoscape.StylesheetJson;
}
