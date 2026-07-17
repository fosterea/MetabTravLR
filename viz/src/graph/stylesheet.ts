/** Cytoscape stylesheet, built from theme tokens. See docs/02_style_and_conventions.md.
 *  Edge WIDTH encodes strength (data 'w'); edge COLOR/dash encodes significance. */
import type cytoscape from 'cytoscape';
import { cssVar } from './cssVars';

export function buildStylesheet(): cytoscape.StylesheetJson {
  const textPrimary = cssVar('--text-primary');
  const tcell = cssVar('--cell-tcell');
  const tcellInk = cssVar('--cell-tcell-ink');
  const other = cssVar('--cell-other');
  const otherInk = cssVar('--cell-other-ink');
  const edgeSig = cssVar('--edge-sig');
  const edgeNonsig = cssVar('--edge-nonsig');
  const accent = cssVar('--accent');
  const border = cssVar('--border');

  return [
    {
      selector: 'node',
      style: {
        label: 'data(label)',
        'font-size': 13,
        // Cytoscape draws to canvas and rejects CSS keyword/quoted font stacks
        // (e.g. ui-sans-serif, 'Segoe UI'); use a plain family list.
        'font-family': 'Helvetica Neue, Arial, sans-serif',
        color: textPrimary,
        'text-valign': 'center',
        'text-halign': 'center',
        'text-wrap': 'wrap',
        'text-max-width': '90px',
        width: 46,
        height: 46,
        'border-width': 2,
        'border-color': border,
      },
    },
    {
      selector: 'node.fam-tcell',
      style: { 'background-color': tcell, color: tcellInk },
    },
    {
      selector: 'node.fam-other',
      style: { 'background-color': other, color: otherInk },
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
