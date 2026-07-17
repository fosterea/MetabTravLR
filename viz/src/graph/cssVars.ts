/** Read design tokens from :root so Cytoscape (which needs concrete colors) stays in sync
 *  with theme.css. Re-read on each stylesheet build so light/dark tokens are respected. */
export function cssVar(name: string): string {
  return getComputedStyle(document.documentElement).getPropertyValue(name).trim();
}
