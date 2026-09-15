"""Build the x (design-matrix factor) building blocks for a linear-regression version of
SpaceTravLR: metabolite communication scores + the diffused-ligand table, attached to a
PROCESSED adata. Thin wrappers over existing, tested code -- no new math.
"""
import sys
from pathlib import Path

# Make the repo root (and src/) importable regardless of CWD, matching the other
# metab_processing/SpaceTravLR scripts.
_root = next(
    (p for p in Path(__file__).resolve().parents
     if (p / ".git").exists() or (p / "setup.py").exists()),
    Path(__file__).resolve().parents[2],
)
for _p in (str(_root), str(_root / "src")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from metab_processing.SpaceTravLR import beta_analysis


def add_x_data(adata, metabolites=None, radius=100, contact_distance=30,
               scale_factor=100, layer='imputed_count'):
    """Attach the x building blocks to a PROCESSED adata (needs `layer`, obsm['spatial'],
    obs['cell_type_int']). Returns adata.

    - obsm['x_metab'] + uns['x_metab_modulators']: metabolite scores (if `metabolites`),
      via beta_analysis.metab_x_to_adata.
    - obsm['received_ligands'] + uns['received_ligands_cols']: the UNFILTERED CellChat
      received-ligand diffusion at `radius`, computed fresh (cell_threshes=None). This matches
      training's L-R / L-TF / metab received-ligand source when COMMOT is OFF (the project
      default). If COMMOT thresholds are present, training would filter the L-R source by them;
      that filtering is NOT applied here (a warning is emitted).
    """
    # Lazy import: pulls in torch (via parallel_estimators) -- keep the module importable
    # without it, mirroring beta_analysis.py.
    from SpaceTravLR.models.parallel_estimators import init_received_ligands

    if metabolites:
        # NOTE: metab_x_to_adata runs its own (metab-pair) diffusion; the call below is a
        # second, independent full diffusion for the CellChat building block. Both use the
        # sparse row-chunked kernel (tractable); ~2x pass, acceptable for now.
        beta_analysis.metab_x_to_adata(adata, metabolites, radius, contact_distance,
                                       scale_factor, layer)

    if 'cell_thresholds' in adata.uns:
        import warnings
        warnings.warn(
            "cell_thresholds present (COMMOT): the stored received_ligands building block is "
            "the UNFILTERED diffusion; training's COMMOT L-R filtering is not reflected.")

    init_received_ligands(
        adata, radius, cell_threshes=None,
        contact_distance=contact_distance, scale_factor=scale_factor,
        layer=layer, extra_lr=None,
    )
    frame = adata.uns['received_ligands_tfl']
    adata.obsm['received_ligands'] = frame.to_numpy()
    adata.uns['received_ligands_cols'] = list(frame.columns)
    # Drop the big cells x ligands frames unconditionally so the saved h5ad doesn't balloon
    # (the info now lives in obsm); this artifact is freshly built, not a cache to preserve.
    adata.uns.pop('received_ligands_tfl', None)
    adata.uns.pop('received_ligands', None)
    return adata


def _preprocess(adata, annot='cell_type', min_cells_for_magic=16, layer_added='imputed_count'):
    """In-house preprocessing: scale spatial coords, encode `cell_type_int`, and MAGIC-impute
    into `layer_added`. Mirrors `SpaceShip.process_adata_` MINUS the UMAP (CellOracle-only) and
    the file-tree writes -- we only need what the x computation consumes. Mutates `adata`.
    """
    import scanpy as sc
    from SpaceTravLR.tools.utils import scale_adata
    from SpaceTravLR.tools.network import encode_labels
    from SpaceTravLR.oracles import BaseTravLR

    adata.obs['cell_type'] = adata.obs[annot]
    scale_adata(adata)  # scales obsm['spatial'] so `radius` is in the expected units
    mapping = encode_labels(adata.obs['cell_type'], reverse_dict=True)
    adata.obs['cell_type_int'] = adata.obs['cell_type'].map(mapping).astype(int)

    if layer_added not in adata.layers:
        if 'normalized_count' not in adata.layers:
            if adata.X.max() > 100:
                sc.pp.log1p(adata)
            adata.layers['normalized_count'] = adata.X.copy()
        BaseTravLR.impute_clusterwise(
            adata, annot='cell_type', layer='normalized_count',
            layer_added=layer_added, min_cells_for_magic=min_cells_for_magic,
        )
        adata.layers.pop('normalized_count', None)
    return adata


def build_x_adata(adata, out_path, *, annot='cell_type', metabolites=None,
                  radius=100, contact_distance=30, scale_factor=100,
                  min_cells_for_magic=16, layer='imputed_count'):
    """End-to-end: preprocess (impute) + `add_x_data` + write a single adata to `out_path`.

    Writes ONLY the one adata (imputed-count layer + the x data) -- no SpaceTravLR output tree,
    no CellOracle/NicheNet networks (those aren't needed for the stored x building blocks).
    Returns the written adata.
    """
    adata = adata.copy()
    _preprocess(adata, annot=annot, min_cells_for_magic=min_cells_for_magic, layer_added=layer)
    add_x_data(adata, metabolites, radius, contact_distance, scale_factor, layer)

    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    adata.write_h5ad(out_path)
    print(f'[build_x_adata] wrote {out_path}')
    return adata
