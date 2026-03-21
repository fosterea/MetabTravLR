Model
================================

:class:`~SpaceTravLR.models.parallel_estimators.SpatialCellularProgramsEstimator`
is the core per-gene fitting engine used internally by
:class:`~SpaceTravLR.oracles.SpaceTravLR`.  It fits a spatially-aware
regulatory model for a **single target gene** using three classes of input
features:

.. list-table:: Input feature groups
   :widths: 25 40 35
   :header-rows: 1

   * - Group
     - Description
     - Column format
   * - **Regulators** (TFs)
     - Transcription-factor expression from a GRN lookup
     - ``GeneSymbol``
   * - **Ligand-receptor pairs**
     - Gaussian-diffused ligand signal x receptor expression
     - ``LigandGene$ReceptorGene``
   * - **Ligand-TF pairs** (NicheNet)
     - Diffused ligand signal x TF expression (NicheNet-filtered)
     - ``LigandGene#TFGene``

The model produces per-cell *spatial beta* coefficients — one column per
modulator plus an intercept — describing how regulatory influence varies
across the tissue.

**Training pipeline overview**

For every cell-type cluster the estimator:

1. Fits a seed linear model (**Group Lasso** / Bayesian Ridge / ARD) on
   the cluster cells to obtain coefficient anchors.
2. Constructs a 2-D spatial neighbourhood image (side ``spatial_dim``)
   capturing local cell-type densities.
3. Trains a **CellularNicheNetwork** (CNN) or **CellularViT** (ViT)
   conditioned on those anchors and neighbourhood images.
4. If the final R² falls below ``score_threshold``, zeroes the anchors
   so the gene betas collapse to the global baseline.

.. code-block:: python

   from SpaceTravLR.models.parallel_estimators import SpatialCellularProgramsEstimator
   from SpaceTravLR.tools.network import RegulatoryFactory

   grn = RegulatoryFactory(colinks_path="colinks.csv", annot="cell_type_int")

   estimator = SpatialCellularProgramsEstimator(
       adata,
       target_gene="Myc",
       grn=grn,
       radius=150,
       spatial_dim=64,
   )

   # Train one model per cell-type cluster
   estimator.fit(num_epochs=80, estimator="lasso", score_threshold=0.2)

   # Retrieve per-cell spatial betas -- shape (n_cells, 1 + n_modulators)
   betas = estimator.get_betas()

   # Visualise regulators as a colour-coded word cloud
   estimator.plot_modulators()

   # Persist and reload
   estimator.export("./output/models")
   estimator.load("./output/models/Myc_estimator.pkl")

.. autoclass:: SpaceTravLR.models.parallel_estimators.SpatialCellularProgramsEstimator
   :members: __init__, fit, init_data, get_betas, betadata, plot_modulators,
             ligands_receptors_interactions, ligand_regulators_interactions,
             check_LR_properties, export, load
   :member-order: bysource
   :undoc-members: False
   :special-members: __init__
