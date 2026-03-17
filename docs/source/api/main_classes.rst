Core
============

SpaceShip — Main way to get where you're going
---------------------------------

:class:`SpaceShip` is the recommended entry point. It orchestrates
data preprocessing, GRN inference, and ligand-receptor communication
and spawning workers into a single configurable object.

.. code-block:: python

   from SpaceTravLR import SpaceShip

   ship = SpaceShip(name='MyTissue', outdir='./output')
   ship.setup_(adata)                 # preprocess, run CellOracle, (optionally COMMOT)
   ship.spawn_worker()                # submit a SLURM job, or call SpaceTravLR directly

.. autoclass:: SpaceTravLR.spaceship.SpaceShip
   :members: spawn_worker
   :undoc-members: False
   :special-members: __init__


SpaceTravLR — Training Orchestrator
-------------------------------------

:class:`SpaceTravLR.oracles.SpaceTravLR` manages the training queue and dispatches
:class:`~SpaceTravLR.models.parallel_estimators.SpatialCellularProgramsEstimator`
for each target gene.

.. code-block:: python

   from SpaceTravLR.oracles import SpaceTravLR

   model = SpaceTravLR(
       adata,
       save_dir='./models',
       annot='cell_type_int',
       grn=grn,
       max_epochs=100,
       radius=200,
   )

   model.run()

.. autoclass:: SpaceTravLR.oracles.SpaceTravLR
   :members: run, imbue_adata_with_space
   :undoc-members: False
   :special-members: __init__


GeneFactory — Perturbation Engine
-----------------------------------

:class:`~SpaceTravLR.gene_factory.GeneFactory` loads pre-trained spatial coefficients
(betas) and uses them to simulate in-silico perturbations.

.. code-block:: python

   from SpaceTravLR.gene_factory import GeneFactory

   gf = GeneFactory.from_json(adata, 'output/params.json')
   gf.load_betas()

   # Single-gene knockout
   result_df = gf.perturb(target='Myc', n_propagation=4, gene_expr=0)

   # Whole-genome screen
   gf.genome_screen(save_to='./ko_results', mode='knockout')

.. autoclass:: SpaceTravLR.gene_factory.GeneFactory
   :members: from_json, load_betas, load_betadata, perturb, perturb_batch, genome_screen, possible_targets, splash_betas
   :undoc-members: False
   :special-members: __init__


VirtualTissue — In-Silico Tissue Simulation
---------------------------------------------

:class:`~SpaceTravLR.virtual_tissue.VirtualTissue` provides a high-level interface for
visualising perturbation effects across the spatial tissue map.

.. code-block:: python

   from SpaceTravLR.virtual_tissue import VirtualTissue

   vt = VirtualTissue(
       adata,
       betadatas_path='./output/betadata',
       ko_path='./ko_results',
   )

   impact = vt.compute_ko_impact(['Myc', 'Sox2'])
   vt.plot_radar(['Myc'], impact_df=impact)
   vt.plot_arrows(perturb_target='Myc', threshold=0.1)

.. autoclass:: SpaceTravLR.virtual_tissue.VirtualTissue
   :members: compute_ko_impact, compute_ko_impact_estimate, plot_arrows, plot_arrows_pseudotime, plot_radar, plot_comparative_radar, plot_comparative_bar, plot_gene_vs_proximity, load_knockout_gex, init_gene_factory, init_cartography
   :undoc-members: False
   :special-members: __init__


SubsampledTissue
~~~~~~~~~~~~~~~~

An extension of :class:`~SpaceTravLR.virtual_tissue.VirtualTissue` that aggregates
results across multiple spatial sub-samples.

.. autoclass:: SpaceTravLR.virtual_tissue.SubsampledTissue
   :members: compute_ko_impact
   :undoc-members: False
   :special-members: __init__


OracleQueue — Training Job Queue
----------------------------------

:class:`~SpaceTravLR.oracles.OracleQueue` manages the set of genes waiting to be
modelled, with file-based locking for safe multi-agent parallel training.

.. autoclass:: SpaceTravLR.oracles.OracleQueue
   :members: remaining_genes, completed_genes, create_lock, delete_lock, kill_old_locks, add_orphan, is_empty, agents
   :undoc-members: False
   :special-members: __init__


.. seealso::

   :doc:`estimator` — full reference for
   :class:`~SpaceTravLR.models.parallel_estimators.SpatialCellularProgramsEstimator`,
   the core per-gene spatial regression engine.



BetaFrame — Spatial Coefficient Matrix
----------------------------------------

:class:`~SpaceTravLR.beta.BetaFrame` is a :class:`pandas.DataFrame` subclass that
stores the spatially-varying regression coefficients (βs) for a single target gene.

.. code-block:: python

   from SpaceTravLR.beta import BetaFrame

   bf = BetaFrame.from_path('output/Myc_betadata.parquet')
   bf_splashed = bf.splash(rw_ligands, rw_ligands_tfl, gex_df)

.. .. autoclass:: SpaceTravLR.beta.BetaFrame
..    :members: from_path, splash
..    :undoc-members: False
..    :special-members: __init__


Betabase — Collection of BetaFrames
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

:class:`~SpaceTravLR.beta.Betabase` manages loading and caching of
:class:`~SpaceTravLR.beta.BetaFrame` objects from disk for all trained genes.

.. autoclass:: SpaceTravLR.beta.Betabase
   :members: load_betas_from_disk, load_betadata, collect_interactions
   :undoc-members: False
   :special-members: __init__


Visionary — Cross-Dataset Prediction
--------------------------------------

:class:`~SpaceTravLR.visionary.Visionary` enables transferring trained spatial models
from a reference dataset to a new test dataset.

.. autoclass:: SpaceTravLR.visionary.Visionary
   :members: reformat, compute_betas, splash_betas
   :undoc-members: False
   :special-members: __init__


CyberBoss — Multi-Resolution Transfer
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

:class:`~SpaceTravLR.visionary.CyberBoss` extends :class:`~SpaceTravLR.visionary.Visionary`
for datasets with different spatial resolutions (e.g. single-cell → Visium spots).

.. autoclass:: SpaceTravLR.visionary.CyberBoss
   :members: reformat, compute_betas
   :undoc-members: False
   :special-members: __init__


Astronaut — Distributed Training Runner
-----------------------------------------

:class:`~SpaceTravLR.astronomer.Astronaut` is a subclass of
:class:`~SpaceTravLR.oracles.SpaceTravLR` that uses pre-computed spatial feature maps
(e.g. ``COVET_SQRT``) rather than deriving them at training time.

.. autoclass:: SpaceTravLR.astronomer.Astronaut
   :members: run
   :undoc-members: False
   :special-members: __init__


