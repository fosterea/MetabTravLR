SpaceTravLR
=============================================================================

Spatially perturbing Transcription factors, Ligands and Receptors
------------

The advent of spatial omics has revolutionised our understanding of tissue biology; however, these
technologies remain largely descriptive and do not capture how changes in gene regulation propagate
across spatial neighbourhoods.  


.. image:: https://raw.githubusercontent.com/Koushul/SpaceTravLR/docs/assets/figure1.png
   :width: 1200px
   :align: center

While in-silico perturbation methods and foundation models aim to
model the impact of genetic perturbations, they are limited to single-cell approaches that lack
spatial resolution.

.. image:: https://raw.githubusercontent.com/Koushul/SpaceTravLR/docs/assets/other_methods.png
   :width: 1200px
   :align: center

We address this major unmet need by developing **SpaceTravLR** (*Spatially perturbing Transcription
factors, Ligands and Receptors*), a novel interpretable machine-learning approach that generalises
across tissues and species, uncovering spatial features linked to functional outcomes, thereby
capturing functional microniches with spatial resolution.


.. image:: https://raw.githubusercontent.com/Koushul/SpaceTravLR/docs/assets/spatial_prop.png
   :width: 1200px
   :align: center

SpaceTravLR infers how single or combinatorial genetic perturbations rewire signals across the
tissue neighbourhood, by propagating effects through underlying spatially resolved molecular
networks, thereby modelling how perturbations can reshape both the targeted cell and its surrounding
neighbourhood.


.. include:: _key_contributors.rst

Installation
------------

From pip:

.. code-block:: bash

   pip install SpaceTravLR

Or from source:

.. code-block:: bash

   uv venv
   source .venv/bin/activate
   uv sync


Train
------------
:doc:`quick_start`

You can also train SpaceTravLR with friends by sharing the launch.py file.

And ensuring everyone shares the same output path.

.. code-block:: python

   from SpaceTravLR.spaceship import SpaceShip

   adata = sc.read_h5ad('/data/brains/human_sample_001.h5ad')

   spacetravlr = SpaceShip(
      name='human_brain_tissue_001', 
      outdir='/output'
   )

   spacetravlr.setup_(adata)
   assert spacetravlr.is_everything_ok()

   spacetravlr.spawn_worker(
      partition='GPU-shared',
      clusters='bridges2',
      gres='gpu:1',
      job_name='SpaceTravLR',
      lifespan=1,
      python_path='.conda/envs/space/bin/python'
   ) 


Infer
------------

.. code-block:: python

   spacetravlr.setup_perturbations(
      adata=adata, 
      use_float16=True
   )

   spacetravlr.perturb(
      target='FOXO1',
      propagation=4,
      gene_expr=0,
   )


Analysis
------------

- :doc:`predicting cell state changes in response to chemokines <ligand_perturbation>`
- :doc:`estimating spatially varying gene-gene networks <gradient_tracking>`
- :doc:`predicting the functional consequences of ligand-receptor interactions <ligand_receptor_interactions>`

 



 
Found a 🐞️ or would like to see a feature implemented? Feel free to submit an
`issue <https://github.com/Koushul/SpaceTravLR/issues/new/choose>`_.
 
 
 
.. toctree::
   :caption: Main
   :maxdepth: 1
   :hidden:
 
   about
   understand_the_outputs

.. toctree::
   :caption: API Reference
   :maxdepth: 2
   :hidden:

   api/index
 
.. toctree::
   :caption: Tutorials
   :maxdepth: 2
   :hidden:
 
   quick_start
   ligand_perturbation
   gradient_tracking
   ligand_receptor_interactions

.. .. toctree::
..    :caption: Background
..    :maxdepth: 1
..    :hidden:

..    timeline
..    references




.. |br| raw:: html

  <br/>

.. |dim| raw:: html

   <span class="__dimensions_badge_embed__" data-id="pub.1129830274" data-style="small_rectangle"></span>
   <script async src="https://badge.dimensions.ai/badge.js" charset="utf-8"></script>
