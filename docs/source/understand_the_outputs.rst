Understanding outputs
=================================

Output directory structure
--------------------------

.. code-block:: text

    output/
    ├── input_data/
    │   ├── _adata.h5ad
    │   ├── celloracle_links.pkl
    │   ├── communication.pkl
    │   ├── LRs.parquet
    ├── betadata/
    │   ├── PAX5_betadata.parquet
    │   ├── FOXO1_betadata.parquet
    │   ├── CD79A_betadata.parquet
    │   ├── ...
    │   ├── IL21_betadata.parquet
    │   ├── IL4_betadata.parquet
    │   ├── CCR4_betadata.parquet
    ├── logs/
    │   ├── training_TIMESTAMP.log


Understanding beta coefficients
--------------------------

Here, each _betadata.parquet has shape cell x modulators. 
For example, the column names are labeled as 'beta_EGR1' for the transcription factor EGR1,
'beta_IL21$IL21R' for the ligand-receptor interaction between IL21 and IL21R, and
'beta_IL27#PATZ1' for the transcriptional association of PATZ1 and IL27. 

These values represent the estimated modulatory effect of each modulator gene/gene pair on the target gene. 
(_betadata.parquet).index.name.

In essense, this also represents the spatially varying links between genes in the network.



Understanding UMAP perturbation plots
--------------------------



Understanding radar plots
--------------------------


