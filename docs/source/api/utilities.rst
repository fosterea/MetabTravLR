Utilities
=========

Helper functions and network utilities used internally by SpaceTravLR and
available for advanced users building custom workflows.


Spatial Utilities
-----------------

.. autofunction:: SpaceTravLR.tools.utils.gaussian_kernel_2d

.. autofunction:: SpaceTravLR.tools.utils.knn_distance_matrix

.. autofunction:: SpaceTravLR.tools.utils.connectivity_to_weights

.. autofunction:: SpaceTravLR.tools.utils.convolve_by_sparse_weights

.. autofunction:: SpaceTravLR.tools.utils.is_mouse_data

.. autofunction:: SpaceTravLR.tools.utils.clean_up_adata

.. autofunction:: SpaceTravLR.models.spatial_map.xyc2spatial_fast




Ligand-Receptor Utilities
--------------------------

.. autofunction:: SpaceTravLR.models.parallel_estimators.received_ligands

.. autofunction:: SpaceTravLR.models.parallel_estimators.init_received_ligands

.. autofunction:: SpaceTravLR.models.parallel_estimators.create_spatial_features

.. autofunction:: SpaceTravLR.models.parallel_estimators.get_filtered_df


Network & GRN Utilities
------------------------

.. autofunction:: SpaceTravLR.tools.network.get_cellchat_db

.. autofunction:: SpaceTravLR.tools.network.expand_paired_interactions

Beta / Coefficient Utilities
------------------------------

.. autofunction:: SpaceTravLR.beta.compute_all_derivatives
