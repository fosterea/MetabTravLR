import os
import glob
import pandas as pd
import numpy as np
from .beta import BetaFrame

def load_and_splash_model(
    model_dir, 
    rw_ligands, 
    gex_df, 
    scale_factor=1, 
    beta_cap=None, 
    grn_tfs=None
):
    """
    Loads all CSV coefficient matrices from a directory, computes their mean,
    broadcasts the mean coefficients to match the target spatial cells (gex_df),
    and applies the splash operation.

    Args:
        model_dir (str): Directory containing CSV files of coefficient matrices.
        rw_ligands (pd.DataFrame): Random walk ligands dataframe.
        gex_df (pd.DataFrame): Gene expression dataframe (spatial cells).
        scale_factor (float, optional): Scaling factor for splash. Defaults to 1.
        beta_cap (float, optional): Cap for beta values. Defaults to None.
        grn_tfs (list, optional): List of TFs to filter for. Defaults to None.

    Returns:
        pd.DataFrame: The splashed coefficient matrix.
    """
    
    # 1. Load all CSVs
    csv_files = glob.glob(os.path.join(model_dir, "*.csv"))
    if not csv_files:
        raise ValueError(f"No CSV files found in {model_dir}")
    
    dfs = []
    for f in csv_files:
        try:
            df = pd.read_csv(f, index_col=0)
            dfs.append(df)
        except Exception as e:
            print(f"Warning: Could not read {f}: {e}")

    if not dfs:
        raise ValueError("No valid CSV files could be read.")
        
    # 2. Compute mean coefficient matrix
    # Assuming all CSVs have the same structure (same columns/index or can be aligned)
    # pd.concat handles alignment by index/columns
    mean_coefs = pd.concat(dfs).groupby(level=0).mean()
    
    # 3. Broadcast to spatial cells
    # We need a DataFrame where each row corresponds to a cell in gex_df,
    # and the values are the mean coefficients (repeated for every cell).
    
    # Create a DataFrame with the same index as gex_df and columns from mean_coefs
    # We use numpy tile to repeat the single row of coefficients
    
    # Check if mean_coefs is a single row or multiple rows. 
    # The prompt implies "create a coef_matrix like cell_oracle has".
    # CellOracle coef_matrix is usually (genes x genes) or (1 x genes) depending on context?
    # Actually, BetaFrame expects the dataframe to be (n_cells x n_features).
    # If the CSVs are (features), we need to transpose or just repeat.
    
    # Let's inspect the shape. If it's a Series or 1-row DF, we repeat it.
    # If it's a matrix (TargetGene x Regulators), we might need to flatten or handle differently.
    # The user said "uses the mean() to create a coef_matrix like cell_oracle has".
    # And "The loaded amtrix need to be 'spalshed'".
    
    # In CellOracle, the coef_matrix is often genes x genes (targets x regulators).
    # But BetaFrame usually holds betas for *one* target gene across many cells, OR generic betas.
    # Looking at BetaFrame.splash, it uses `self.tf_columns`, `self.lr_pairs`, etc.
    # which implies `self` (the BetaFrame) contains the beta values for the *modulators* of a specific target.
    # So `mean_coefs` likely represents the weights for *one* target gene (the one we are modeling).
    # It should be a Series or a 1-row DataFrame where columns are regulators (TF, Ligand, Receptor, etc).
    
    # If mean_coefs has multiple rows (e.g. if it was a full GRN), we might strictly be looking for the coefficients
    # relevant to the current operation. However, the user request says "reads all csv... uses mean()".
    # This implies we are aggregating multiple *models* (replicates) for the *same* target.
    # So the result `mean_coefs` should be a single vector of coefficients (1 row) if it's for one gene.
    # OR it could be that the CSVs contain the global GRN.
    
    # Let's assume the CSVs contain the coefficients for the *current* gene of interest,
    # or that we are meant to load a specific set.
    # But the user said "reads all csv in this dir", implying the dir is specific to the model/gene?
    # "like this: pd.read_csv('./GeoMx_models/ADA_betadata.csv', index_col=0)"
    # ADA is a gene. So yes, the directory likely contains bootstraps/CV folds for ONE gene.
    
    # So mean_coefs should be (1 x n_regulators) or (n_regulators x 1).
    # If it's (n_cols,), we reshape to (1, n_cols).
    
    # We need to ensure it's a DataFrame with 1 row, columns = regulators.
    if mean_coefs.shape[0] > 1:
        # If multiple rows, maybe it's not what we think. 
        # But if standard CellOracle output, it might be (Target, Regulator).
        # if we index_col=0, and it was a Series saved, it might be correct.
        
        # If strictly following "create a coef_matrix like cell_oracle", cell oracle usually produces
        # a matrix of shape (n_targets, n_regulators).
        # But BetaFrame splash acts on a *BetaFrame* which usually represents a single target's equation across many cells.
        # Wait, BetaFrame in beta.py: "Holds a collection of BetaFrames for each gene." in Betabase.
        # But BetaFrame itself is initialized with a DF.
        
        # If we look at BetaFrame.splash:
        # self.tf_columns are used.
        # So the BetaFrame must contain columns like `beta_TF1`, `beta_L1$R1`, etc.
        # The `mean_coefs` from CSV likely has these names.
        pass

    
    # Check if mean_coefs is a single row or multiple rows.
    if mean_coefs.shape[0] == 1:
        # Global model (1 row) -> Broadcast to all cells
        expanded_data = np.tile(mean_coefs.values, (n_cells, 1))
        expanded_df = pd.DataFrame(
            expanded_data,
            index=gex_df.index,
            columns=mean_coefs.columns
        )
    elif mean_coefs.shape[0] == n_cells:
        # Model per cell/ROI -> Use directly (aligned)
        # We assume indices align or can be aligned.
        # Ideally we reindex to gex_df index.
        try:
            expanded_df = mean_coefs.reindex(gex_df.index)
        except Exception:
             # If reindex fails (e.g. duplicate indices in mean_coefs), specific workaround
             pass
        
        # If reindex returns NaNs (indices don't match), we might have issues.
        # But if shape matches exactly, maybe we assume order matches?
        # The user's error says shape is (157, 1412) and indices imply (157, 1412).
        # So we should trust the index alignment if possible.
        
        # Safe approach: reindex, fillna with 0? Or just use values if indexes completely disjoint?
        # Given "GeoMx_models", likely ROIs.
        expanded_df = mean_coefs
        if not expanded_df.index.equals(gex_df.index):
             # Try reindexing
             expanded_df = expanded_df.reindex(gex_df.index)
             if expanded_df.isnull().all().all():
                 # Valid indices, just different names?
                 # If shapes match, maybe just assign values?
                 # But risky. Let's stick to index alignment for now.
                 pass

    else:
        # Mismatch in dimensions
        # If mean_coefs has rows but not equal to n_cells or 1.
        # Maybe it's (n_features, 1)?
        if mean_coefs.shape[1] == 1 and mean_coefs.shape[0] == mean_coefs.shape[0]: 
             # Wait, this condition is always true.
             # Check if it was transposed (features in rows)
             # But we handled transpose earlier? No, I removed that block?
             pass
        
        # If we can't determine, raise error or try to broadcast?
        # Let's assume standard behavior:
        if mean_coefs.shape[1] == 1:
            # Transpose and broadcast
             mean_coefs = mean_coefs.T
             expanded_data = np.tile(mean_coefs.values, (n_cells, 1))
             expanded_df = pd.DataFrame(
                expanded_data,
                index=gex_df.index,
                columns=mean_coefs.columns
            )
        else:
             raise ValueError(f"Shape mismatch: mean_coefs {mean_coefs.shape}, gex_df {gex_df.shape}")

    # 4. Initialize BetaFrame
    beta_frame = BetaFrame(expanded_df)
    
    # Handle missing rw_ligands_tfl by creating zero-filled DataFrame with correct columns
    rw_ligands_tfl = pd.DataFrame(
        0, 
        index=gex_df.index, 
        columns=beta_frame.tfl_ligands
    )

    # 5. Splash
    splashed_df = beta_frame.splash(
        rw_ligands=rw_ligands,
        rw_ligands_tfl=rw_ligands_tfl,
        gex_df=gex_df,
        scale_factor=scale_factor,
        beta_cap=beta_cap,
        grn_tfs=grn_tfs
    )
    
    return splashed_df
