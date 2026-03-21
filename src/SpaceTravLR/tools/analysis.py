# from utils import *
import pandas as pd
import numpy as np
import scanpy as sc
from scipy.spatial import KDTree
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from adjustText import adjust_text
import anndata as ad
from scipy.spatial import cKDTree, KDTree
from scipy.stats import combine_pvalues
from statsmodels.stats.multitest import multipletests
from tqdm import tqdm
from collections import defaultdict

def randomize_delta_X(delta_X, method='permute_rows'):
    """
    Randomize delta_X for permutation null.
    
    method: 'permute_rows' - shuffle which cell gets which perturbation (breaks cell-perturbation link)
            'permute_genes' - shuffle genes within each cell (breaks gene-level structure)
    """
    if method == 'permute_rows':
        perm_idx = np.random.permutation(len(delta_X))
        return delta_X.iloc[perm_idx].set_axis(delta_X.index, axis=0)
    elif method == 'permute_genes':
        vals = delta_X.values.copy()
        for row in vals:
            np.random.shuffle(row)
        return pd.DataFrame(vals, index=delta_X.index, columns=delta_X.columns)
    else:
        raise ValueError(f"method must be 'permute_rows' or 'permute_genes', got {method}")
    
def permutation_test_probabilities(
    chart,
    delta_X,
    embedding,
    n_permutations=100,
    n_neighbors=240,
    annot='banksy_cluster',
    randomize_method='permute_rows',
):
    """
    Permutation test: randomize delta_X, recompute probabilities, compare to observed.
    Returns z_df, p_df.
    """
    unique_zones = chart.adata.obs[annot].unique()
    zone_idxs = {zone: chart.adata.obs[annot] == zone for zone in unique_zones}
    
    def average_probabilities(P_obs, zone_idxs):
        P_obs_avg = defaultdict(dict)
        for orig_zone in unique_zones:
            for trans_zone in unique_zones:
                P_obs_avg[orig_zone][trans_zone] = P_obs[np.ix_(zone_idxs[orig_zone], zone_idxs[trans_zone])].mean()
        return P_obs_avg

    # 1. Observed: compute probabilities from real delta_X
    P_obs = chart.compute_transition_probabilities(
        delta_X, embedding, n_neighbors=n_neighbors, remove_null=False
    )

    P_obs_real = average_probabilities(P_obs, zone_idxs)
    
    # Permutations
    perm_results_list = []
    for i in tqdm(range(n_permutations), desc="Permuting"):
        delta_X_perm = randomize_delta_X(delta_X, method=randomize_method)
        P_perm = chart.compute_transition_probabilities(
            delta_X_perm, embedding, n_neighbors=n_neighbors, remove_null=False
        )
        P_perm_avg = average_probabilities(P_perm, zone_idxs)
        perm_results_list.append(P_perm_avg)
    
    # Convert list to a 3D numpy array: (n_perms, n_zones, n_fates)
    perm_results_arr = np.stack([
        [[x[r][c] for c in unique_zones] for r in unique_zones] for x in perm_results_list
    ])
    
    # Calculate mean and std across the permutation axis (axis=0)
    expected_mean = np.mean(perm_results_arr, axis=0)
    expected_std = np.std(perm_results_arr, axis=0)
    
    # Avoid division by zero for zones with 0 variance
    expected_std[expected_std == 0] = 1
    
    # Observed values as a matrix
    obs_val = np.array([[P_obs_real[r][c] for c in unique_zones] for r in unique_zones])
    
    # Standard Z-score: (Obs - Mean) / SD
    z_scores = (obs_val - expected_mean) / expected_std

    # P-value: Proportion of permutations where the absolute permuted value 
    # is greater than or equal to the absolute observed value (two-tailed)
    p_values = np.mean(
        np.abs(perm_results_arr - expected_mean) > np.abs(obs_val - expected_mean),
        axis=0,
    )

    z_df = pd.DataFrame(z_scores, index=unique_zones, columns=unique_zones)
    p_df = pd.DataFrame(p_values, index=unique_zones, columns=unique_zones)
    
    return z_df, p_df

def permutation_test_transitions(
    chart,
    delta_X,
    embedding,
    n_permutations=100, # Start small to test, then scale up
    n_neighbors=240,
    annot='banksy_cluster',
    thresh=0,
    randomize_method='permute_rows',
):
    """
    Permutation test: randomize delta_X, recompute transitions, compare to observed.
    Returns z_df, p_df.
    """    
    # Save original state to restore later
    original_obs = chart.adata.obs.copy()
    
    # 1. Observed: compute transitions from real delta_X
    if 'transition' in chart.adata.obs.columns:
        chart.adata.obs = chart.adata.obs.drop(columns=['transition'])
        
    P_obs = chart.compute_transition_probabilities(
        delta_X, embedding, n_neighbors=n_neighbors
    )
    
    chart.get_transition_annot(
        P_obs,
        allowed_fates=chart.adata.obs[annot].unique(),
        thresh=thresh,
        annot=annot,
    )
    
    observed = chart.adata.obs.groupby([annot, 'transition']).size().unstack(fill_value=0)
    
    # 2. Permutations
    perm_results_list = []
    
    for i in tqdm(range(n_permutations), desc=f"Permuting ({randomize_method})"):
        delta_X_perm = randomize_delta_X(delta_X, method=randomize_method)
        
        P_perm = chart.compute_transition_probabilities(
            delta_X_perm, embedding, n_neighbors=n_neighbors
        )
        
        # Fresh transition column for this iteration
        if 'transition' in chart.adata.obs.columns:
            chart.adata.obs = chart.adata.obs.drop(columns=['transition'])
            
        chart.get_transition_annot(
            P_perm,
            allowed_fates=chart.adata.obs[annot].unique(),
            thresh=thresh,
            annot=annot,
        )
        
        # Get counts for this permutation and ensure it matches 'observed' columns/index
        counts = chart.adata.obs.groupby([annot, 'transition']).size().unstack(fill_value=0)
        counts = counts.reindex_like(observed).fillna(0)
        perm_results_list.append(counts.values)
    
    # Convert list to a 3D numpy array: (n_perms, n_clusters, n_fates)
    perm_results_arr = np.array(perm_results_list)
    
    # Restore original transition and metadata
    chart.adata.obs = original_obs
    
    # 3. Calculate Z-scores and P-values
    # Calculate mean and std across the permutation axis (axis=0)
    expected_mean = np.mean(perm_results_arr, axis=0)
    expected_std = np.std(perm_results_arr, axis=0)
    
    # Avoid division by zero for clusters with 0 variance
    expected_std[expected_std == 0] = 1
    
    # Observed values as a matrix
    obs_val = observed.values
    
    # Standard Z-score: (Obs - Mean) / SD
    z_scores = (obs_val - expected_mean) / expected_std
    
    # P-value: Proportion of permutations where the absolute permuted value 
    # is greater than or equal to the absolute observed value (two-tailed)
    # Note: For transitions, you might prefer a one-tailed "greater than" test 
    # if you only care about significantly high transitions.
    p_values = np.mean(
        np.abs(perm_results_arr - expected_mean) >= np.abs(obs_val - expected_mean),
        axis=0,
    )
    
    z_df = pd.DataFrame(z_scores, index=observed.index, columns=observed.columns)
    p_df = pd.DataFrame(p_values, index=observed.index, columns=observed.columns)
    
    return z_df, p_df


def get_spatial_perturbation_degs(
    adata, gene, k_neighbors=20, exclude_perturbed=True, cell_type='Neuron', 
    cell_type_col='cell_type', control_label='Control', perturbation_col='perturbation', 
    exclude_source=True, max_distance=np.inf
    ):
    # 0. Basic Validation
    if 'spatial' not in adata.obsm:
        raise ValueError("adata.obsm['spatial'] not found.")
    
    if cell_type and cell_type_col not in adata.obs.columns:
        raise ValueError(f"Cell type column '{cell_type_col}' not found in adata.obs.")

    # 1. Identify Source Cells (can be any cell type)
    perturbed_source_mask = adata.obs[perturbation_col] == gene
    perturbed_source_indices = np.where(perturbed_source_mask)[0]
    
    if len(perturbed_source_indices) == 0:
        raise ValueError(f"No cells found with perturbation for '{gene}'.")
        
    # Clean controls: cells with NO perturbations for ANY sgrna
    unperturbed_mask = adata.obs[perturbation_col] == control_label
    control_source_indices = np.where(unperturbed_mask)[0]
    
    if len(control_source_indices) == 0:
        raise ValueError("No completely unperturbed cells found to serve as control sources.")

    # 2. Find Niche Cells (Neighbors)
    # We query ALL cells first, then filter. 
    # k+1 to allow for the possibility that the source cell is in the tree.
    spatial_coords = adata.obsm['spatial']
    tree = KDTree(spatial_coords)
    
    # Cap k_neighbors at total cells - 1
    actual_k = min(k_neighbors, adata.n_obs - 1)
    
    _, neighbor_idx_p = tree.query(spatial_coords[perturbed_source_indices], k=actual_k + 1, distance_upper_bound=max_distance)
    _, neighbor_idx_c = tree.query(spatial_coords[control_source_indices], k=actual_k + 1, distance_upper_bound=max_distance)
    
    # Flatten and get unique indices
    niche_perturbed_idx = np.unique(neighbor_idx_p.flatten())
    niche_control_idx = np.unique(neighbor_idx_c.flatten())

    # 3. Filtration
    # A. Exclude the source cells themselves (we want bystander effects only)
    if exclude_source:
        niche_perturbed_idx = np.setdiff1d(niche_perturbed_idx, perturbed_source_indices)
        # niche_control_idx = np.setdiff1d(niche_control_idx, control_source_indices)
    
    # B. Bystander filtration: remove any cell that has ANY perturbation
    if exclude_perturbed:
        all_perturbed_indices = np.where(~unperturbed_mask)[0]
        niche_perturbed_idx = np.setdiff1d(niche_perturbed_idx, all_perturbed_indices)
        niche_control_idx = np.setdiff1d(niche_control_idx, all_perturbed_indices)
    
    # C. Cell Type filtration: only keep neighbors of the specified cell type
    if cell_type:
        type_mask = adata.obs[cell_type_col] == cell_type
        type_indices = np.where(type_mask)[0]
        niche_perturbed_idx = np.intersect1d(niche_perturbed_idx, type_indices)
        niche_control_idx = np.intersect1d(niche_control_idx, type_indices)

    # 4. Label and Test
    label_col = f'temp_spatial_cond_{gene}'
    adata.obs[label_col] = 'none'
    
    # a neighbor to perturbed and control is just perturbed neighbor
    niche_control_idx = np.setdiff1d(niche_control_idx, niche_perturbed_idx)
    
    # Assign labels using positional indices
    adata.obs.iloc[niche_control_idx, adata.obs.columns.get_loc(label_col)] = 'control_neighbor'
    adata.obs.iloc[niche_perturbed_idx, adata.obs.columns.get_loc(label_col)] = 'perturbed_neighbor'
    
    n_p = len(niche_perturbed_idx)
    n_c = len(niche_control_idx)
    
    if n_p < 3 or n_c < 3:
        adata.obs.drop(columns=[label_col], inplace=True)
        raise ValueError(f"Insufficient niche {cell_type} cells found for '{gene}' (P: {n_p}, C: {n_c}).")
        
    sc.tl.rank_genes_groups(
        adata, 
        groupby=label_col, 
        groups=['perturbed_neighbor'], 
        reference='control_neighbor',
        method='wilcoxon',
        key_added=f'rank_genes_{gene}'
    )
    
    # 5. Extract results
    deg_df = sc.get.rank_genes_groups_df(adata, group='perturbed_neighbor', key=f'rank_genes_{gene}')
    
    # Cleanup
    adata.obs.drop(columns=[label_col], inplace=True)
    
    return deg_df

def aggregate_degs(deg_dfs, sign_consistency_frac=0.7, sig_frac=0.7, sig_threshold=0.05):
    # 1. Initialize DataFrames from the input dictionary
    all_pvals = pd.DataFrame({
        batch: df.set_index('gene')['pval']
        for batch, df in deg_dfs.items()
    }).apply(pd.to_numeric, errors='coerce').fillna(1.0)
    
    all_pvals_adj = pd.DataFrame({
        batch: df.set_index('gene')['pval_adj']
        for batch, df in deg_dfs.items()
    }).apply(pd.to_numeric, errors='coerce')
    
    all_lfc = pd.DataFrame({
        batch: df.set_index('gene')['log2fc']
        for batch, df in deg_dfs.items()
    }).apply(pd.to_numeric, errors='coerce')
    
    all_scores = pd.DataFrame({
        batch: df.set_index('gene')['score']
        for batch, df in deg_dfs.items()
    })

    # 2. Calculate sign consistency (Fraction of majority)
    n_batches_total = all_lfc.notna().sum(axis=1)
    lfc_signs = np.sign(all_lfc)
    n_pos = (lfc_signs == 1).sum(axis=1)
    n_neg = (lfc_signs == -1).sum(axis=1)
    n_total = n_batches_total.replace(0, np.nan)
    sign_consistency = np.maximum(n_pos, n_neg) / n_total
    
    # 3. Calculate significance metrics for filtering
    is_sig = all_pvals_adj < sig_threshold
    n_batches_sig = is_sig.sum(axis=1)

    # 4. Filter for concordant genes
    min_batches = sig_frac * len(deg_dfs)
    concordant_mask = (sign_consistency >= sign_consistency_frac) & (n_batches_sig >= min_batches)
    concordant_genes = sign_consistency[concordant_mask].dropna().index
    
    if len(concordant_genes) == 0:
        return pd.DataFrame(columns=['gene', 'log2fc', 'score', 'pval', 'pval_adj', 'n_batches', 'n_batches_sig', 'sign_consistency'])

    # Slice data to only include concordant genes
    all_pvals = all_pvals.loc[concordant_genes]
    all_lfc = all_lfc.loc[concordant_genes]
    all_scores = all_scores.loc[concordant_genes]

    # 5. Fisher's method for p-value aggregation
    combined_pvals = {}
    for gene in all_pvals.index:
        pvals = all_pvals.loc[gene].dropna().values
        pvals = np.clip(pvals, 1e-300, 1.0)
        _, p_combined = combine_pvalues(pvals, method='fisher')
        combined_pvals[gene] = p_combined
    
    genes = list(combined_pvals.keys())
    pvals_arr = np.array([combined_pvals[g] for g in genes])
    
    _, pvals_adj, _, _ = multipletests(pvals_arr, method='fdr_bh')
    
    # 6. Construct Result
    result = pd.DataFrame({
        'gene': genes,
        'log2fc': all_lfc.mean(axis=1).values,
        'score': all_scores.mean(axis=1).values,
        'pval': pvals_arr,
        'pval_adj': pvals_adj,
        'n_batches': n_batches_total.loc[genes].values,
        'n_batches_sig': n_batches_sig.loc[genes].values,
        'sign_consistency': sign_consistency.loc[genes].values,
    }).sort_values('pval_adj')
    
    return result

def find_neighbors(adata_perturb, cell_indices, n_neighbors, cell_target = None, max_distance = None):
    neighbors_df = []

    coords = adata_perturb.obsm['spatial']
    tree = cKDTree(coords)

    for cell_id in cell_indices:

        if isinstance(cell_id, str):
            query_idx = adata_perturb.obs_names.get_loc(cell_id)
        else:
            query_idx = cell_id

        query_coord = coords[query_idx]
        distances, indices = tree.query(query_coord, k=n_neighbors + 1)
        neighbor_indices = indices[1:]
        neighbor_distances = distances[1:]
        neighbor_barcodes = adata_perturb.obs_names[neighbor_indices]
        cell_types = adata_perturb.obs['cell_type'].iloc[neighbor_indices]

        df = pd.DataFrame({
            'cell_barcode': neighbor_barcodes,
            'cell_index': neighbor_indices,
            'distance': neighbor_distances,
            'cell_type': cell_types,
            # 'batch': adata_perturb.obs['batch'].iloc[neighbor_indices]
        })

        if cell_target is not None:
            df = df.query(f'cell_type == "{cell_target}"')
        if max_distance is not None:
            df = df.query('distance < @max_distance')
        neighbors_df.append(df)

    return pd.concat(neighbors_df)

def find_de_genes(adata1, adata2, label1='Experimental', label2='SpaceTravLR', method='wilcoxon'):
    a1 = adata1.copy()
    a2 = adata2.copy()

    if a1.shape[0] == 0 or a2.shape[0] == 0:
        de_df = pd.DataFrame(columns=['gene', 'log2fc', 'pval', 'pval_adj', 'score'])
        de_df.columns = ['gene', 'log2fc', 'pval', 'pval_adj', 'score']
        de_df['abs_log2fc'] = np.abs(de_df['log2fc'])
        return de_df
    
    condition_key = 'de_comparison_group'
    a1.obs[condition_key] = label1
    a2.obs[condition_key] = label2
    
    combined = ad.concat([a1, a2], join='outer', merge='same')

    sc.tl.rank_genes_groups(combined, groupby=condition_key, groups=[label1], reference=label2, method=method, use_raw=False)
    de_df = sc.get.rank_genes_groups_df(combined, group=label1)
    de_df = de_df[['names', 'logfoldchanges', 'pvals', 'pvals_adj', 'scores']]
    de_df.columns = ['gene', 'log2fc', 'pval', 'pval_adj', 'score']
    de_df['abs_log2fc'] = np.abs(de_df['log2fc'])
    return de_df

def plot_embedding(data, ax, guide_col, clusters, size=4, show_legend=False):
    # Create a dataframe for seaborn
    plot_df = pd.DataFrame(
        data.obsm['spatial'], 
        columns=['x', 'y'], 
        index=data.obs_names
    )
    plot_df['leiden'] = data.obs['leiden'].values

    # Add guide info
    # guide_col = select_guide.value
    if guide_col and guide_col in data.obs.columns:
        plot_df['guide'] = data.obs[guide_col].values > 0
    else:
        plot_df['guide'] = False

    # Plot background (points not in groups)
    bg_mask = ~plot_df['leiden'].isin(clusters)
    sns.scatterplot(
        data=plot_df[bg_mask],
        x='x', y='y',
        color='lightgrey',
        s=size,
        linewidth=0.075,
        ax=ax,
        legend=False
    )

    # Plot foreground (points in groups)
    fg_mask = plot_df['leiden'].isin(clusters)
    if fg_mask.any():
        sns.scatterplot(
            data=plot_df[fg_mask],
            x='x', y='y',
            hue='leiden',
            palette='tab20',
            hue_order=clusters,
            s=size,
            linewidth=0.075,
            edgecolor='black',
            ax=ax,
            legend=show_legend
        )

    # Plot guide cells
    if plot_df['guide'].any():
        guide_mask = plot_df['guide']
        sns.scatterplot(
            data=plot_df[guide_mask],
            x='x', y='y',
            color='red',
            s=30,
            linewidth=0.5,
            edgecolor='black',
            ax=ax,
            label=guide_col if show_legend else None,
            legend=show_legend
        )

    ax.set_axis_off()
    ax.set_aspect('equal')

def plot_gene_comparison_advanced(df1, df2, 
                                  label1="Experimental", 
                                  label2="SpaceTravLR", 
                                  highlight_genes=None, 
                                  top_n_labels=100,
                                  figsize=(10, 10),
                                  target_ko='',
                                  alpha=0.9,
                                  savepath=None):

    merged = pd.merge(df1[['gene', 'log2fc']], 
                      df2[['gene', 'log2fc']], 
                      on='gene', 
                      suffixes=(f'_{label1}', f'_{label2}'))
    
    x_col = f'log2fc_{label1}'
    y_col = f'log2fc_{label2}'
    merged = merged.dropna(subset=[x_col, y_col])

    conditions = [
        (merged[x_col] > 0) & (merged[y_col] > 0),
        (merged[x_col] < 0) & (merged[y_col] < 0),
        (merged[x_col] > 0) & (merged[y_col] < 0),
        (merged[x_col] < 0) & (merged[y_col] > 0)
    ]
    choices = ['Concordant Up', 'Concordant Down', 'Discordant', 'Discordant']
    merged['Category'] = np.select(conditions, choices, default='Neutral')
    merged['magnitude'] = np.sqrt(merged[x_col]**2 + merged[y_col]**2)
    
    pearson_r, p_val = stats.pearsonr(merged[x_col], merged[y_col])
    spearman_r, _ = stats.spearmanr(merged[x_col], merged[y_col])
    
    fig, ax = plt.subplots(figsize=figsize, dpi=250)
    sns.set_style("ticks")
    
    palette = {
        'Concordant Up': '#16E0BD',
        'Concordant Down': '#FC6471',
        'Discordant': '#8d8dec',
        'Neutral': '#DDDDDD'
    }
    
    sns.scatterplot(
        data=merged, x=x_col, y=y_col, 
        hue='Category', palette=palette, 
        alpha=alpha, s=80, edgecolor='white', linewidth=0.5,
        ax=ax, legend='brief'
    )
    
    limit = max(merged[x_col].abs().max(), merged[y_col].abs().max()) * 1.1
    ax.plot([-limit, limit], [-limit, limit], ls="--", 
    color="gray", alpha=0.75, lw=1.5, label="x=y")
    ax.axhline(0, color='black', lw=1, alpha=0.2)
    ax.axvline(0, color='black', lw=1, alpha=0.2)

    texts = []
    labeled_genes = set()
    
    if highlight_genes is not None:
        subset = merged[merged['gene'].isin(highlight_genes)]
        for _, row in subset.iterrows():
            texts.append(ax.text(row[x_col], row[y_col], row['gene'], 
                                 fontsize=10, fontweight='bold', color='black'))
            labeled_genes.add(row['gene'])
            ax.scatter(row[x_col], row[y_col], s=150, facecolors='none', edgecolors='black', lw=1.5)

    if top_n_labels > 0:
        top_mag = merged.nlargest(top_n_labels + len(labeled_genes), 'magnitude')
        for _, row in top_mag.iterrows():
            if row['gene'] not in labeled_genes:
                texts.append(ax.text(row[x_col], row[y_col], row['gene'], 
                                     fontsize=9, color='#333333'))
                labeled_genes.add(row['gene'])

    adjust_text(texts, arrowprops=dict(arrowstyle='-', 
        color='gray', lw=0.5),
        # force_explode=(1, 1),
        #         expand_points=(0.1, 0.1)
                )

    stats_text = (f"Spearman $\\rho$: {spearman_r:.2f} (N={len(merged)})")
    
    q_counts = merged['Category'].value_counts()
    ax.text(limit*0.5, limit*0.8, f"Both Up\nn={q_counts.get('Concordant Up', 0)}", 
            ha='center', va='center', fontsize=6, color=palette['Concordant Up'], fontweight='bold')
    ax.text(-limit*0.5, -limit*0.8, f"Both Down\nn={q_counts.get('Concordant Down', 0)}", 
            ha='center', va='center', fontsize=6, color=palette['Concordant Down'], fontweight='bold')

    ax.set_title(f'DEGs in neighbors of {target_ko} KO\n(Spearman: {spearman_r:.2f}, N={len(merged)})', 
    fontsize=13, pad=5)


    ax.set_xlabel(f'Log2FC ({label1})', fontsize=10)
    ax.set_ylabel(f'Log2FC ({label2})', fontsize=10)
    
    ax.set_xlim(-limit, limit)
    ax.set_ylim(-limit, limit)
    
    handles, labels = ax.get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    ax.legend(by_label.values(), by_label.keys(), 
    loc='lower right', frameon=True, fontsize=8)
    
    ax.get_legend().remove()

    ax.grid(True, alpha=0.25)
    
    sns.despine(offset=10, trim=False)
    plt.tight_layout()
    if savepath:
        plt.savefig(savepath, dpi=300)
    plt.show()