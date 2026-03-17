# refer to /scvelo/plotting/velocity_embedding_grid.py
from collections import Counter
from functools import cache, lru_cache
from matplotlib.colors import LinearSegmentedColormap
import pandas as pd

from matplotlib import pyplot as plt
from scipy.spatial import Delaunay
import numpy as np
import numpy as np
from sklearn.neighbors import NearestNeighbors
from velocyto.estimation import colDeltaCor, colDeltaCorpartial
from tqdm import tqdm
import seaborn as sns
from scipy.spatial import KDTree
import cellrank as cr
from .shift import estimate_transition_probabilities, project_probabilities
from .layout import get_grid_layout, plot_quiver
import glob
from .modplot import velovect, animate_velovect
from scipy.interpolate import griddata
from sklearn.neighbors import KNeighborsRegressor
from adjustText import adjust_text


def normalize_gradient(gradient, method="sqrt"):
    if method == "sqrt":

        size = np.sqrt(np.power(gradient, 2).sum(axis=1))
        size_sq = np.sqrt(size)
        size_sq[size_sq == 0] = 1
        factor = np.repeat(np.expand_dims(size_sq, axis=1), 2, axis=1)

    return gradient / factor

def get_gradient(value_on_grid):
    n = int(np.sqrt(value_on_grid.shape[0]))
    value_on_grid_as_matrix = value_on_grid.reshape(n, n)
    dy, dx = np.gradient(value_on_grid_as_matrix)
    gradient = np.stack([dx.flatten(), dy.flatten()], axis=1)

    return gradient

# def alpha_shape(points, alpha, only_outer=True):
#     assert points.shape[0] > 3, "Need at least four points"
#     def add_edge(edges, i, j):
#         """
#         Add an edge between the i-th and j-th points,
#         if not in the list already
#         """
#         if (i, j) in edges or (j, i) in edges:
#             # already added
#             assert (j, i) in edges, "Can't go twice over same directed edge right?"
#             if only_outer:
#                 # if both neighboring triangles are in shape, it's not a boundary edge
#                 edges.remove((j, i))
#             return
#         edges.add((i, j))
#     tri = Delaunay(points)
#     edges = set()
#     # Loop over triangles:
#     # ia, ib, ic = indices of corner points of the triangle
#     for ia, ib, ic in tri.simplices:
#         pa = points[ia]
#         pb = points[ib]
#         pc = points[ic]
#         # Computing radius of triangle circumcircle
#         # www.mathalino.com/reviewer/derivation-of-formulas/derivation-of-formula-for-radius-of-circumcircle
#         a = np.sqrt((pa[0] - pb[0]) ** 2 + (pa[1] - pb[1]) ** 2)
#         b = np.sqrt((pb[0] - pc[0]) ** 2 + (pb[1] - pc[1]) ** 2)
#         c = np.sqrt((pc[0] - pa[0]) ** 2 + (pc[1] - pa[1]) ** 2)
#         s = (a + b + c) / 2.0
#         area = np.sqrt(s * (s - a) * (s - b) * (s - c))

#         circum_r = a * b * c / (4.0 * area)
#         if circum_r < alpha:
#             add_edge(edges, ia, ib)
#             add_edge(edges, ib, ic)
#             add_edge(edges, ic, ia)
    
#     return edges

def xy_from_adata(adata):
    return pd.DataFrame(
        adata.obsm['spatial'], 
        columns=['x', 'y'], 
        index=adata.obs_names
    )

def get_cells_within_radius(df, indices, radius):
    result_indices = set()
    for idx in indices:
        x, y = df.loc[idx, ['x', 'y']]
        distances = np.sqrt((df['x'] - x) ** 2 + (df['y'] - y) ** 2)
        within_radius = df[distances <= radius].index
        result_indices.update(within_radius)
    return list(result_indices)

def plot_cells(df, indices, radius):
    cells_within_radius = get_cells_within_radius(df, indices, radius)
    
    plt.scatter(df['x'], df['y'], color='grey', s=4, label='NA')
    plt.scatter(df.loc[cells_within_radius, 'x'], 
                df.loc[cells_within_radius, 'y'], color='red', s=4, label='Within Radius')
    plt.scatter(df.loc[indices, 'x'], df.loc[indices, 'y'], color='blue', s=4, label='Given Indices')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.legend()
    plt.show()
    
    return cells_within_radius

    
class Cartography:
    def __init__(self, adata, color_dict, base_layer='imputed_count'):
        self.adata = adata
        self.xy = xy_from_adata(adata)
        self.base_layer = base_layer
        self.unperturbed_expression = adata.to_df(layer=base_layer)
        self.color_dict = color_dict
                
    def compute_perturbation_corr(self, gene_mtx, delta_X, embedding=None, k=200):
        if embedding is None:
            
            corr = colDeltaCor(
                np.ascontiguousarray(gene_mtx.T), 
                np.ascontiguousarray(delta_X.T), 
            )
        
        else:
            nn = NearestNeighbors(n_neighbors=k+1)
            nn.fit(embedding)
            _, indices = nn.kneighbors(embedding)

            indices = indices[:, 1:] # remove self transition

            corr = colDeltaCorpartial(
                np.ascontiguousarray(gene_mtx.T), 
                np.ascontiguousarray(delta_X.T), 
                indices
            )

            self.nn_indices = indices
       
        corr = np.nan_to_num(corr, nan=1)
        return corr


    def load_perturbation(self, perturb_target, betadata_path):
        perturbed_df = pd.read_parquet(
            f'{betadata_path}/{perturb_target}_4n_0x.parquet')
        self.adata.layers[perturb_target] = perturbed_df.loc[self.adata.obs.index, self.adata.var.index].values
    
    
    def get_corr(self, perturb_target, embedding_label=None, k=200):
        assert perturb_target in self.adata.layers
        delta_X = (self.adata.to_df(layer=perturb_target) - self.unperturbed_expression).values
        gene_mtx = self.unperturbed_expression.values

        if embedding_label is not None:
            assert embedding_label in self.adata.obsm
            embedding = self.adata.obsm[embedding_label]
        else:
            embedding = None

        return self.compute_perturbation_corr(gene_mtx, delta_X, embedding, k)
    

    def compute_transitions(self, corr_mtx, source_ct, annot='cell_type'):

        n_cells = self.adata.shape[0]

        if hasattr(self, "nn_indices"):
            P = np.zeros((n_cells, n_cells))
            row_idx = np.repeat(np.arange(n_cells), self.nn_indices.shape[1])
            col_idx = self.nn_indices.ravel()
            P[row_idx, col_idx] = 1
        else:
            P = np.ones((n_cells, n_cells))

        T = 0.05
        np.fill_diagonal(P, 0)
        P *= np.exp(corr_mtx / T)   
        P /= P.sum(1)[:, None]

        mask = np.where(corr_mtx <= 0, 0, 1) # if corr was negative or zero, it should not be a transition
        P *= mask
        
        transition_df = pd.DataFrame(P[self.adata.obs[annot] == source_ct])
        transition_df.columns = self.adata.obs_names
        transition_df.columns.name = source_ct
        return transition_df
    
    @staticmethod
    def assess_transitions(transition_df, base_celltypes, source_ct, annot):
        rx = transition_df.T.join(base_celltypes).groupby(annot).mean()
        rx.columns.name = source_ct
        range_df = pd.DataFrame([rx.min(1), rx.mean(1), rx.max(1)], index=['min', 'mean', 'max']).T
        range_df.columns.name = f'Source Cells: {source_ct}'
        range_df.index.name = 'Transition Target'
        return range_df.sort_values(by='mean', ascending=False)
    
    def get_cellfate(self, transition_df, allowed_fates, thresh=0.002, annot='cell_type', null_ct='null', self_thresh=0):
        source_ct = transition_df.columns.name
        assert source_ct in allowed_fates

        transitions = []

        base_celltypes = self.adata.obs[annot]

        for ix in tqdm(range(transition_df.shape[0])):

            fate_df = transition_df.iloc[ix].to_frame().join(
                base_celltypes).groupby(annot).mean().loc[allowed_fates]
            
            ct = fate_df.sort_values(ix, ascending=False).iloc[0].to_frame()

            self_fate = fate_df.query(f'{annot} == @source_ct').values[0][0]
            transition_fate = fate_df.query(f'{annot} == @ct.columns[0]').values[0][0]
        
            if (
                (transition_fate > self_fate)
                and (transition_fate > thresh)
                and (transition_fate > self_thresh)
            ):
                transitions.append(ct.columns[0])
            elif self_fate < thresh:
                transitions.append(null_ct)
            else:
                transitions.append(source_ct)

        
        print(f'source ct {source_ct}', Counter(transitions), np.mean(transition_fate), self_thresh)
        return transitions

    def get_transition_annot(self, corr, allowed_fates, thresh=0.0002, annot='leiden'):
        
        all_fates = []

        if thresh is None:
            thresh = np.median(corr)

        for source_ct in self.adata.obs[annot].unique():

            transition_df = self.compute_transitions(corr, source_ct=source_ct, annot=annot)

            range_df = self.assess_transitions(transition_df, self.adata.obs[annot], source_ct, annot)
            self_thresh = range_df.loc[source_ct, 'min'] # transition should exceed the minimum self transition
 
            fates = self.get_cellfate(transition_df, 
                    allowed_fates=allowed_fates, thresh=thresh, annot=annot, self_thresh=self_thresh)

            ct_df = pd.DataFrame(
                fates, 
                index=self.adata.obs[self.adata.obs[annot] == source_ct].index,
                columns=['transition'])
            all_fates.append(ct_df)
        
        all_fates = pd.concat(all_fates, axis=0)
        self.adata.obs = pd.concat([self.adata.obs, all_fates], axis=1)
    
    def make_celltype_dict(self, annot='cell_type', basis='spatial'):
        assert 'transition' in self.adata.obs
        assert annot in self.adata.obs
        
        ct_points_wt = {}
        for ct in self.adata.obs[annot].unique():
            points = np.asarray(
                self.adata.obsm[basis][self.adata.obs[annot] == ct])
            if basis == 'spatial':
                delta = 30
                points = np.vstack(
                    (points +[-delta,delta], points +[-delta,-delta], 
                    points +[delta,delta], points +[delta,-delta]))
            ct_points_wt[ct] = points

        ct_points_ko = {}
        for ct in self.adata.obs['transition'].unique():
            points = np.asarray(
                self.adata.obsm[basis][self.adata.obs['transition'] == ct])
            if basis == 'spatial':
                delta = 30
                points = np.vstack(
                    (points +[-delta,delta], points +[-delta,-delta], 
                    points +[delta,delta], points +[delta,-delta]))
            ct_points_ko[ct] = points
            
        return ct_points_wt, ct_points_ko

    def compute_transition_probabilities(self, delta_X, embedding, n_neighbors=200, remove_null=True, normalize=False):
            
        P = estimate_transition_probabilities(
            self.adata, delta_X, embedding, n_neighbors=n_neighbors, n_jobs=1)
        
        if remove_null:
            P_null = estimate_transition_probabilities(
                self.adata, delta_X * 0, embedding, n_neighbors=n_neighbors, n_jobs=1)
            P = P - P_null
            
        if normalize:
            P = (P - P.min()) / (P.max() - P.min())
            P = P / P.sum(axis=1)[:, np.newaxis]

        return P
    
    @staticmethod
    def knn_regression(x, y, x_new, y_new, value, n_knn=30):
        data = np.stack([x, y], axis=1)
        model = KNeighborsRegressor(n_neighbors=n_knn)
        model.fit(data, value)
        data_new = np.stack([x_new, y_new], axis=1)
        return model.predict(data_new)

    def compute_transition_vector_field(
        self, 
        perturbed_df,
        hue='cell_type',
        normalize=False, 
        n_neighbors=150, 
        grid_scale=1, 
        vector_scale=0.85,
        remove_null=True,
        rescale=1,
        rename=None,
        highlight_clusters=None,
        limit_clusters=False):
        
        assert 'X_umap' in self.adata.obsm
        assert 'cell_type' in self.adata.obs
        
        layout_embedding = self.adata.obsm['X_umap']
        
        if rename is None:
            rename = {}
        
        perturbed_df = perturbed_df.loc[self.adata.obs_names]
        
        if limit_clusters and highlight_clusters is not None:
            mask = ~self.adata.obs[hue].isin(highlight_clusters)
            perturbed_df.loc[mask] = self.adata.to_df(layer='imputed_count').loc[mask]

        delta_X = perturbed_df.values - self.adata.layers['imputed_count']
        delta_X = delta_X.round(3)
            
            
        P = self.compute_transition_probabilities(
            delta_X * rescale, 
            layout_embedding, 
            n_neighbors=n_neighbors, 
            remove_null=remove_null
        )

        V_simulated = project_probabilities(P, layout_embedding, normalize=normalize)
        
        grid_scale = 10 * grid_scale / np.mean(abs(np.diff(layout_embedding)))
        grid_x, grid_y = get_grid_layout(layout_embedding, grid_scale=grid_scale)
        grid_points = np.array(np.meshgrid(grid_x, grid_y)).T.reshape(-1, 2)
        size_x, size_y = len(grid_x), len(grid_y)
        vector_field = np.zeros((size_x, size_y, 2))
        x_thresh = (grid_x[1] - grid_x[0]) / 2
        y_thresh = (grid_y[1] - grid_y[0]) / 2
        
        get_neighborhood = lambda grid_point, layout_embedding: np.where(
            (np.abs(layout_embedding[:, 0] - grid_point[0]) <= x_thresh) &  
            (np.abs(layout_embedding[:, 1] - grid_point[1]) <= y_thresh)   
        )[0]

        for idx, grid_point in enumerate(grid_points):

            indices = get_neighborhood(grid_point, layout_embedding)
            if len(indices) <= 0:
                continue
            nbr_vector = np.mean(V_simulated[indices], axis=0)
            nbr_vector *= len(indices)       # upweight vectors with lots of cells
                
            grid_idx_x, grid_idx_y = np.unravel_index(idx, (size_x, size_y))
            vector_field[grid_idx_x, grid_idx_y] = nbr_vector



        vector_field = vector_field.reshape(-1, 2)
        
        vector_scale = vector_scale / np.max(vector_field)
        vector_field *= vector_scale
        
        return grid_point, vector_field
    
    
    def get_vector_field(
        self, 
        perturbed_df,
        limit_clusters=True, 
        highlight_clusters=None, 
        n_neighbors=200, 
        remove_null=True, 
        normalize=True, 
        grid_scale=1,
        annot='cell_type',
        threshold=0,
        vector_scale=0.4):
        
        layout_embedding = self.adata.obsm['X_umap']
        perturbed_df = perturbed_df.loc[self.adata.obs_names]
        
        if limit_clusters and highlight_clusters is not None:
            mask = ~self.adata.obs[annot].isin(highlight_clusters)
            perturbed_df.loc[mask] = self.adata.to_df(layer='imputed_count').loc[mask]

        delta_X = perturbed_df.values - self.adata.layers['imputed_count']
        delta_X = delta_X.round(3)
            
            
        P = self.compute_transition_probabilities(
            delta_X, 
            layout_embedding, 
            n_neighbors=n_neighbors, 
            remove_null=remove_null
        )

        V_simulated = project_probabilities(P, layout_embedding, normalize=normalize)
        
        grid_scale = 10 * grid_scale / np.mean(abs(np.diff(layout_embedding)))
        grid_x, grid_y = get_grid_layout(layout_embedding, grid_scale=grid_scale)
        grid_points = np.array(np.meshgrid(grid_x, grid_y)).T.reshape(-1, 2)
        size_x, size_y = len(grid_x), len(grid_y)
        vector_field = np.zeros((size_x, size_y, 2))
        x_thresh = (grid_x[1] - grid_x[0]) / 2
        y_thresh = (grid_y[1] - grid_y[0]) / 2
        
        get_neighborhood = lambda grid_point, layout_embedding: np.where(
            (np.abs(layout_embedding[:, 0] - grid_point[0]) <= x_thresh) &  
            (np.abs(layout_embedding[:, 1] - grid_point[1]) <= y_thresh)   
        )[0]

        for idx, grid_point in enumerate(grid_points):

            indices = get_neighborhood(grid_point, layout_embedding)
            if len(indices) <= 0:
                continue
            nbr_vector = np.mean(V_simulated[indices], axis=0)
            nbr_vector *= len(indices)       # upweight vectors with lots of cells
                
            grid_idx_x, grid_idx_y = np.unravel_index(idx, (size_x, size_y))
            vector_field[grid_idx_x, grid_idx_y] = nbr_vector

        vector_field = vector_field.reshape(-1, 2)
        
        vector_scale = vector_scale / np.max(vector_field)
        vector_field *= vector_scale

        if threshold and threshold > 0:
            mags = np.linalg.norm(vector_field, axis=1)
            vector_field[mags < threshold] = 0
        
        return grid_points, vector_field
 
    def plot_umap_quiver(
            self, 
            perturb_target='', 
            hue='cell_type',
            normalize=False, 
            n_neighbors=150, 
            grid_scale=1, 
            vector_scale=0.85,
            scatter_size=25,
            legend_on_loc=False,
            legend_fontsize=7,
            figsize=(5, 5),
            dpi=300,
            alpha=0.8,
            linewidth=0.1,
            betadata_path='.',
            alt_colors=None,
            remove_null=True,
            perturbed_df = None,
            rescale=1,
            scale=5,
            grains=20,
            rename=None,
            ax=None,
            curve=True,
            arrowstyle='fancy',
            arrowsize=0.5,
            arrow_linewidth=0.55,
            quiver_headwidth=3,
            quiver_headlength=3,
            quiver_headaxislength=3,
            quiver_width=0.002,
            grey_out=True,
            highlight_clusters=None,
            limit_clusters=False,
            value=None,
            arrow_alpha_non_highlighted=0.3,
            threshold=0,
            dynamic_alpha=True,
            lightgrey="#9c8d7c"
        ):
        assert 'X_umap' in self.adata.obsm
        assert 'cell_type' in self.adata.obs
        
        layout_embedding = self.adata.obsm['X_umap']
        
        if rename is None:
            rename = {}
        
        if perturbed_df is None:
            
            pattern = f'{betadata_path}/{perturb_target}_*n_*x.parquet'
            matching_files = glob.glob(pattern)
            if matching_files:
                perturbed_df = pd.read_parquet(matching_files[0])
            else:
                raise FileNotFoundError(f"No perturbed data file found for {perturb_target}")
        
        
        perturbed_df = perturbed_df.loc[self.adata.obs_names]
        
        if limit_clusters and highlight_clusters is not None:
            mask = ~self.adata.obs[hue].isin(highlight_clusters)
            perturbed_df.loc[mask] = self.adata.to_df(layer='imputed_count').loc[mask]

        delta_X = perturbed_df.values - self.adata.layers['imputed_count']
        delta_X = delta_X.round(3)
            
            
        P = self.compute_transition_probabilities(
            delta_X * rescale, 
            layout_embedding, 
            n_neighbors=n_neighbors, 
            remove_null=remove_null
        )

        V_simulated = project_probabilities(P, layout_embedding, normalize=normalize)
        
        grid_scale = 10 * grid_scale / np.mean(abs(np.diff(layout_embedding)))
        grid_x, grid_y = get_grid_layout(layout_embedding, grid_scale=grid_scale)
        grid_points = np.array(np.meshgrid(grid_x, grid_y)).T.reshape(-1, 2)
        size_x, size_y = len(grid_x), len(grid_y)
        vector_field = np.zeros((size_x, size_y, 2))
        x_thresh = (grid_x[1] - grid_x[0]) / 2
        y_thresh = (grid_y[1] - grid_y[0]) / 2
        
        get_neighborhood = lambda grid_point, layout_embedding: np.where(
            (np.abs(layout_embedding[:, 0] - grid_point[0]) <= x_thresh) &  
            (np.abs(layout_embedding[:, 1] - grid_point[1]) <= y_thresh)   
        )[0]

        for idx, grid_point in enumerate(grid_points):

            indices = get_neighborhood(grid_point, layout_embedding)
            if len(indices) <= 0:
                continue
            nbr_vector = np.mean(V_simulated[indices], axis=0)
            nbr_vector *= len(indices)       # upweight vectors with lots of cells
                
            grid_idx_x, grid_idx_y = np.unravel_index(idx, (size_x, size_y))
            vector_field[grid_idx_x, grid_idx_y] = nbr_vector



        vector_field = vector_field.reshape(-1, 2)
        
        vector_scale = vector_scale / np.max(vector_field)
        vector_field *= vector_scale

        if threshold and threshold > 0:
            mags = np.linalg.norm(vector_field, axis=1)
            vector_field[mags < threshold] = 0
        
        if ax is None:
            fig, ax = plt.subplots(figsize=figsize, dpi=dpi)
        else:
            fig = ax.get_figure()
            ax = ax
        
        color_dict = self.color_dict.copy()
        
        # Create a modified color dictionary if highlighting specific clusters
        plot_df = pd.DataFrame(
            self.adata.obsm['X_umap'], 
            columns=['x', 'y'], 
            index=self.adata.obs_names).join(self.adata.obs)
        
        if highlight_clusters is not None:
            highlight_color_dict = {ct: lightgrey for ct in color_dict}
            for ct in highlight_clusters:
                if ct in color_dict:
                    highlight_color_dict[ct] = color_dict[ct]
                    
            if not grey_out:
                for ct in color_dict:
                    highlight_color_dict[ct] = color_dict[ct]
                    
            plot_df['highlighted'] = plot_df[hue].isin(highlight_clusters)
            
            vector_magnitudes = np.linalg.norm(V_simulated, axis=1)
            vector_magnitudes = 0.1 + 0.9 * (vector_magnitudes - vector_magnitudes.min()) / (vector_magnitudes.max() - vector_magnitudes.min())
            vector_magnitudes = np.clip(vector_magnitudes, 0.01, 1) * alpha

            sns.scatterplot(
                data=plot_df,
                x='x', y='y',
                hue=hue, 
                s=scatter_size,
                ax=ax,
                alpha=vector_magnitudes if dynamic_alpha else alpha,
                edgecolor='black',
                linewidth=linewidth,
                palette=highlight_color_dict,
                legend=not legend_on_loc
            )
        else:
            plot_df['highlighted'] = True
            
            sns.scatterplot(
                data=plot_df,
                x='x', y='y',
                hue=hue, 
                s=scatter_size,
                ax=ax,
                alpha=alpha,
                edgecolor='black',
                linewidth=linewidth,
                palette=color_dict,
                legend=not legend_on_loc
            )
            
        if highlight_clusters is not None:
            highlighted_regions = np.zeros(len(grid_points), dtype=bool)
            
            for i, grid_point in enumerate(grid_points):
                indices = get_neighborhood(grid_point, layout_embedding)
                if len(indices) > 0:
                    cell_indices = self.adata.obs_names[indices]
                    if plot_df.loc[cell_indices, 'highlighted'].any():
                        highlighted_regions[i] = True
            
            highlighted_points = grid_points[highlighted_regions]
            highlighted_vectors = vector_field[highlighted_regions]
            
            
            non_highlighted_points = grid_points[~highlighted_regions]
            non_highlighted_vectors = vector_field[~highlighted_regions]
            
            
            if len(highlighted_points) > 0:
                if curve:
                        sort_idx = np.argsort(grid_points[:, 0])
                        x_ = grid_points[sort_idx, 0]
                        y_ = grid_points[sort_idx, 1]
                        u_ = vector_field[sort_idx, 0] 
                        v_ = vector_field[sort_idx, 1]
                        xi = np.linspace(x_.min(), x_.max(), 100)
                        yi = np.linspace(y_.min(), y_.max(), 100)
                        xi, yi = np.meshgrid(xi, yi)
                        ui = griddata((x_, y_), u_, (xi, yi), method='linear')
                        vi = griddata((x_, y_), v_, (xi, yi), method='linear')
                        
                        alpha_values = np.full_like(ui, 0.15)
                        for i in range(len(xi)):
                            for j in range(len(yi)):
                                point = np.array([xi[i,j], yi[i,j]])
                                indices = get_neighborhood(point, layout_embedding)
                                if len(indices) > 0:
                                    cell_indices = self.adata.obs_names[indices]
                                    if plot_df.loc[cell_indices, 'highlighted'].any():
                                        alpha_values[i,j] = 1.0

                        velovect(ax, 
                            xi[0,:], yi[:,0], ui, vi,
                            arrowstyle=arrowstyle,
                            color='black',
                            arrowsize=arrowsize,
                            linewidth=arrow_linewidth,
                            # alpha=alpha_values,
                            scale=scale, grains=grains)
                else:
                    ax.quiver(
                        highlighted_points[:, 0], highlighted_points[:, 1],   
                        highlighted_vectors[:, 0], highlighted_vectors[:, 1], 
                        angles='xy', scale_units='xy', scale=1, 
                        headwidth=quiver_headwidth, headlength=quiver_headlength, headaxislength=quiver_headaxislength,
                        width=quiver_width, alpha=vector_magnitudes
                    )
                    
                
            
            if len(non_highlighted_points) > 0:
                
                if curve:
                    pass
                    
                else:
                    ax.quiver(
                        non_highlighted_points[:, 0], non_highlighted_points[:, 1],   
                        non_highlighted_vectors[:, 0], non_highlighted_vectors[:, 1], 
                        angles='xy', scale_units='xy', scale=1, 
                        headwidth=quiver_headwidth, headlength=quiver_headlength, headaxislength=quiver_headaxislength,
                        width=quiver_width, alpha=arrow_alpha_non_highlighted
                    )
                

        else:
            plot_quiver(grid_points, vector_field, background=None, ax=ax)
         
        ax.set_frame_on(False)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_xlabel('')
        ax.set_ylabel('')
        ax.set_title('')
        
        alt_colors = self.color_dict
        all_cts = self.adata.obs[hue]

        if legend_on_loc:
            texts = []
            for cluster in sorted(all_cts.unique()):
                cluster_cells = all_cts == cluster
                x = np.mean(self.adata.obsm['X_umap'][cluster_cells, 0])
                y = np.mean(self.adata.obsm['X_umap'][cluster_cells, 1])
                
                if highlight_clusters is not None and cluster not in highlight_clusters:
                    color = lightgrey if grey_out else alt_colors[cluster]
                else:
                    color = alt_colors[cluster]
                
                text = ax.text(x, y, rename.get(cluster, cluster), 
                        fontsize=legend_fontsize, 
                        ha='center', 
                        va='center',
                        color='black',
                        bbox=dict(
                            facecolor=color,
                            alpha=1,
                            edgecolor='black',
                            boxstyle='round',
                            linewidth=0.25
                        ))
                texts.append(text)
            
            if texts:
                adjust_text(texts, 
                           arrowprops=dict(arrowstyle='->', color='gray', lw=0.5, alpha=0.7),
                           ax=ax)
                
        
        if not legend_on_loc:
            handles = [plt.scatter([], [], c=alt_colors[label], label=label) for label in sorted(all_cts.unique())]
            legend = ax.legend(handles=handles, bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0., fontsize=legend_fontsize)
        
        return grid_points, vector_field, P
    
    def plot_umap_pseudotime(
            self, 
            perturb_target, 
            iroot,
            hue='cell_type',
            normalize=False, 
            n_neighbors=150, 
            grid_scale=1, 
            vector_scale=0.85,
            scatter_size=25,
            legend_on_loc=False,
            legend_fontsize=7,
            figsize=(5, 5),
            dpi=300,
            alpha=0.8,
            linewidth=0.1,
            betadata_path='.',
            alt_colors=None,
            remove_null=True,
            perturbed_df = None,
            rescale=1,
            scale=5,
            grains=20,
            rename=None,
            ax=None,
            curve=True,
            grey_out=True,
            highlight_clusters=None,
            limit_clusters=False,
            value=None,
            arrow_alpha_non_highlighted=0.3,
        ):
        import scanpy as sc
        
        assert 'X_umap' in self.adata.obsm
        assert 'cell_type' in self.adata.obs
        
        layout_embedding = self.adata.obsm['X_umap']
        
        if rename is None:
            rename = {}
        
        if perturbed_df is None:
            
            pattern = f'{betadata_path}/{perturb_target}_*n_*x.parquet'
            matching_files = glob.glob(pattern)
            if matching_files:
                perturbed_df = pd.read_parquet(matching_files[0])
            else:
                raise FileNotFoundError(f"No perturbed data file found for {perturb_target}")
        
        
        perturbed_df = perturbed_df.loc[self.adata.obs_names]
        
        if limit_clusters and highlight_clusters is not None:
            mask = ~self.adata.obs[hue].isin(highlight_clusters)
            perturbed_df.loc[mask] = self.adata.to_df(layer='imputed_count').loc[mask]

        delta_X = perturbed_df.values - self.adata.layers['imputed_count']
        delta_X = delta_X.round(3)
            
            
        P = self.compute_transition_probabilities(
            delta_X * rescale, 
            layout_embedding, 
            n_neighbors=n_neighbors, 
            remove_null=remove_null
        )
        
        V_simulated = project_probabilities(P, layout_embedding, normalize=normalize)
        
        grid_scale = 10 * grid_scale / np.mean(abs(np.diff(layout_embedding)))
        grid_x, grid_y = get_grid_layout(layout_embedding, grid_scale=grid_scale)
        grid_points = np.array(np.meshgrid(grid_x, grid_y)).T.reshape(-1, 2)
        size_x, size_y = len(grid_x), len(grid_y)
        vector_field = np.zeros((size_x, size_y, 2))
        x_thresh = (grid_x[1] - grid_x[0]) / 2
        y_thresh = (grid_y[1] - grid_y[0]) / 2
        
        get_neighborhood = lambda grid_point, layout_embedding: np.where(
            (np.abs(layout_embedding[:, 0] - grid_point[0]) <= x_thresh) &  
            (np.abs(layout_embedding[:, 1] - grid_point[1]) <= y_thresh)   
        )[0]

        for idx, grid_point in enumerate(grid_points):

            indices = get_neighborhood(grid_point, layout_embedding)
            if len(indices) <= 0:
                continue
            nbr_vector = np.mean(V_simulated[indices], axis=0)
            nbr_vector *= len(indices)       # upweight vectors with lots of cells
                
            grid_idx_x, grid_idx_y = np.unravel_index(idx, (size_x, size_y))
            vector_field[grid_idx_x, grid_idx_y] = nbr_vector

        vector_field = vector_field.reshape(-1, 2)
        
        vector_scale = vector_scale / np.max(vector_field)
        vector_field *= vector_scale
        
        # Pseudotime analysis
        fig, ax = plt.subplots(figsize=figsize, dpi=dpi)

        self.adata.uns['iroot'] = np.flatnonzero(
            self.adata.obs['cell_type_2'] == iroot)[0]
        sc.tl.diffmap(self.adata)
        sc.tl.dpt(self.adata, n_dcs=10, n_branchings=0)
        value = np.where(np.isinf(self.adata.obs.dpt_pseudotime.values), 1, self.adata.obs.dpt_pseudotime.values)
        
        
        embedding = layout_embedding
        x, y = embedding[:, 0], embedding[:, 1]
        x_new, y_new = grid_points[:, 0], grid_points[:, 1]

        pseudotime_on_grid = self.knn_regression(
            x, y, x_new, y_new, value, n_knn=30)
        gradient = get_gradient(value_on_grid=pseudotime_on_grid)
        gradient = normalize_gradient(gradient, method="sqrt")
        l2_norm = np.linalg.norm(gradient, ord=2, axis=1)
        scale_factor = 1 / l2_norm.mean()
        ref_flow = gradient * 1

        zero_mask = (vector_field[:,0] == 0) & (vector_field[:,1] == 0)
        ref_flow[zero_mask] = 0
        
        cmap = LinearSegmentedColormap.from_list('custom', ['red', 'white', 'green'])
        
        
        inner_product = np.sum(vector_field * ref_flow, axis=1)
        vmax = np.abs(inner_product).max()
        scatter = ax.scatter(
            grid_points[:,0], grid_points[:,1], 
            c=inner_product, cmap=cmap, 
            s=20,
            vmin=-vmax, 
            vmax=vmax, 
            alpha=1, marker='s')

        ax.quiver(grid_points[:,0], grid_points[:,1],
                vector_field[:,0], vector_field[:,1],
                angles='xy',
                scale_units='xy', scale=1,
                alpha=0.8, width=0.003, 
                color='black')

        ax.quiver(grid_points[:,0], grid_points[:,1],
                ref_flow[:, 0], ref_flow[:, 1], 
                angles='xy', scale_units='xy', scale=1,
                alpha=0.8, color='blue', width=0.003)
        
        ax.set_frame_on(False)
        ax.set_title('')
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_xlabel('')
        ax.set_ylabel('')
        
        # Legend handling
        alt_colors = self.color_dict
        all_cts = self.adata.obs[hue]

        if legend_on_loc:
            texts = []
            for cluster in all_cts.unique():
                cluster_cells = all_cts == cluster
                x = np.mean(self.adata.obsm['X_umap'][cluster_cells, 0])
                y = np.mean(self.adata.obsm['X_umap'][cluster_cells, 1])
                
                # Use the appropriate color
                if highlight_clusters is not None and cluster not in highlight_clusters:
                    color = 'lightgrey'
                else:
                    color = alt_colors[cluster]
                
                text = ax.text(x, y, rename.get(cluster, cluster), 
                        fontsize=legend_fontsize, 
                        ha='center', 
                        va='center',
                        color='black',
                        bbox=dict(
                            facecolor=color,
                            alpha=1,
                            edgecolor=None,
                            boxstyle='round',
                            linewidth=0.15
                        ))
                texts.append(text)
            
            # Adjust text positions to prevent overlaps
            if texts:
                adjust_text(texts, 
                           arrowprops=dict(arrowstyle='->', color='gray', lw=0.5, alpha=0.7),
                           ax=ax)
                
        
        if not legend_on_loc:
            if highlight_clusters is not None:
                handles = []
                for label in all_cts.unique():
                    if label in highlight_clusters:
                        color = alt_colors[label]
                    else:
                        color = 'lightgrey'
                    handles.append(plt.scatter([], [], c=color, label=label))
            else:
                handles = [plt.scatter([], [], c=alt_colors[label], label=label) 
                           for label in all_cts.unique()]
                
            legend = ax.legend(handles=handles, bbox_to_anchor=(1.05, 1), loc='upper left', 
                               borderaxespad=0., fontsize=legend_fontsize)
        
        return grid_points, vector_field, P
    
    def get_grids(self, P, projection_params):
        
        self.adata.obsp['_shift'] = P.copy()
        ck = cr.kernels.ConnectivityKernel(self.adata, conn_key='_shift')
        ck.compute_transition_matrix(density_normalize=True)
        
        return ck.plot_projection(**projection_params)
    
    def vector_field_df(self, X_grid, V_grid):
        spatial_coords = self.adata.obsm['spatial']
        grid_tree = KDTree(X_grid)
        dists, idxs = grid_tree.query(spatial_coords, k=4)

        # Convert distances to weights (inverse distance weighting)
        weights = 1 / (dists + 1e-10)  # Add small constant to avoid division by zero
        weights = weights / weights.sum(axis=1, keepdims=True)  # Normalize weights

        # Calculate angles of grid vectors
        grid_angles = np.degrees(np.arctan2(V_grid[:, 1], V_grid[:, 0]))

        # Calculate weighted average angle for each cell
        cell_angles = np.sum(grid_angles[idxs] * weights, axis=1)

        # Create dataframe
        vector_field_df = pd.DataFrame({
            'x': spatial_coords[:, 0],
            'y': spatial_coords[:, 1],
            'angle': cell_angles
        })

        self.adata.obs.index.name = None
        
        vector_field_df.index = self.adata.obs.index
        self.adata.obs.index.name = None
        
        return vector_field_df
    
    def plot_umap(
            self, 
            hue='cell_type',
            basis='X_umap',
            figsize=(5, 5),
            dpi=180,
            alpha=0.9,
            scatter_size=5,
            linewidth=0.1,
            legend_on_loc=True,
            legend_fontsize=8,
            highlight_clusters=None,
            alt_colors=None,
            rename=None,
            adata=None,
            ax=None,
            color_dict=None,
            layer='imputed_count',
            cmap='viridis',
            colorbar=True,
            quiver_on_genes=False,
            n_neighbors=300,
            arrow_linewidth=0.55,
            scale=1,
            curve=False,
            grains=20,
            grid_scale=1,
            arrowstyle='fancy',
            vector_scale=0.85,
            vector_color='black',
            arrowsize=0.5,
            threshold=0.0,
            vector_cmap=None,
            label='',
            headwidth=3, headlength=3, headaxislength=3,
            width=0.005,
        ):
        if adata is None:
            adata = self.adata
            
        assert basis in adata.obsm
        
        if rename is None:
            rename = {}
            
        if ax is None:
            f, ax = plt.subplots(figsize=figsize, dpi=dpi)
        
        if color_dict is None:
            color_dict = self.color_dict.copy()
        
        plot_df = pd.DataFrame(
            adata.obsm[basis], 
            columns=['x', 'y'], 
            index=adata.obs_names).join(adata.obs)
        
        # If hue is a gene or list of genes, plot continuous expression
        is_gene_name = isinstance(hue, str) and (hue in adata.var_names)
        is_gene_list = isinstance(hue, (list, tuple, np.ndarray)) and all([g in adata.var_names for g in hue])
        if is_gene_name or is_gene_list:
            genes = [hue] if is_gene_name else list(hue)
            layer_used = layer
            expr_df = adata.to_df(layer=layer_used)
            values = expr_df[genes].mean(1) if len(genes) > 1 else expr_df[genes[0]]
            values = (values - values.min()) / (values.max() - values.min())
            
            plot_df['__gene_expr__'] = values.loc[plot_df.index]
            

            sc = ax.scatter(
                plot_df['x'], plot_df['y'],
                c=plot_df['__gene_expr__'],
                cmap=cmap,
                s=scatter_size,
                alpha=alpha,
                edgecolor='black',
                linewidth=linewidth,
            )

            if colorbar:
                # plt.colorbar(sc, ax=ax, fraction=0.046, pad=0.04)
                plt.colorbar(sc, label=label, 
                    orientation='vertical',
                    fraction=0.046, pad=0.04,
                    shrink=0.5)
                
            if quiver_on_genes:
                layout_embedding = adata.obsm[basis]
                n_cells, n_genes = adata.shape
                delta_X = np.zeros_like(adata.layers['imputed_count'], dtype=np.float64)
                for g in genes:
                    if g in adata.var_names:
                        gi = int(np.where(adata.var_names == g)[0][0])
                        delta_X[:, gi] = expr_df[g].values
                delta_X = np.round(delta_X, 5)

                P = self.compute_transition_probabilities(
                    delta_X,
                    layout_embedding,
                    n_neighbors=n_neighbors,
                    remove_null=True,
                    normalize=False,
                )

                V_simulated = project_probabilities(P, layout_embedding, normalize=False)

                adaptive_scale = 10 * grid_scale / np.mean(abs(np.diff(layout_embedding)))
                grid_x, grid_y = get_grid_layout(layout_embedding, grid_scale=adaptive_scale)
                grid_points = np.array(np.meshgrid(grid_x, grid_y)).T.reshape(-1, 2)
                size_x, size_y = len(grid_x), len(grid_y)
                vector_field = np.zeros((size_x, size_y, 2))
                x_thresh = (grid_x[1] - grid_x[0]) / 2
                y_thresh = (grid_y[1] - grid_y[0]) / 2

                get_neighborhood = lambda gp, emb: np.where(
                    (np.abs(emb[:, 0] - gp[0]) <= x_thresh) &
                    (np.abs(emb[:, 1] - gp[1]) <= y_thresh)
                )[0]

                for idx, gp in enumerate(grid_points):
                    indices = get_neighborhood(gp, layout_embedding)
                    if len(indices) <= 0:
                        continue
                    nbr_vector = np.mean(V_simulated[indices], axis=0)
                    nbr_vector *= len(indices)
                    gx, gy = np.unravel_index(idx, (size_x, size_y))
                    vector_field[gx, gy] = nbr_vector

                vector_field = vector_field.reshape(-1, 2)
                vmax = np.max(np.abs(vector_field)) if vector_field.size else 0
                if vmax > 0:
                    vector_field = (vector_scale / vmax) * vector_field
                    magnitudes = np.linalg.norm(vector_field, axis=1)
                    zero_mask = magnitudes < threshold
                    vector_field[zero_mask] = 0
                    if curve:
                        sort_idx = np.argsort(grid_points[:, 0])
                        x_ = grid_points[sort_idx, 0]
                        y_ = grid_points[sort_idx, 1]
                        u_ = vector_field[sort_idx, 0]
                        v_ = vector_field[sort_idx, 1]
                        xi = np.linspace(x_.min(), x_.max(), 100)
                        yi = np.linspace(y_.min(), y_.max(), 100)
                        xi, yi = np.meshgrid(xi, yi)
                        ui = griddata((x_, y_), u_, (xi, yi), method='linear')
                        vi = griddata((x_, y_), v_, (xi, yi), method='linear')

                        mag_grid = np.sqrt(np.square(ui) + np.square(vi))
                        ui[mag_grid < threshold] = 0
                        vi[mag_grid < threshold] = 0

                        velovect(
                            ax,
                            xi[0, :], yi[:, 0], ui, vi,
                            arrowstyle=arrowstyle,
                            color=vector_color,
                            arrowsize=arrowsize,
                            linewidth=arrow_linewidth,
                            scale=scale, grains=grains,
                        )
                    else:
                        colors = None
                        if vector_cmap is not None:
                            try:
                                cmap_obj = plt.get_cmap(vector_cmap)
                            except Exception:
                                cmap_obj = vector_cmap
                            norm = (magnitudes - magnitudes.min()) / (magnitudes.max() - magnitudes.min() + 1e-12)
                            colors = cmap_obj(norm)
                        ax.quiver(
                            grid_points[:, 0], grid_points[:, 1],
                            vector_field[:, 0], vector_field[:, 1],
                            angles='xy', scale_units='xy', scale=scale,
                            headwidth=headwidth, headlength=headlength, headaxislength=headaxislength,
                            width=width, alpha=1,
                            color=(colors if colors is not None else vector_color)
                        )

            ax.set_frame_on(False)
            ax.set_xticks([])
            ax.set_yticks([])
            ax.set_xlabel('')
            ax.set_ylabel('')
            ax.set_title('')
            return ax, grid_points, vector_field, P
        
        if highlight_clusters is not None:
            highlight_color_dict = {ct: 'lightgrey' for ct in color_dict}
            for ct in highlight_clusters:
                if ct in color_dict:
                    highlight_color_dict[ct] = color_dict[ct]
                    
            plot_df['highlighted'] = plot_df[hue].isin(highlight_clusters)
            
            sns.scatterplot(
                data=plot_df,
                x='x', y='y',
                hue=hue, 
                s=scatter_size,
                ax=ax,
                alpha=alpha,
                edgecolor='black',
                linewidth=linewidth,
                palette=highlight_color_dict,
                legend=not legend_on_loc
            )
        else:
            plot_df['highlighted'] = True
            
            sns.scatterplot(
                data=plot_df,
                x='x', y='y',
                hue=hue, 
                s=scatter_size,
                ax=ax,
                alpha=alpha,
                edgecolor='black',
                linewidth=linewidth,
                palette=color_dict,
                legend=not legend_on_loc
            )

        ax.set_frame_on(False)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_xlabel('')
        ax.set_ylabel('')
        ax.set_title('')
        
        if alt_colors is None:
            alt_colors = self.color_dict
            
        all_cts = adata.obs[hue]

        if legend_on_loc:
            texts = []
            for cluster in all_cts.unique():
                cluster_cells = all_cts == cluster
                x = np.mean(adata.obsm[basis][cluster_cells, 0])
                y = np.mean(adata.obsm[basis][cluster_cells, 1])
                
                # Use the appropriate color
                if highlight_clusters is not None and cluster not in highlight_clusters:
                    color = 'lightgrey'
                else:
                    color = color_dict[cluster]
                
                text = ax.text(x, y, rename.get(cluster, cluster), 
                        fontsize=legend_fontsize, 
                        ha='center', 
                        va='center',
                        color='black',
                        bbox=dict(
                            facecolor=color,
                            alpha=1,
                            edgecolor=None,
                            boxstyle='round',
                            linewidth=0.15
                        ))
                texts.append(text)
            
            if texts:
                adjust_text(texts, 
                           arrowprops=dict(arrowstyle='->', color='gray', lw=0.5, alpha=0.7),
                           ax=ax)
        
        if not legend_on_loc:
            if highlight_clusters is not None:
                handles = []
                for label in all_cts.unique():
                    if label in highlight_clusters:
                        color = alt_colors[label]
                    else:
                        color = 'lightgrey'
                    handles.append(plt.scatter([], [], c=color, label=label))
            else:
                handles = [plt.scatter([], [], c=alt_colors[label], label=label) 
                           for label in all_cts.unique()]
                
            legend = ax.legend(handles=handles, bbox_to_anchor=(1.05, 1), loc='upper left', 
                               borderaxespad=0., fontsize=legend_fontsize)
        
        return ax
    
    
