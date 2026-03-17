import warnings
warnings.filterwarnings("ignore")

from functools import cache
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
import matplotlib as mpl
from itertools import cycle
from scipy.stats import pearsonr
from sklearn.preprocessing import StandardScaler, RobustScaler, MinMaxScaler
from sklearn.neighbors import KNeighborsRegressor
import glob
import enlighten
import numpy as np
import random
import os
from scipy.spatial import cKDTree
from .models.parallel_estimators import create_spatial_features
from sklearn.neighbors import NearestNeighbors
from .plotting.cartography import Cartography, xy_from_adata
from .gene_factory import GeneFactory
from .beta import BetaFrame

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



class VirtualTissue:
    
    def __init__(
        self, 
        adata, 
        betadatas_path=None,  
        ko_path=None, 
        ovx_path=None, 
        color_dict=None,
        spf_radius=200,
        annot='cell_type',
        n_props=4):
        
        self.adata = adata
        self.betadatas_path = betadatas_path
        self.ko_path = ko_path
        self.ovx_path = ovx_path
        self.annot = annot
        

        self.n_props = n_props
        
        if ovx_path is None:
            self.ovx_path = ko_path
        
        if color_dict is None:
            
            self.color_dict = {
                c: self.random_color() for c in self.adata.obs[self.annot].unique()
            }
        else:
            self.color_dict = color_dict
        
        self.xy = xy_from_adata(self.adata) 
        self.spf_radius = spf_radius
        
        self.spf = create_spatial_features(
            x=adata.obsm['spatial'][:, 0], 
            y=adata.obsm['spatial'][:, 1], 
            celltypes=adata.obs[self.annot], 
            obs_index=adata.obs_names,
            radius = self.spf_radius
        )
        
    def random_color(self):
        return "#{:06x}".format(random.randint(0, 0xFFFFFF))
            
    def load_betadata(self, gene):
        return BetaFrame.from_path(
            f'{self.betadatas_path}/{gene}_betadata.parquet', 
            obs_names=self.adata.obs_names
        )
        
    def plot_gene_vs_proximity(
        self, perturb_target, perturbed_df, gene, color_gene, 
        cell_filter, cell_groups, cmap='rainbow_r',
        proximity_threshold=150, gene_threshold=0.005, ax=None, mode='ko'):
        
        datadf = self.spf[
            [i+'_within' for i in cell_groups
                ]].sum(1).to_frame().join(self.adata.obs[self.annot]).query(
                f'{self.annot}.isin(["{cell_filter}"])').join(self.xy).join(
            ((perturbed_df-self.adata.to_df(layer='imputed_count'))/self.adata.to_df(layer='imputed_count'))*100
        )
        datadf = datadf[datadf[0] < proximity_threshold]
        
        if ax is None:
            ax = plt.gca()
        
        try:
            corr = pearsonr(datadf[datadf[gene]>gene_threshold][0], datadf[datadf[gene]>gene_threshold][gene]).statistic
            ax.set_title(f"{perturb_target} {mode.upper()} in\n{cell_filter}\nCorrelation: {corr:.4f}")
        except:
            corr = 0
            ax.set_title(f"{perturb_target} {mode.upper()} in\n{cell_filter}")
            
        scatter = ax.scatter(
            datadf[0], 
            datadf[gene], 
            c=datadf[color_gene],
            cmap=cmap,
        )
        plt.colorbar(scatter, label=f'{color_gene} % change', shrink=0.75, ax=ax, format='%.2f')
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.set_ylabel(f'{gene} % change')
        ax.set_xlabel(f'Number of {" & ".join(cell_groups)} cells within {self.spf_radius}um')
        
        return ax, datadf

    def _knn_regression(self, x, y, x_new, y_new, value, n_knn=30):
        data = np.stack([x, y], axis=1)
        model = KNeighborsRegressor(n_neighbors=n_knn)
        model.fit(data, value)
        data_new = np.stack([x_new, y_new], axis=1)

        return model.predict(data_new)

    def signature2gradient(self, grid_points, vector_field, n_knn=30, genes=None, precomputed=None):
        embedding = self.chart.adata.obsm['X_umap']
        x, y = embedding[:, 0], embedding[:, 1]
        x_new, y_new = grid_points[:, 0], grid_points[:, 1]

        if precomputed is None:
            value = self.chart.adata.to_df(
                layer='imputed_count')[genes].sum(1).values
        else:
            value = self.chart.adata.obs[precomputed].values

        signature_on_grid = self._knn_regression(
            x, y, x_new, y_new, value, n_knn=n_knn)
        gradient = get_gradient(value_on_grid=signature_on_grid)
        gradient = normalize_gradient(gradient, method="sqrt")
        l2_norm = np.linalg.norm(gradient, ord=2, axis=1)
        scale_factor = 1 / l2_norm.mean()
        ref_flow = gradient * 2

        zero_mask = (vector_field[:,0] == 0) & (vector_field[:,1] == 0)
        ref_flow[zero_mask] = 0

        return ref_flow
    
    def init_gene_factory(self):

        self.gf = GeneFactory.from_json(
            self.adata, 
            self.betadatas_path + '/run_params.json', 
            override_params={
                'save_dir': self.betadatas_path
            }
        )
        
    def init_cartography(self, adata=None, restrict_to=None):
        if adata is None:
            adata = self.adata.copy()
            
        if restrict_to is not None:
            atmp = adata[adata.obs[self.annot].isin(restrict_to)]
        else:
            atmp = adata
        self.chart = Cartography(atmp, self.color_dict)
        
    def plot_arrows_pseudotime(self, perturb_target, perturbed_df=None, mode='max', **params):
        if perturbed_df is None:
            perturbed_df = pd.read_parquet(
                f'{self.ovx_path}/{perturb_target}_4n_{mode}x.parquet')
        
        params.setdefault('perturbed_df', perturbed_df)
        params.setdefault('perturb_target', perturb_target)
        params.setdefault('legend_on_loc', True)
        
        grid_points, vector_field, P = self.chart.plot_umap_pseudotime(**params)
        return grid_points, vector_field
    
    def plot_arrows(self, perturb_target='', threshold=0, perturbed_df=None, ax=None, mode='max', **params):
        if perturbed_df is None:
            perturbed_df = pd.read_parquet(
                f'{self.ovx_path}/{perturb_target}_4n_{mode}x.parquet')
        
        params.setdefault('perturbed_df', perturbed_df)
        params.setdefault('perturb_target', perturb_target)
        params.setdefault('legend_on_loc', True)
        params.setdefault('ax', ax)
        params.setdefault('threshold', threshold)
                
        grid_points, vector_field, P = self.chart.plot_umap_quiver(**params)
        
        return grid_points, vector_field

    @cache
    def load_knockout_gex(self, perturb_target):
        ko = pd.read_parquet(
            f"{self.ko_path}/{perturb_target}_4n_0x.parquet"
        )
        
        assert ko[perturb_target].sum() == 0
        
        return ko

    def compute_ko_impact_estimate(
        self, genes, cache_path='', force_recompute=False):
        if os.path.exists(cache_path+'ko_impact_df.csv') and not force_recompute:
            return pd.read_csv(cache_path+'ko_impact_df.csv', index_col=0)
        
        ko_data = []
        files = glob.glob(self.ko_path+f'/{self.n_props}n_0x.parquet')
        
        pbar = enlighten.manager.get_manager().counter(
            total=len(genes),
            desc='Computing KO impact',
            unit='KO',
            auto_refresh=True
        )
        for ko_file in files:
            kotarget = ko_file.split('/')[-1].split('_')[0]

            if kotarget not in genes:
                continue
            
            pbar.desc = f'{kotarget:<15}'
            pbar.refresh()
            
            data = pd.read_parquet(ko_file)
            
            data = data.loc[self.adata.obs_names] - self.adata.to_df(layer='imputed_count')
            data = data.join(self.adata.obs.cell_type).groupby('cell_type').mean().abs().mean(axis=1)

            ds = {}
            for k, v in data.sort_values(ascending=False).to_dict().items():
                ds[k] = v

            data = pd.DataFrame.from_dict(ds, orient='index')
            data.columns = [kotarget]
            ko_data.append(data)
            pbar.update(1)
        
        out = pd.concat(ko_data, axis=1)
        out.to_csv(cache_path+'ko_impact_df.csv')
        
        return out
    
    def compute_ko_impact(self, genes, 
        annot='cell_type',
        layer='imputed_count',
        baseline_only=False,
        ):
        
        # if os.path.exists(cache_path+'ko_impact_df.csv') and not force_recompute:
        #     return pd.read_csv(cache_path+'ko_impact_df.csv', index_col=0)
        
        if isinstance(genes[0], list):
            genes = [g for sublist in genes for g in sublist]
        
        ko_data = []
        if genes is None:  
            files = glob.glob(self.ko_path+'/*_0x.parquet')
        else:
            files = [f"{self.ko_path}/{gene}_4n_0x.parquet" for gene in genes]
        
        pbar = enlighten.manager.get_manager().counter(
            total=len(genes),
            desc='Computing KO impact',
            unit='KO',
            auto_refresh=True
        )
        
        for ko_file in files:
            kotarget = ko_file.split('/')[-1].split('_')[0]

            if kotarget not in genes:
                continue
            
            pbar.desc = f'{kotarget:<15}'
            pbar.refresh()
            
            if baseline_only:
                data = self.adata.to_df(layer=layer)
                data[kotarget] = 0
            else:
                data = pd.read_parquet(ko_file)
            
            data = data.loc[self.adata.obs_names] - self.adata.to_df(layer=layer)
            data = data.join(self.adata.obs[annot]).groupby(annot).mean().abs().mean(axis=1)

            ds = {}
            for k, v in data.sort_values(ascending=False).to_dict().items():
                ds[k] = v

            data = pd.DataFrame.from_dict(ds, orient='index')
            data.columns = [kotarget]
            ko_data.append(data)
            pbar.update(1)
        
        out = pd.concat(ko_data, axis=1)
        # out.to_csv(cache_path+'ko_impact_df.csv')
        
        return out
    
    def plot_comparative_radar(
        self,
        gene,
        impact_dfs,
        labels=None,
        show_for=None,
        figsize=(8, 6),
        dpi=300,
        annot='cell_type',
        rename=None,
        label_size=20,
        legend_size=12,
        color_dict=None,
        show_legend=False):

        colors = [color_dict[l] for l in labels]

        processed = []
        for df in impact_dfs:
            if gene not in df.columns:
                data = pd.Series(0, index=df.index)
            else:
                data = df[gene]
            if show_for is not None:
                data = data.loc[show_for]
            if rename is not None:
                data.index = data.index.map(lambda x: rename.get(x, x))
            processed.append(data)

        # Ensure all have the same index order
        idx = processed[0].index
        for d in processed:
            if not all(d.index == idx):
                raise ValueError("All impact_dfs must have the same annotation categories (index) in the same order.")

        # Stack into a DataFrame for scaling
        stacked = pd.concat(processed, axis=1)
        stacked.columns = labels

        # Scale and normalize to 0-100
        scaler = StandardScaler()
        scaled = scaler.fit_transform(stacked)
        scaled = (scaled - scaled.min()) / (scaled.max() - scaled.min()) * 100
        scaled_df = pd.DataFrame(scaled, index=idx, columns=labels)

        # Filter out methods with no variance after normalization
        methods_to_plot = []
        colors_to_plot = []
        for i, label in enumerate(labels):
            variance = scaled_df[label].var()
            if variance > 1e-6:  # Only include methods with variance
                methods_to_plot.append(label)
                colors_to_plot.append(colors[i])

        # If no methods have variance, return empty plot
        if len(methods_to_plot) == 0:
            fig, ax = plt.subplots(figsize=figsize, dpi=dpi)
            ax.text(0.5, 0.5, f'No variance in data for {gene}', 
                   ha='center', va='center', transform=ax.transAxes)
            ax.set_title(f"{gene}", fontsize=label_size+2, pad=20, fontweight='bold')
            plt.tight_layout()
            return fig

        # Radar plot setup
        categories = list(idx)
        N = len(categories)
        angles = np.linspace(0, 2 * np.pi, N, endpoint=False).tolist()
        angles += angles[:1]  # close the loop

        fig, ax = plt.subplots(figsize=figsize, dpi=dpi, subplot_kw={'projection': 'polar'})

        circles = [0, 25, 50, 75, 100]
        ax.set_rticks(circles)
        ax.set_yticklabels([])
        num_vars = len(categories)
        plot_angles = np.linspace(0, 2 * np.pi, num_vars, endpoint=False)
        ax.set_rlim(0, 110)
        for circle in circles:
            if circle > 0:  # Skip the center point
                points = np.array([[circle * np.cos(angle), circle * np.sin(angle)] 
                                   for angle in plot_angles])
                for i in range(len(points)):
                    j = (i + 1) % len(points)
                    ax.plot([np.arctan2(points[i, 1], points[i, 0]), 
                             np.arctan2(points[j, 1], points[j, 0])],
                            [np.hypot(points[i, 0], points[i, 1]), 
                             np.hypot(points[j, 0], points[j, 1])],
                            color='gray', alpha=0.15, linewidth=0.5)
        for angle in plot_angles:
            ax.plot([angle, angle], [0, 110], 
                    color='gray', alpha=0.15, linewidth=0.5)

        for i, label in enumerate(methods_to_plot):
            values = scaled_df[label].values
            values_list = values.tolist()
            values_list += values_list[:1]  # Repeat first value to close polygon
            angles_plot = np.concatenate((plot_angles, [plot_angles[0]]))  # Complete the polygon

            ax.plot(angles_plot, values_list, '-', linewidth=2, label=label, color=colors_to_plot[i])
            ax.fill(angles_plot, values_list, color=colors_to_plot[i], alpha=0.15)

        ax.tick_params(pad=20)

        ax.spines['polar'].set_visible(False)
    
        # Set category labels
        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(categories, fontsize=label_size)
        ax.set_yticks([0, 25, 50, 75, 100])
        ax.set_yticklabels(['' for x in [0, 25, 50, 75, 100]], fontsize=label_size-4)
        ax.set_title(f"{gene}", fontsize=label_size+2, pad=20, fontweight='bold')
        ax.grid(False)
        if show_legend and len(methods_to_plot) > 0:
            legend = ax.legend(bbox_to_anchor=(0.5, -0.15), 
            loc='upper center', ncol=3, frameon=False, fontsize=legend_size)
            for text, line in zip(legend.get_texts(), legend.get_lines()):
                text.set_color(line.get_color())
        plt.tight_layout()
        return fig

    def plot_radar(
        self, 
        genes, 
        impact_df=None, 
        show_for=None, 
        figsize=(20, 6), 
        dpi=300, 
        annot='cell_type', 
        rename=None,
        label_size=20,
        normalize=True,
        legend_size=12,
        cache_path=None,
        colors=None,
        fig=None,
        axes=None
        ):
        
        if isinstance(genes[0], str):
            genes = [genes]
            
        splits = len(genes)
        if axes is None:
            fig, axs = plt.subplots(1, splits, figsize=figsize, dpi=dpi,
                subplot_kw={'projection': 'polar'})
        else:
            axs = axes
            fig = fig
        
        if splits == 1:
            axs = [axs]
        else:
            axs = axs.flatten()
            
        if impact_df is None:
            impact_df = self.compute_ko_impact(genes=genes, annot=annot)
        
        
        if show_for is not None:
            impact_df = impact_df.loc[show_for]
            
        if rename is not None:
            impact_df.index = impact_df.index.map(lambda x: rename.get(x, x))
        
        if normalize:
            ko_concat_norm = pd.DataFrame(
                StandardScaler().fit_transform(impact_df),
                # RobustScaler().fit_transform(impact_df),
                # MinMaxScaler().fit_transform(impact_df),
                index=impact_df.index, 
                columns=impact_df.columns
            )
            
        else:
            ko_concat_norm = impact_df

        ko_concat_norm = (ko_concat_norm - ko_concat_norm.min().min()) /\
            (ko_concat_norm.max().max() - ko_concat_norm.min().min()) * 100
        
        if not normalize:
            ko_concat_norm = ko_concat_norm * 5
            
        for ax_idx, (ax, geneset) in enumerate(zip(axs, genes)):
            ax.grid(False)
            circles = [0, 25, 50, 75, 100]
            ax.set_rticks(circles)
            ax.set_yticklabels([])
            num_vars = len(ko_concat_norm.index)
            angles = np.linspace(0, 2*np.pi, num_vars, endpoint=False)
            ax.set_rlim(0, 110)
            for circle in circles:
                if circle > 0:  # Skip the center point
                    points = np.array([[circle * np.cos(angle), circle * np.sin(angle)] 
                                     for angle in angles])
                    
                    # Connect the points to form the polygon
                    for i in range(len(points)):
                        j = (i + 1) % len(points)
                        
                        # Make the innermost polygon (circle == 25) dotted and more visible
                        if circle == -1:
                            ax.plot([np.arctan2(points[i, 1], points[i, 0]), 
                                    np.arctan2(points[j, 1], points[j, 0])],
                                   [np.hypot(points[i, 0], points[i, 1]), 
                                    np.hypot(points[j, 0], points[j, 1])],
                                   color='gray', alpha=0.6, linewidth=2, linestyle='--')
                        else:
                            ax.plot([np.arctan2(points[i, 1], points[i, 0]), 
                                    np.arctan2(points[j, 1], points[j, 0])],
                                   [np.hypot(points[i, 0], points[i, 1]), 
                                    np.hypot(points[j, 0], points[j, 1])],
                                   color='gray', alpha=0.35, linewidth=0.5)
            
            for angle in angles:
                ax.plot([angle, angle], [0, 110], 
                       color='gray', alpha=0.15, linewidth=0.5)
            
            for i, col in enumerate(geneset):
                values = ko_concat_norm[col].values
                if not np.allclose(values, values[0]):
                    label = rename.get(col, col) if rename is not None else col
                    
                    values_list = values.tolist()
                    values_list += values_list[:1]  # Repeat first value to close polygon
                    
                    angles_plot = np.concatenate((angles, [angles[0]]))  # Complete the polygon

                    # Use custom color if provided, otherwise use default matplotlib colors
                    color = None
                    if colors:
                        # Check if colors is a list of lists (one list per geneset)
                        if isinstance(colors[0], (list, tuple)) if len(colors) > 0 else False:
                            # Use colors for the current geneset (ax_idx)
                            if ax_idx < len(colors) and i < len(colors[ax_idx]):
                                color = colors[ax_idx][i]
                        else:
                            # Use colors as a simple list for all genesets
                            if i < len(colors):
                                color = colors[i]
                    
                    ax.plot(angles_plot, values_list, '-', linewidth=1, label=label, color=color)
                    ax.fill(angles_plot, values_list, alpha=0.3, color=color)

            ax.set_xticks(angles)
            labels = ko_concat_norm.index
            ax.set_xticklabels(
                labels, size=label_size,
                rotation=0  # Keep labels horizontal
            )
            
            ax.tick_params(pad=20)

            ax.spines['polar'].set_visible(False)
            
            legend = ax.legend(bbox_to_anchor=(0.5, -0.15), 
                loc='upper center', ncol=2, frameon=False, fontsize=legend_size)
            if legend:
                for text, line in zip(legend.get_texts(), legend.get_lines()):
                    text.set_color(line.get_color())

        # if splits > 1:
        #     for i in range(1, splits):
        #         fig.add_artist(plt.Line2D([i/splits, i/splits], [0.1, 0.9], 
        #                                 transform=fig.transFigure, color='black', 
        #                                 linestyle='--', linewidth=1, alpha=0.5))
        
        plt.tight_layout()
        
        return fig, axs

    def plot_comparative_bar(
        self, 
        genes, 
        impact_dfs, 
        show_for=None, 
        figsize=(20, 6), 
        dpi=300, 
        annot='cell_type', 
        rename=None, 
        label_size=20, 
        legend_size=12, 
        cache_path=None,
        legend_labels=None):
        """
        Prettier comparative bar plot for gene impacts across cell types and datasets.
        Bars are colored differently depending on x position (cell type), and
        each impact_df is shown as solid, striped, or dotted bars, respectively.
        If a gene is missing from an impact_df, its bars are set to 0.
        """

        n_dfs = len(impact_dfs)
        n_genes = len(genes)

        # Determine cell types to show
        if show_for is not None:
            cell_types = show_for
        else:
            # Use intersection of all cell types present in all dfs
            cell_types = set(impact_dfs[0].index)
            for df in impact_dfs[1:]:
                cell_types = cell_types & set(df.index)
            cell_types = list(cell_types)
        if len(cell_types) == 0:
            raise ValueError("No cell types to show in bar plot.")

        # Prepare data: for each df, get values for cell_types and genes, and normalize per gene (min-max across cell types)
        normed_dfs = []
        for df in impact_dfs:
            # If gene is missing, fill with 0
            subdf = df.reindex(cell_types)
            for gene in genes:
                if gene not in subdf.columns:
                    subdf[gene] = 0.0
            subdf = subdf[genes]
            normed = subdf.copy()
            for gene in genes:
                vals = normed[gene].values.astype(float)
                minv = np.nanmin(vals)
                maxv = np.nanmax(vals)
                if np.isclose(maxv, minv):
                    normed[gene] = 0
                else:
                    normed[gene] = (vals - minv) / (maxv - minv)
            normed_dfs.append(normed)

        # Do NOT sort cell types; keep the order as in cell_types for all genes
        cell_types_per_gene = [cell_types for _ in genes]

        # Set up color palette for cell types (x axis)
        n_cell_types = len(cell_types)
        celltype_palette = sns.color_palette("tab20", n_colors=n_cell_types)
        bar_edge_color = "#333333"

        # Define hatches for each impact_df: solid, striped, dotted
        hatch_styles = ["", "//", "o"]
        # If more than 3 dfs, cycle through hatches
        hatch_cycle = cycle(hatch_styles)

        # Bar chart: for each gene, plot grouped bars for each cell type, with bars for each impact_df
        fig, axes = plt.subplots(
            1, n_genes, 
            figsize=figsize, 
            dpi=dpi, 
            sharey=True, 
            constrained_layout=True
        )
        if n_genes == 1:
            axes = [axes]

        for i, gene in enumerate(genes):
            ax = axes[i]
            this_cell_types = cell_types_per_gene[i]
            x = np.arange(len(this_cell_types))  # cell type positions
            width = 0.7 / n_dfs  # width of each bar, leave some space

            # Draw grid for y axis
            ax.yaxis.grid(True, linestyle='--', linewidth=0.7, alpha=0.4, zorder=0)
            ax.set_axisbelow(True)

            for j, normed in enumerate(normed_dfs):
                vals = normed.loc[this_cell_types, gene].values
                label = None
                if legend_labels is not None and j < len(legend_labels):
                    label = legend_labels[j]
                else:
                    label = f"Set {j+1}"

                # Pick hatch style for this impact_df
                hatch = hatch_styles[j % len(hatch_styles)]

                # For each bar, set color by cell type and hatch by impact_df
                bar_containers = []
                for k, ct in enumerate(this_cell_types):
                    bar_color = celltype_palette[k % len(celltype_palette)]
                    bar = ax.bar(
                        x[k] + j*width - (width*(n_dfs-1)/2),
                        vals[k],
                        width=width,
                        label=label if k == 0 else None,  # only label first bar for legend
                        color=bar_color,
                        edgecolor=bar_edge_color,
                        linewidth=0.8,
                        zorder=3,
                        hatch=hatch
                    )
                    bar_containers.append(bar)

            # Set axis
            ax.set_xticks(x)
            ax.set_xticklabels(
                [rename.get(ct, ct) if rename and ct in rename else ct for ct in this_cell_types], 
                rotation=35, ha='right', fontsize=label_size, color="#222222"
            )
            ax.set_title(
                rename[gene] if rename and gene in rename else gene, 
                size=label_size+2, 
                color="#222222", 
                pad=16, 
            )
            ax.tick_params(axis='y', labelsize=label_size, colors="#222222")
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            ax.spines['left'].set_color("#888888")
            ax.spines['bottom'].set_color("#888888")
            ax.set_facecolor("#fafafa")
            if i == 0:
                ax.set_ylabel("Relative Impact", fontsize=label_size+1, color="#222222", labelpad=10)

        # Legend for impact_dfs (hatch styles)
        # Create custom legend handles for hatches
        legend_handles = []
        legend_labels_final = []
        for j in range(n_dfs):
            hatch = hatch_styles[j % len(hatch_styles)]
            color = "#bbbbbb"
            patch = mpatches.Patch(
                facecolor=color, edgecolor=bar_edge_color, hatch=hatch, linewidth=0.8
            )
            if legend_labels is not None and j < len(legend_labels):
                legend_label = legend_labels[j]
            else:
                legend_label = f"Set {j+1}"
            legend_handles.append(patch)
            legend_labels_final.append(legend_label)
        # Place both legends
        legend1 = fig.legend(
            legend_handles, legend_labels_final, 
            bbox_to_anchor=(0.5, -0.08), 
            loc='upper center', 
            ncol=n_dfs, 
            fontsize=legend_size+2, 
            frameon=False, 
            handletextpad=0.8, 
            columnspacing=1.2,
            title="Dataset (bar style)",
            title_fontsize=legend_size+2
        )
        for text in legend1.get_texts():
            text.set_color("#222222")

        # Remove excess whitespace and prettify layout
        plt.subplots_adjust(wspace=0.18, bottom=0.28, top=0.90)
        fig.patch.set_facecolor("#ffffff")
        plt.tight_layout()

        if cache_path is not None:
            plt.savefig(cache_path, bbox_inches='tight', dpi=dpi, facecolor=fig.get_facecolor())
        return fig

    def smooth_over_manifold(self, obs_key, k=15, use_rep='X_umap', new_key_suffix='_smoothed'):
        """
        Smooths a variable in adata.obs over a specified embedding (default UMAP) using KNN.

        Args:
            adata (AnnData): The annotated data matrix.
            obs_key (str): The column name in adata.obs to smooth.
            k (int): The number of nearest neighbors to use for smoothing.
            use_rep (str): The key in adata.obsm containing the coordinates (default 'X_umap').
            new_key_suffix (str): Suffix added to the original key for the new smoothed column.

        Returns:
            AnnData: The modified AnnData object (modified in place).
        """

        adata = self.chart.adata
        
        if use_rep not in adata.obsm.keys():
            raise ValueError(f"Embedding '{use_rep}' not found in adata.obsm.")
        if obs_key not in adata.obs.columns:
            raise ValueError(f"Column '{obs_key}' not found in adata.obs.")
            
        coords = adata.obsm[use_rep]
        target_values = adata.obs[obs_key].values
        
        nbrs = NearestNeighbors(n_neighbors=k, algorithm='auto').fit(coords)
        _, indices = nbrs.kneighbors(coords)
        
        neighbor_values = target_values[indices]
        
        new_key = f"{obs_key}{new_key_suffix}"
        
        if pd.api.types.is_numeric_dtype(adata.obs[obs_key]):
            smoothed = np.nanmean(neighbor_values, axis=1)
            adata.obs[new_key] = smoothed
            print(f"Smoothed continuous variable '{obs_key}' -> '{new_key}'")
            
        else:
            mode_result = stats.mode(neighbor_values, axis=1, keepdims=False)
            smoothed = mode_result.mode
            adata.obs[new_key] = pd.Categorical(
                smoothed, categories=adata.obs[obs_key].cat.categories)
            print(f"Smoothed categorical variable '{obs_key}' -> '{new_key}'")

        return adata

    def compute_branched_pseudotime(self, pairs, annot, source_cell_type, knn=10, n_components=5, n_source_cells=20):
        import scanpy.external as sce
        adata = self.adata

        source_cells = adata[adata.obs[annot]==source_cell_type].obs_names
        source_cells = np.random.choice(source_cells, n_source_cells)

        pseudo_frames = []

        for pair in pairs:
            _adata = adata[adata.obs[annot].isin([pair[0], pair[1]])].copy()

            sce.tl.palantir(_adata, n_components=n_components, knn=knn)

            for ij, cell in enumerate(source_cells):
                pltr = sce.tl.palantir_results(
                    _adata, 
                    cell
                )

                _df = pltr.pseudotime.to_frame()
                _df.columns=[f'pseudotime_{pairs[0]}+{pairs[1]}_{ij}']
                _adata.obs = _adata.obs.join(_df)

                pseudo_frames.append(
                    _adata.obs[
                        f'pseudotime_{pairs[0]}+{pairs[1]}_{ij}'].to_frame().copy())

        _df_pst = pd.concat(pseudo_frames).fillna(0).sum(1).to_frame()
        _df_pst.columns = ['pseudotime']
        _df_pst = _df_pst.groupby(_df_pst.index).mean()

        if 'pseudotime' in self.adata.obs:
            del self.adata.obs['pseudotime']

        self.adata.obs = self.adata.obs.join(_df_pst)
        self.chart.adata.obs = self.chart.adata.obs.join(_df_pst)


    def calculate_cell_type_alignment(
        self,
        cell_coords, 
        grid_points, 
        grid_cosine_sim, 
        cell_types, 
        agg_func='mean'):

        cell_coords = np.asarray(cell_coords)
        grid_points = np.asarray(grid_points)
        grid_cosine_sim = np.asarray(grid_cosine_sim)
        cell_types = np.asarray(cell_types)

        tree = cKDTree(grid_points)
        distances, indices = tree.query(cell_coords, k=1)
        cell_cosine_sim = grid_cosine_sim[indices]
        df = pd.DataFrame({
            'alignment': cell_cosine_sim,
            'cell_type': cell_types
        })
        
        alignment_per_ctype = df.groupby('cell_type')['alignment'].agg(agg_func)
        
        return alignment_per_ctype.sort_values(ascending=False), df

    def get_umap_centroid_cells(adata, celltype_key="cell_type", umap_key="X_umap"):

        umap = adata.obsm[umap_key]
        celltypes = adata.obs[celltype_key].values
        cell_ids = adata.obs_names

        result = {}

        for ct in np.unique(celltypes):
            mask = celltypes == ct
            coords = umap[mask]

            centroid = coords.mean(axis=0)
            dists = np.linalg.norm(coords - centroid, axis=1)

            closest_idx = np.argmin(dists)
            result[ct] = cell_ids[mask][closest_idx]

        return result


    def compute_vector_alignment(
        self, 
        grid_points, 
        vector_field_a, 
        vector_field_b, 
        annot='cell_type_2',
        obs_key='pseudotime', 
        k=300):

        assert obs_key in self.chart.adata.obs

        self.chart.adata = self.smooth_over_manifold(k=k, obs_key=obs_key)

        ref_flow = self.signature2gradient(
            grid_points=grid_points,
            vector_field=vector_field_a,
            n_knn=100,
            precomputed=obs_key+'_smoothed'
        )

        ref_flow_rand = self.signature2gradient(
            grid_points=grid_points,
            vector_field=vector_field_b,
            n_knn=100,
            precomputed=obs_key+'_smoothed'
        )

        eps = 1e-8

        inner_prod = np.sum(vector_field_a * ref_flow, axis=1)
        norm_v = np.linalg.norm(vector_field_a, axis=1)
        norm_r = np.linalg.norm(ref_flow, axis=1)
        cosine_sim = inner_prod / ((norm_v * norm_r) + eps)

        inner_prod_rand = np.sum(vector_field_b * ref_flow_rand, axis=1)
        norm_v_rand = np.linalg.norm(vector_field_b, axis=1)
        norm_r_rand = np.linalg.norm(ref_flow_rand, axis=1)
        cosine_sim_rand = inner_prod_rand / ((norm_v_rand * norm_r_rand) + eps)

        cell_coords = self.chart.adata.obsm['X_umap']
        cell_types = self.chart.adata.obs[annot].values

        alignment_df, df = self.calculate_cell_type_alignment(
            cell_coords=cell_coords,
            grid_points=grid_points,
            grid_cosine_sim=cosine_sim,
            # grid_cosine_sim=inner_prod,
            cell_types=cell_types
        )

        alignment_df_rand, df_rand = self.calculate_cell_type_alignment(
            cell_coords=cell_coords,
            grid_points=grid_points,
            grid_cosine_sim=cosine_sim_rand,
            # grid_cosine_sim=inner_prod,
            cell_types=cell_types
        )

        return alignment_df, df, alignment_df_rand, df_rand



    
class SubsampledTissue(VirtualTissue):
    
    def __init__(
        self, 
        adata, 
        betadatas_paths=None,  
        ko_paths=None, 
        ovx_paths=None, 
        color_dict=None,
        annot='cell_type',
        suffix = '',
        n_props=4
        ):
        
        
        self.adata = adata
        self.betadatas_paths = betadatas_paths
        self.ko_paths = ko_paths
        self.ovx_paths = ovx_paths
        self.annot = annot
        self.suffix = suffix
        self.n_props = n_props

        
        if ovx_paths is None:
            self.ovx_paths = ko_paths
        
        if color_dict is None:
            
            self.color_dict = {
                c: self.random_color() for c in self.adata.obs[self.annot].unique()
            }
        else:
            self.color_dict = color_dict
        
        self.xy = xy_from_adata(self.adata) 


    def compute_ko_impact(self, genes, cache_path='', force_recompute=False):
        if os.path.exists(cache_path+'ko_impact_df.csv') and not force_recompute:
            return pd.read_csv(cache_path+'ko_impact_df.csv', index_col=0)
        
        ko_data = []
        
        pbar = enlighten.manager.get_manager().counter(
            total=len(sum(genes, [])),
            desc='Computing KO impact',
            unit='KO',
            auto_refresh=True
        )
        for kotarget in sum(genes, []):
            pbar.desc = f'{kotarget:<15}'
            pbar.refresh()

            files = [f'{ko_path}/{kotarget}_{self.n_props}n_0x{self.suffix}.parquet' for ko_path in self.ko_paths]
            
            data = pd.concat([
                pd.read_parquet(ko_file) for ko_file in files
            ], axis=0)
            data = data.loc[self.adata.obs.index]
            # data = self.adata.to_df(layer='imputed_count')
            # data[kotarget] = 0
            
            data = data.loc[self.adata.obs_names] - self.adata.to_df(layer='imputed_count')
            data = data.join(self.adata.obs.cell_type).groupby('cell_type').mean().abs().mean(axis=1)

            ds = {}
            for k, v in data.sort_values(ascending=False).to_dict().items():
                ds[k] = v

            data = pd.DataFrame.from_dict(ds, orient='index')
            data.columns = [kotarget]
            ko_data.append(data)
            pbar.update(1)
        
        out = pd.concat(ko_data, axis=1)
        out.to_csv(cache_path+'ko_impact_df.csv')
        
        return out
