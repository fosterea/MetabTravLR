import copy
from anndata import AnnData
import enlighten
from sklearn.metrics import r2_score
import torch
import os
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from torch.utils.data import DataLoader, Dataset
from sklearn.linear_model import ARDRegression, BayesianRidge
from group_lasso import GroupLasso
from SpaceTravLR.models.spatial_map import xyc2spatial_fast
from SpaceTravLR.tools.network import RegulatoryFactory, expand_paired_interactions
from .pixel_attention import CellularNicheNetwork, CellularViT
from ..tools.utils import gaussian_kernel_2d, is_mouse_data, set_seed
from ..tools.network import get_cellchat_db

# import commot as ct
from scipy.spatial.distance import cdist
import numba
from wordcloud import WordCloud
import matplotlib.pyplot as plt
import pickle
from easydict import EasyDict as edict
from matplotlib.colors import rgb2hex

tt = torch.tensor
set_seed(42)


import warnings

warnings.filterwarnings("ignore")


@numba.njit(parallel=True)
def calculate_weighted_ligands(gauss_weights, lig_df_values, u_ligands):
    n_ligands = len(u_ligands)
    n_cells = len(gauss_weights)
    weighted_ligands = np.zeros((n_ligands, n_cells))

    for i in numba.prange(n_ligands):
        for j in range(n_cells):
            weighted_ligands[i, j] = np.mean(gauss_weights[j] * lig_df_values[:, i])

    return weighted_ligands


def compute_radius_weights(xy, lig_df, radius, scale_factor):
    ligands = lig_df.columns
    gauss_weights = [
        scale_factor * gaussian_kernel_2d(xy[i], xy, radius=radius)
        for i in range(len(lig_df))
    ]
    u_ligands = list(np.unique(ligands))
    lig_df_values = lig_df[u_ligands].values
    weighted_ligands = calculate_weighted_ligands(
        gauss_weights, lig_df_values, u_ligands
    )

    return pd.DataFrame(weighted_ligands, index=u_ligands, columns=lig_df.index).T

@numba.njit(parallel=True)
def _gaussian_kernel_2d_batch(xy: np.ndarray, radius: float) -> np.ndarray:
    """Compute full N×N weight matrix in one parallelized pass."""
    n = xy.shape[0]
    W = np.empty((n, n), dtype=np.float64)
    inv_2r2 = -1.0 / (2.0 * radius * radius)
    for i in numba.prange(n):
        xi, yi = xy[i, 0], xy[i, 1]
        for j in range(n):
            dx = xi - xy[j, 0]
            dy = yi - xy[j, 1]
            W[i, j] = np.exp((dx * dx + dy * dy) * inv_2r2)
    return W  # (n_cells, n_cells)


@numba.njit(parallel=True)
def _weighted_mean(W: np.ndarray, lig_values: np.ndarray) -> np.ndarray:
    n = W.shape[0]
    n_lig = lig_values.shape[1]
    out = np.zeros((n, n_lig), dtype=np.float64)
    for i in numba.prange(n):
        for k in range(n):
            w = W[i, k]
            for j in range(n_lig):
                out[i, j] += w * lig_values[k, j]
        for j in range(n_lig):
            out[i, j] /= n
    return out


def compute_radius_weights_fast(xy, lig_df, radius, scale_factor):
    W = scale_factor * _gaussian_kernel_2d_batch(
        np.ascontiguousarray(xy, dtype=np.float64), radius
    )
    lig_values = np.ascontiguousarray(lig_df.values, dtype=np.float64)
    result = _weighted_mean(W, lig_values)
    return pd.DataFrame(result, index=lig_df.index, columns=lig_df.columns)




def received_ligands(xy, ligands_df, lr_info, scale_factor=1):
    """
    Compute the amount of ligand received on 
    the surface of each cell based on location.

    Args:
        xy (np.ndarray): Array of spatial coordinates (x, y).
        ligands_df (pd.DataFrame): ligand gene expression values.

    Returns:
        pd.DataFrame: DataFrame of received ligands for each cell.
    """
    lr_info = lr_info.copy()
    lr_info = lr_info[lr_info["ligand"].isin(np.unique(ligands_df.columns))]

    lr_info = lr_info[
        lr_info["ligand"].isin(np.unique(ligands_df.columns))
    ].drop_duplicates(subset="ligand", keep="first")

    full_df = []

    for radius in lr_info["radius"].unique():
        radius_ligands = lr_info[lr_info["radius"] == radius]["ligand"].values
        full_df.append(
            compute_radius_weights_fast(xy, ligands_df[radius_ligands], radius, scale_factor)
        )

    full_df = pd.concat([df for df in full_df if not df.empty], axis=1)
    full_df = (
        full_df.reindex(ligands_df.index).reindex(ligands_df.columns, axis=1).fillna(0)
    )

    return full_df


def get_filtered_df(counts_df, cell_thresholds=None, genes=None, min_expression=1e-9):
    """Get filtered expression of ligands/ receptors based on celltype/ thresholds"""

    ligand_counts = counts_df[np.unique(genes)]

    if min_expression > 0:
        mask = np.where(ligand_counts > min_expression, 1, 0)
        ligand_counts = ligand_counts * mask

    if cell_thresholds is not None:
        cell_thresholds = cell_thresholds.loc[counts_df.index]

        assert cell_thresholds.index.equals(counts_df.index), (
            "error aligning cell_thresholds and counts_df, check if obs_names has duplicates"
        )

        mask = cell_thresholds.reindex(ligand_counts.columns, axis=1).fillna(0).values
        mask = np.where(mask > 0, 1, 0)
        ligand_counts = mask * ligand_counts

    # return ligand_counts.reindex(genes, axis=1)
    return ligand_counts


def init_received_ligands(
    adata, radius, cell_threshes, contact_distance=50, scale_factor=1, layer="imputed_count"
):
    species = "mouse" if is_mouse_data(adata) else "human"

    df_ligrec = get_cellchat_db(species)

    lr = expand_paired_interactions(df_ligrec)
    lr = lr[lr.ligand.isin(adata.var_names) & (lr.receptor.isin(adata.var_names))]
    lr["radius"] = np.where(
        lr["signaling"] == "Secreted Signaling", radius, contact_distance
    )

    counts_df = adata.to_df(layer=layer)
    ligands = np.unique(lr.ligand)

    adata.uns["received_ligands"] = received_ligands(
        xy=adata.obsm["spatial"],
        ligands_df=get_filtered_df(
            counts_df, cell_thresholds=cell_threshes, genes=ligands
        ),
        lr_info=lr,
        scale_factor=scale_factor
    )

    adata.uns["received_ligands_tfl"] = received_ligands(
        xy=adata.obsm["spatial"],
        ligands_df=get_filtered_df(
            counts_df, None, genes=ligands
        ),  # Only Commot LRs should be filtered
        lr_info=lr,
    )

    return adata


def create_spatial_features(x, y, celltypes, obs_index, radius=200):
    coords = np.column_stack((x, y))
    unique_celltypes = np.unique(celltypes)
    result = np.zeros((len(x), len(unique_celltypes)))
    distances = cdist(coords, coords)
    for i, celltype in enumerate(unique_celltypes):
        mask = celltypes == celltype
        neighbors = (distances <= radius)[:, mask]
        result[:, i] = np.sum(neighbors, axis=1)

    if result.shape != (len(x), len(unique_celltypes)):
        raise ValueError(f"Expected: {(len(x), len(unique_celltypes))}")

    columns = [f"{ct}_within" for ct in unique_celltypes]
    df = pd.DataFrame(result, columns=columns, index=obs_index)

    return df


if torch.backends.mps.is_available():
    device = torch.device("mps")
elif torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")


class RotatedTensorDataset(Dataset):
    def __init__(
        self, sp_maps, X_cell, y_cell, cluster, spatial_features, rotate_maps=True
    ):
        self.sp_maps = sp_maps
        self.X_cell = X_cell
        self.y_cell = y_cell
        self.cluster = cluster
        self.spatial_features = spatial_features
        self.rotate_maps = rotate_maps

    def __len__(self):
        return len(self.X_cell)

    def __getitem__(self, idx):
        sp_map = self.sp_maps[idx, self.cluster : self.cluster + 1, :, :]
        if self.rotate_maps:
            k = np.random.choice([0, 1, 2, 3])
            sp_map = np.rot90(sp_map, k=k, axes=(1, 2))

        return (
            torch.from_numpy(sp_map.copy()).float(),
            torch.from_numpy(self.X_cell[idx]).float(),
            torch.from_numpy(np.array(self.y_cell[idx])).float(),
            torch.from_numpy(self.spatial_features[idx]).float(),
        )


def init_ligands_and_receptors(
    species,
    adata,
    annot,
    target_gene,
    receptor_thresh,
    radius,
    contact_distance,
    tf_ligand_cutoff,
    regulators,
    grn,
):
    ligand_mixtures = edict()

    # df_ligrec = ct.pp.ligand_receptor_database(
    #         database='CellChat',
    #         species=species,
    #         signaling_type=None
    #     )

    # df_ligrec.columns = ['ligand', 'receptor', 'pathway', 'signaling']

    df_ligrec = get_cellchat_db(species)

    lr = expand_paired_interactions(df_ligrec)
    lr = lr[lr.ligand.isin(adata.var_names) & (lr.receptor.isin(adata.var_names))]

    receptors = list(lr.receptor.values)
    _layer = (
        "normalized_count" if "normalized_count" in adata.layers else "imputed_count"
    )

    # receptor_levels = adata.to_df(layer=_layer)[np.unique(receptors)].join(
    #     adata.obs[annot]).groupby(annot).mean().max(0).to_frame()
    # receptor_levels.columns = ['mean_max']

    # lr = lr[lr.receptor.isin(
    #     receptor_levels.index[receptor_levels['mean_max'] > receptor_thresh])]

    lr["radius"] = np.where(
        lr["signaling"] == "Secreted Signaling", radius, contact_distance
    )

    lr = lr[~((lr.receptor == target_gene) | (lr.ligand == target_gene))]
    lr["pairs"] = lr.ligand.values + "$" + lr.receptor.values
    lr = lr.drop_duplicates(subset="pairs", keep="first")
    ligands = list(lr.ligand.values)
    receptors = list(lr.receptor.values)

    # current_dir = os.path.dirname(os.path.abspath(__file__))
    data_path = f'/Users/koush/Projects/SpaceOracle/data/ligand_target_{species}.parquet'
    # data_path = (
    #     f"https://zenodo.org/records/17594271/files/ligand_target_{species}.parquet"
    # )

    nichenet_lt = pd.read_parquet(data_path)

    nichenet_lt = nichenet_lt.loc[np.intersect1d(nichenet_lt.index, regulators)][
        np.intersect1d(nichenet_lt.columns, ligands)
    ]

    tfl_pairs = []
    tfl_regulators = []
    tfl_ligands = []

    if grn is not None:
        ligand_regulators = {
            lig: set(grn.get_regulators(adata, lig)) for lig in nichenet_lt.columns
        }
    else:
        from collections import defaultdict

        ligand_regulators = defaultdict(list)

    for tf_ in nichenet_lt.index:
        row = nichenet_lt.loc[tf_]
        top_5 = row.nlargest(5)
        for lig_, value in top_5.items():
            if (
                target_gene not in ligand_regulators[lig_]
                and tf_ not in ligand_regulators[lig_]
                and value > tf_ligand_cutoff
            ):
                tfl_ligands.append(lig_)
                tfl_regulators.append(tf_)
                tfl_pairs.append(f"{lig_}#{tf_}")

    assert len(ligands) == len(receptors)
    assert len(tfl_regulators) == len(tfl_ligands)

    ligand_mixtures.lr = lr
    ligand_mixtures.ligands = ligands
    ligand_mixtures.receptors = receptors
    ligand_mixtures.tfl_pairs = tfl_pairs
    ligand_mixtures.tfl_regulators = tfl_regulators
    ligand_mixtures.tfl_ligands = tfl_ligands

    return ligand_mixtures


class SpatialCellularProgramsEstimator:
    """Per-gene spatial regression estimator with ligand-receptor and GRN features.

    ``SpatialCellularProgramsEstimator`` fits a spatially-aware gene-regulatory
    model for a single *target gene*.  For every annotated cell-type cluster it
    trains a :class:`~SpaceTravLR.models.pixel_attention.CellularNicheNetwork`
    (CNN) or
    :class:`~SpaceTravLR.models.pixel_attention.CellularViT` (ViT) that takes
    three inputs:

    1. **Spatial neighbourhood map** – a 2-D density image of each cell-type
       derived from ``adata.obsm["spatial"]``.
    2. **Regulatory features** – expression of transcription factors inferred
       from a supplied GRN, along with ligand-receptor interaction scores and
       ligand-TF interaction scores computed from the CellChat database.
    3. **Spatial context features** – per-cell neighbour-count vectors (one
       column per cell type within ``radius`` microns).

    The model outputs per-cell *spatial beta* coefficients (one per modulator
    + intercept) describing how strongly each regulator, L-R pair or TF-ligand
    pair influences the target gene *at that spatial location*.

    Pipeline overview::

        estimator = SpatialCellularProgramsEstimator(adata, "Myc", grn=grn)
        estimator.fit(num_epochs=50)
        betas = estimator.get_betas()   # DataFrame (cells × modulators)
        estimator.export("./models")

    Attributes:
        adata (AnnData): The spatial single-cell dataset.
        target_gene (str): The gene being modelled.
        regulators (list[str]): Transcription-factor regulators of the target.
        ligands (list[str]): Ligand genes from active L-R pairs.
        receptors (list[str]): Receptor genes paired with ``ligands``.
        tfl_ligands (list[str]): Ligands from TF-ligand (NicheNet) pairs.
        tfl_regulators (list[str]): TFs paired with ``tfl_ligands``.
        lr_pairs (pd.Series): Active ligand-receptor pair identifiers
            (format ``"LigandGene$ReceptorGene"``).
        tfl_pairs (list[str]): Active TF-ligand pair identifiers
            (format ``"LigandGene#TFGene"``).
        modulators (list[str]): Ordered concatenation of regulators, L-R
            pairs, and TF-ligand pairs – defines the column order of the beta
            matrix.
        modulators_genes (list[str]): Unique gene names spanning all
            modulator categories.
        models (dict[int, CellularNicheNetwork]): Trained per-cluster neural
            networks (available after :meth:`fit`).
        scores (dict[int, float]): Per-cluster R² scores on the training set
            (available after :meth:`fit`).
        loss_dict (dict[int, list[float]]): Per-batch MSE losses collected
            during training (available after :meth:`fit`).
        spatial_maps (np.ndarray): Cached neighbourhood image tensors of
            shape ``(n_cells, n_clusters, spatial_dim, spatial_dim)``.
        spatial_features (pd.DataFrame): Min-max scaled neighbour-count
            features (shape ``n_cells × n_clusters``).
        xy (np.ndarray): Spatial coordinates array of shape ``(n_cells, 2)``.
        device (torch.device): PyTorch compute device (CUDA / MPS / CPU).
        species (str): ``"mouse"`` or ``"human"`` inferred from gene names.
    """

    def __init__(
        self,
        adata,
        target_gene,
        spatial_dim=64,
        cluster_annot="cell_type_int",
        layer="imputed_count",
        radius=100,
        contact_distance=30,
        use_ligands=True,
        tf_ligand_cutoff=0.01,
        receptor_thresh=0.1,
        regulators=None,
        grn=None,
        colinks_path=None,
        scale_factor=1,
        extra_regulators=None,
    ):
        """Initialise the estimator and resolve regulatory inputs.

        The constructor performs three main steps:

        1. **GRN lookup** – fetch the list of transcription-factor regulators
           for ``target_gene`` either from an explicit ``regulators`` list or
           via ``grn.get_regulators``.
        2. **Ligand-receptor discovery** – query the CellChat database to
           enumerate L-R pairs and NicheNet TF-ligand pairs that are relevant
           to ``target_gene`` (skipped when ``use_ligands=False``).
        3. **Modulator assembly** – build the ordered ``modulators`` list that
           defines the column ordering of the beta coefficient matrix.

        Args:
            adata (AnnData): Spatial dataset. Must contain
                ``adata.obsm["spatial"]``, the requested ``layer``, and
                ``cluster_annot`` in ``adata.obs``.
            target_gene (str): Name of the gene to model. Must be present in
                ``adata.var_names``.
            spatial_dim (int): Side length (pixels) of the square spatial
                neighbourhood image used as CNN input. Default ``64``.
            cluster_annot (str): ``adata.obs`` column holding integer-encoded
                cell-type labels. Default ``"cell_type_int"``.
            layer (str): ``adata.layers`` key for gene-expression values.
                Default ``"imputed_count"``.
            radius (float): Search radius (same spatial units as
                ``adata.obsm["spatial"]``) for *secreted* ligand diffusion.
                Default ``100``.
            contact_distance (float): Search radius for *contact* signalling
                (juxtacrine / cell-surface). Default ``30``.
            use_ligands (bool): When ``False``, skip all ligand-receptor and
                TF-ligand feature computation. Default ``True``.
            tf_ligand_cutoff (float): Minimum NicheNet ligand-target score
                required to include a TF-ligand pair. Default ``0.01``.
            receptor_thresh (float): Reserved threshold parameter (currently
                unused in filtering). Default ``0.1``.
            regulators (list[str] | None): Explicit list of TF regulators.
                When supplied, ``grn`` and ``colinks_path`` are ignored.
            grn (RegulatoryFactory | None): Pre-built GRN object exposing a
                ``get_regulators(adata, gene)`` method. Mutually exclusive
                with ``colinks_path``.
            colinks_path (str | None): Path to a co-link CSV used to
                construct a :class:`~SpaceTravLR.tools.network.RegulatoryFactory`
                internally. Required when both ``grn`` and ``regulators`` are
                ``None``.
            scale_factor (float): Multiplicative weight applied to Gaussian
                kernel values during ligand diffusion. Default ``1``.
            extra_regulators (list[str] | None): Additional gene names to
                append to the regulator list after GRN lookup.
        """
        assert isinstance(adata, AnnData), "adata must be an AnnData object"
        assert target_gene in adata.var_names, "target_gene must be in adata.var_names"
        assert layer in adata.layers, "layer must be in adata.layers"
        assert cluster_annot in adata.obs.columns, (
            "cluster_annot must be in adata.obs.columns"
        )

        self.adata = adata
        self.scale_factor = scale_factor
        self.use_ligands = use_ligands
        self.target_gene = target_gene
        self.cluster_annot = cluster_annot
        self.layer = layer
        self.device = device
        self.radius = radius
        self.contact_distance = contact_distance
        self.spatial_dim = spatial_dim
        self.tf_ligand_cutoff = tf_ligand_cutoff
        self.receptor_thresh = receptor_thresh
        self.xy = pd.DataFrame(
            adata.obsm["spatial"], index=adata.obs.index, columns=["x", "y"]
        )

        self.species = "mouse" if is_mouse_data(adata) else "human"

        if regulators is None:
            if grn is None:
                assert colinks_path is not None, (
                    "colinks_path must be provided if grn is None"
                )
                self.grn = RegulatoryFactory(
                    colinks_path=colinks_path, annot=cluster_annot
                )
            else:
                self.grn = grn

            self.regulators = self.grn.get_regulators(self.adata, self.target_gene)

        else:
            self.regulators = list(regulators)
            self.grn = None

        if extra_regulators is not None:
            self.regulators = list(set(self.regulators) | set(extra_regulators))

        # Exclude target gene from regulators to avoid data leak
        self.regulators = [
            i for i in self.regulators if i in adata.var_names and i != self.target_gene
        ]

        if self.use_ligands:
            ligand_mixtures = init_ligands_and_receptors(
                species=self.species,
                adata=self.adata,
                annot=self.cluster_annot,
                target_gene=self.target_gene,
                receptor_thresh=self.receptor_thresh,
                radius=self.radius,
                contact_distance=self.contact_distance,
                tf_ligand_cutoff=self.tf_ligand_cutoff,
                regulators=self.regulators,
                grn=self.grn,
            )

            self.lr = ligand_mixtures.lr
            self.ligands = ligand_mixtures.ligands
            self.receptors = ligand_mixtures.receptors
            self.tfl_pairs = ligand_mixtures.tfl_pairs
            self.tfl_regulators = ligand_mixtures.tfl_regulators
            self.tfl_ligands = ligand_mixtures.tfl_ligands

        else:
            self.lr = pd.DataFrame(
                columns=["ligand", "receptor", "pathway", "signaling"]
            )
            self.lr["pairs"] = self.lr.ligand.values + "$" + self.lr.receptor.values
            self.ligands = []
            self.receptors = []
            self.tfl_pairs = []
            self.tfl_regulators = []
            self.tfl_ligands = []

        self.lr_pairs = self.lr["pairs"]

        self.n_clusters = len(self.adata.obs[self.cluster_annot].unique())
        self.modulators = self.regulators + list(self.lr_pairs) + self.tfl_pairs

        self.modulators_genes = list(
            np.unique(
                self.regulators
                + self.ligands
                + self.receptors
                + self.tfl_regulators
                + self.tfl_ligands
            )
        )

        assert len(self.ligands) == len(self.receptors)
        assert np.isin(self.ligands, self.adata.var_names).all()
        assert np.isin(self.receptors, self.adata.var_names).all()
        assert np.isin(self.regulators, self.adata.var_names).all()

    def plot_modulators(self, use_expression=True):
        """Visualise all modulator genes as a word cloud.

        Renders a word cloud where each word is a gene that modulates the
        target (regulators, ligands, receptors, TF-ligand partners). Word
        size is proportional to mean expression when ``use_expression=True``.
        Colour encodes modulator category:

        * **viridis** – ligands (L-R and TF-ligand).
        * **magma** – receptors.
        * **rainbow** – transcription factor regulators.

        Args:
            use_expression (bool): Scale word size by mean expression across
                cells. When ``False`` all words are equal size. Default
                ``True``.

        Returns:
            None: Displays the plot via :func:`matplotlib.pyplot.show`.
        """
        if use_expression:
            # Get mean expression values for each gene
            genes = list(
                set(
                    self.regulators
                    + self.ligands
                    + self.tfl_ligands
                    + self.receptors
                    + self.tfl_regulators
                )
            )
            expr_values = self.adata.to_df(layer=self.layer)[genes].mean(axis=0)
            word_freq = {gene: float(expr) for gene, expr in zip(genes, expr_values)}
        else:
            word_freq = {
                reg: 1
                for reg in set(
                    self.regulators
                    + self.ligands
                    + self.tfl_ligands
                    + self.receptors
                    + self.tfl_regulators
                )
            }

        ligand_cmap = plt.get_cmap("viridis")
        receptor_cmap = plt.get_cmap("magma")
        regulator_cmap = plt.get_cmap("rainbow")

        def my_color_func(
            word, font_size, position, orientation, font_path, random_state
        ):
            rnd = random_state.random()  # random float in [0.0, 1.0)
            if word in set(self.ligands).union(self.tfl_ligands):
                color = ligand_cmap(rnd)
                return rgb2hex(color[:3])
            elif word in set(self.receptors):
                color = receptor_cmap(rnd)
                return rgb2hex(color[:3])
            elif word in set(self.regulators).union(self.tfl_regulators):
                color = regulator_cmap(rnd)
                return rgb2hex(color[:3])
            else:
                return "grey"

        wordcloud = WordCloud(
            width=800,
            height=300,
            contour_width=1,
            contour_color="black",
            background_color="white",
            color_func=my_color_func,
        ).generate_from_frequencies(word_freq)

        plt.figure(figsize=(16, 8))
        plt.imshow(wordcloud, interpolation="bilinear", aspect="equal")
        plt.axis("off")
        plt.title(f"{self.target_gene} modulators", fontsize=20)
        plt.show()

    @staticmethod
    def ligands_receptors_interactions(received_ligands_df, receptor_gex_df):
        """Compute element-wise ligand × receptor interaction scores.

        For each paired column ``(ligand_i, receptor_i)`` the interaction
        score is the Hadamard (element-wise) product of the *received* ligand
        signal (Gaussian-weighted neighbourhood average) and the cell's own
        receptor expression::

            score_{i,j} = received_ligand_{i,j} × receptor_expr_{i,j}

        The resulting columns are named ``"LigandGene$ReceptorGene"``,
        matching the format used throughout the modulator pipeline.

        Args:
            received_ligands_df (pd.DataFrame): DataFrame of shape
                ``(n_cells, n_pairs)`` containing diffused ligand abundances
                (output of :func:`received_ligands`).
            receptor_gex_df (pd.DataFrame): DataFrame of the same shape
                containing receptor expression values for each paired
                receptor.

        Returns:
            pd.DataFrame: Interaction score matrix of shape
            ``(n_cells, n_pairs)`` with columns in ``"Lig$Rec"`` format.
        """
        assert isinstance(received_ligands_df, pd.DataFrame)
        assert isinstance(receptor_gex_df, pd.DataFrame)
        assert received_ligands_df.index.equals(receptor_gex_df.index)
        assert received_ligands_df.shape[1] == receptor_gex_df.shape[1]

        _received_ligands = received_ligands_df.values
        _self_receptor_expression = receptor_gex_df.values
        lr_interactions = _received_ligands * _self_receptor_expression

        return pd.DataFrame(
            lr_interactions,
            columns=[
                i[0] + "$" + i[1]
                for i in zip(received_ligands_df.columns, receptor_gex_df.columns)
            ],
            index=receptor_gex_df.index,
        )

    @staticmethod
    def ligand_regulators_interactions(received_ligands_df, regulator_gex_df):
        """Compute element-wise ligand × TF (NicheNet) interaction scores.

        Analogous to :meth:`ligands_receptors_interactions` but for the
        NicheNet TF-ligand axis. For each ``(ligand_i, tf_i)`` pair the
        score is the product of the received ligand signal and the TF's
        expression in the same cell::

            score_{i,j} = received_ligand_{i,j} × tf_expr_{i,j}

        Columns are named ``"LigandGene#TFGene"``.

        Args:
            received_ligands_df (pd.DataFrame): Received ligand signals,
                shape ``(n_cells, n_pairs)``.
            regulator_gex_df (pd.DataFrame): TF expression values of the
                same shape.

        Returns:
            pd.DataFrame: Interaction score matrix of shape
            ``(n_cells, n_pairs)`` with columns in ``"Lig#TF"`` format.
        """
        assert isinstance(received_ligands_df, pd.DataFrame)
        assert isinstance(regulator_gex_df, pd.DataFrame)
        assert received_ligands_df.index.equals(regulator_gex_df.index)
        assert received_ligands_df.shape[1] == regulator_gex_df.shape[1]

        _received_ligands = received_ligands_df.values
        _self_regulator_expression = regulator_gex_df.values
        ltf_interactions = _received_ligands * _self_regulator_expression

        return pd.DataFrame(
            ltf_interactions,
            columns=[
                i[0] + "#" + i[1]
                for i in zip(received_ligands_df.columns, regulator_gex_df.columns)
            ],
            index=regulator_gex_df.index,
        )

    @staticmethod
    def check_LR_properties(adata, layer):
        """Retrieve expression counts and optional cell-type thresholds.

        A lightweight helper used by :meth:`init_data` to unpack the two
        artefacts needed for L-R filtering: the full expression DataFrame
        and the per-cell-type expression threshold mask stored in
        ``adata.uns["cell_thresholds"]``.

        Args:
            adata (AnnData): The dataset to query.
            layer (str): Layer key for expression values.

        Returns:
            tuple[pd.DataFrame, pd.DataFrame | None]:
                ``(counts_df, cell_thresholds)``.
                ``cell_thresholds`` is ``None`` when the key is absent from
                ``adata.uns``.
        """
        counts_df = adata.to_df(layer=layer)
        cell_thresholds = adata.uns.get("cell_thresholds", None)

        if cell_thresholds is None:
            print("warning: cell_thresholds not found in adata.uns")

        return counts_df, cell_thresholds

    def init_data(self):
        """Build all training matrices and cache them on ``self``.

        This method orchestrates the full feature-engineering pipeline and
        **must** be called (implicitly via :meth:`fit`) before
        :meth:`get_betas` can be used.  It performs the following steps in
        order:

        1. Compute (or reuse cached) *received ligand* signals and store them
           in ``adata.uns["received_ligands"]`` / ``"received_ligands_tfl"``.
        2. Compute L-R interaction scores and store them in
           ``adata.uns["ligand_receptor"]``.
        3. Compute TF-ligand interaction scores and store them in
           ``adata.uns["ligand_regulator"]``.
        4. Build (or reuse cached) spatial neighbourhood image tensors
           ``adata.obsm["spatial_maps"]``.
        5. Build (or reuse cached) spatial context features
           ``adata.obsm["spatial_features"]`` and normalise them with
           :class:`~sklearn.preprocessing.MinMaxScaler`.
        6. Construct the training DataFrame ``self.train_df`` combining all
           feature groups.
        7. Filter out any L-R / TF-ligand pairs that produced zero-variance
           features.
        8. Update ``self.modulators`` and ``self.modulators_genes`` to
           reflect surviving pairs.

        Side effects:
            Populates ``self.spatial_maps``, ``self.spatial_features``,
            ``self.train_df``, ``self.xy``, ``self.xy_df``, ``self.lr_pairs``,
            ``self.tfl_pairs``, ``self.ligands``, ``self.receptors``,
            ``self.tfl_ligands``, ``self.tfl_regulators``.

        Returns:
            tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
                ``(sp_maps, X, y, cluster_labels)`` ready for the training
                loop in :meth:`fit`.

                * ``sp_maps`` – shape ``(n_cells, n_clusters, D, D)``.
                * ``X`` – design matrix of shape ``(n_cells, n_modulators)``.
                * ``y`` – target expression vector of shape ``(n_cells,)``.
                * ``cluster_labels`` – integer cluster label per cell.
        """

        lr_info = self.check_LR_properties(self.adata, self.layer)
        counts_df, cell_thresholds = lr_info

        if not (
            ("received_ligands" in self.adata.uns.keys())
            | ("received_ligands_tfl" in self.adata.uns.keys())
        ):
            print(f"Initializing received ligands with {self.scale_factor}")
            self.adata = init_received_ligands(
                self.adata,
                radius=self.radius,
                contact_distance=self.contact_distance,
                cell_threshes=cell_thresholds,
                scale_factor=self.scale_factor
            )

        if len(self.lr["pairs"]) > 0:
            self.adata.uns["ligand_receptor"] = self.ligands_receptors_interactions(
                self.adata.uns["received_ligands"][self.ligands],
                get_filtered_df(counts_df, cell_thresholds, self.receptors)[
                    self.receptors
                ],
            )

        else:
            self.adata.uns["received_ligands"] = pd.DataFrame(
                index=self.adata.obs.index
            )
            self.adata.uns["ligand_receptor"] = pd.DataFrame(index=self.adata.obs.index)

        if len(self.tfl_pairs) > 0:
            self.adata.uns["ligand_regulator"] = self.ligand_regulators_interactions(
                self.adata.uns["received_ligands_tfl"][self.tfl_ligands],
                self.adata.to_df(layer=self.layer)[self.tfl_regulators],
            )
        else:
            self.adata.uns["ligand_regulator"] = pd.DataFrame(
                index=self.adata.obs.index
            )

        self.xy = np.array(self.adata.obsm["spatial"])
        cluster_labels = np.array(self.adata.obs[self.cluster_annot])

        self.xy_df = pd.DataFrame(
            self.xy, columns=["x", "y"], index=self.adata.obs.index
        )

        if not "spatial_maps" in self.adata.obsm.keys():
            self.spatial_maps = xyc2spatial_fast(
                xyc=np.column_stack([self.xy, cluster_labels]),
                m=self.spatial_dim,
                n=self.spatial_dim,
            )

            self.adata.obsm["spatial_maps"] = self.spatial_maps

        else:
            self.spatial_maps = self.adata.obsm["spatial_maps"]

        self.train_df = (
            self.adata.to_df(layer=self.layer)[[self.target_gene] + self.regulators]
            .join(self.adata.uns["ligand_receptor"])
            .join(self.adata.uns["ligand_regulator"])
        )

        if not "spatial_features" in self.adata.obsm.keys():
            self.spatial_features = create_spatial_features(
                self.adata.obsm["spatial"][:, 0],
                self.adata.obsm["spatial"][:, 1],
                self.adata.obs[self.cluster_annot],
                self.adata.obs.index,
                radius=self.radius,
            )

            self.adata.obsm["spatial_features"] = self.spatial_features.copy()

        else:
            self.spatial_features = self.adata.obsm["spatial_features"]

        self.spatial_features = pd.DataFrame(
            MinMaxScaler().fit_transform(self.spatial_features.values),
            columns=self.spatial_features.columns,
            index=self.spatial_features.index,
        )

        self.lr_pairs = self.lr_pairs[self.lr_pairs.isin(self.train_df.columns)]
        self.tfl_pairs = [i for i in self.tfl_pairs if i in self.train_df.columns]

        self.ligands = []
        self.receptors = []
        self.tfl_regulators = []
        self.tfl_ligands = []

        for i in self.lr_pairs:
            lig, rec = i.split("$")
            self.ligands.append(lig)
            self.receptors.append(rec)

        for i in self.tfl_pairs:
            lig, reg = i.split("#")
            self.tfl_ligands.append(lig)
            self.tfl_regulators.append(reg)

        self.modulators = self.regulators + list(self.lr_pairs) + self.tfl_pairs
        self.modulators_genes = list(
            np.unique(
                self.regulators
                + self.ligands
                + self.receptors
                + self.tfl_regulators
                + self.tfl_ligands
            )
        )

        assert len(self.ligands) == len(self.receptors)

        X = self.train_df.drop(columns=[self.target_gene]).values
        y = self.train_df[self.target_gene].values
        sp_maps = self.spatial_maps

        assert sp_maps.shape[0] == X.shape[0] == y.shape[0] == len(cluster_labels)

        return sp_maps, X, y, cluster_labels

    @torch.no_grad()
    def get_betas(self):
        """Extract per-cell spatial beta coefficients from trained models.

        Iterates over each unique cluster label, forwards the corresponding
        spatial neighbourhood images and spatial context features through the
        cluster's :class:`~SpaceTravLR.models.pixel_attention.CellularNicheNetwork`,
        and collects the returned beta vectors.  Cells whose cluster has no
        trained model (e.g. clusters with R²  < threshold) receive an
        all-zeros beta row.

        The beta matrix has columns:

        * ``"beta0"`` – the learned intercept / baseline expression.
        * ``"beta_<modulator>"`` – one column per entry in ``self.modulators``
          following the order: regulators → L-R pairs → TF-ligand pairs.

        This method is decorated with ``@torch.no_grad()`` so no gradients
        are computed during inference.

        Returns:
            pd.DataFrame: Shape ``(n_cells, 1 + n_modulators)`` indexed by
            ``adata.obs.index``.  Rows are sorted to match the original cell
            order regardless of the per-cluster iteration order.
        """
        index_tracker = []
        betas = []
        for cluster_target in np.unique(self.cluster_labels):
            mask = self.cluster_labels == cluster_target
            indices = self.cell_indices[mask]
            index_tracker.extend(indices)

            if cluster_target not in self.models:
                print(f"No model found for {cluster_target}")
                b = np.zeros((len(indices), (len(self.modulators) + 1)))

            else:
                cluster_sp_maps = torch.from_numpy(
                    self.sp_maps[mask][:, cluster_target : cluster_target + 1, :, :]
                ).float()
                spf = torch.from_numpy(self.spatial_features.values[mask]).float()

                b = (
                    self.models[cluster_target]
                    .get_betas(cluster_sp_maps.to(self.device), spf.to(self.device))
                    .cpu()
                    .numpy()
                )

            betas.extend(b)

        return pd.DataFrame(
            betas,
            index=index_tracker,
            columns=["beta0"] + ["beta_" + i for i in self.modulators],
        ).reindex(self.adata.obs.index)

    @property
    def betadata(self):
        """Alias for :meth:`get_betas` kept for backward compatibility.

        Returns:
            pd.DataFrame: Per-cell spatial beta coefficients (see
            :meth:`get_betas` for full description).
        """
        return self.get_betas()

    def fit(
        self,
        num_epochs=100,
        threshold_lambda=1e-6,
        learning_rate=5e-3,
        batch_size=512,
        pbar=None,
        estimator="lasso",
        vision_model="cnn",
        score_threshold=0.2,
        l1_reg=1e-9,
        skip_clusters=None,
    ):
        """Train per-cluster spatial cellular programme models.

        For each unique cluster label the method:

        1. Fits a *seed estimator* (Group Lasso, ARD, or Bayesian Ridge) on
           the cluster's cells to obtain initial ``_betas`` anchors.
        2. If the seed R² exceeds ``score_threshold``, trains a
           :class:`~SpaceTravLR.models.pixel_attention.CellularNicheNetwork`
           or :class:`~SpaceTravLR.models.pixel_attention.CellularViT` with
           those anchors using Adam + MSE loss.
        3. If the neural-network R² after training remains below
           ``score_threshold``, the model anchors are zeroed out so the
           gene's spatial betas collapse to the baseline (intercept only).

        The seed estimator types are:

        * ``"lasso"`` *(default)* – Group Lasso with feature groups
          ``(regulators, L-R pairs, TF-ligand pairs)`` and
          ``threshold_lambda`` as the group regularisation strength.
        * ``"bayesian"`` – :class:`~sklearn.linear_model.BayesianRidge`.
        * ``"ard"`` – :class:`~sklearn.linear_model.ARDRegression`.
          Scales as *O(n²)* in samples; avoid for large datasets.

        Args:
            num_epochs (int): Number of neural-network training epochs per
                cluster. Set to ``0`` to skip neural training (seed betas
                only). Default ``100``.
            threshold_lambda (float): Group-Lasso regularisation strength
                (and ARD ``threshold_lambda``). Default ``1e-6``.
            learning_rate (float): Adam learning rate. Default ``5e-3``.
            batch_size (int): Mini-batch size for the DataLoader.
                Default ``512``.
            pbar: An :mod:`enlighten` progress-bar counter.  When ``None``
                a new manager is created automatically.
            estimator (str): Seed estimator type; one of ``"lasso"``,
                ``"bayesian"``, ``"ard"``. Default ``"lasso"``.
            vision_model (str): Neural architecture; one of ``"cnn"``
                (:class:`~SpaceTravLR.models.pixel_attention.CellularNicheNetwork`)
                or ``"transformer"``
                (:class:`~SpaceTravLR.models.pixel_attention.CellularViT`).
                Default ``"cnn"``.
            score_threshold (float): Minimum seed R² required to proceed
                with neural training. Clusters below this threshold receive
                zero-anchor models. Default ``0.2``.
            l1_reg (float): Element-wise L1 penalty applied on top of the
                Group Lasso group penalty. Default ``1e-9``.
            skip_clusters (list[int] | None): Cluster integer labels to
                skip entirely (progress bar is still advanced). Default
                ``None``.

        Returns:
            None: Populates ``self.models``, ``self.scores``, and
            ``self.loss_dict`` in-place.
        """
        sp_maps, X, y, cluster_labels = self.init_data()

        if skip_clusters is None:
            skip_clusters = []

        assert estimator in ["lasso", "bayesian", "ard"]
        assert vision_model in ["cnn", "transformer"]

        self.estimator = estimator
        self.vision_model = vision_model
        self.models = {}
        self.Xn = X
        self.yn = y
        self.sp_maps = sp_maps
        self.cell_indices = self.adata.obs.index.copy()
        self.cluster_labels = cluster_labels

        if pbar is None:
            manager = enlighten.get_manager()
            pbar = manager.counter(
                total=sp_maps.shape[0] * num_epochs,
                desc="Estimating Spatial Betas",
                unit="cells",
                color="green",
                auto_refresh=True,
            )

        if num_epochs:
            print(f"Fitting {self.target_gene} with {len(self.modulators)} modulators")
            print(f"\t{len(self.regulators)} Transcription Factors")
            print(f"\t{len(self.lr_pairs)} Ligand-Receptor Pairs")
            print(f"\t{len(self.tfl_pairs)} TranscriptionFactor-Ligand Pairs")

        self.scores = {}

        self.loss_dict = {}

        for cluster in np.unique(cluster_labels):
            if int(cluster) in skip_clusters:
                pbar.update(
                    num_epochs * len(self.cell_indices[cluster_labels == cluster])
                )
                continue

            mask = cluster_labels == cluster
            X_cell, y_cell = self.Xn[mask], self.yn[mask]

            if self.estimator == "ard":
                """
                ARD allocates a n_samples * n_samples matrix so isn't very scalable
                """
                m = ARDRegression(threshold_lambda=threshold_lambda)
                m.fit(X_cell, y_cell)
                y_pred = m.predict(X_cell)
                r2 = r2_score(y_cell, y_pred)
                _betas = np.hstack([m.intercept_, m.coef_])
                coefs = None

            elif self.estimator == "bayesian":
                m = BayesianRidge()
                m.fit(X_cell, y_cell)
                y_pred = m.predict(X_cell)
                r2 = r2_score(y_cell, y_pred)
                _betas = np.hstack([m.intercept_, m.coef_])

            elif self.estimator == "lasso":
                groups = (
                    [1] * len(self.regulators)
                    + [2] * len(self.lr_pairs)
                    + [3] * len(self.tfl_pairs)
                )
                groups = np.array(groups)
                gl = GroupLasso(
                    groups=groups,
                    group_reg=threshold_lambda,
                    l1_reg=l1_reg,
                    frobenius_lipschitz=True,
                    scale_reg="inverse_group_size",
                    warm_start=True,
                    random_state=42,
                    # subsampling_scheme=1,
                    supress_warning=True,
                    n_iter=1500,
                    # warm_start=True,
                    tol=1e-5,
                )

                gl.fit(X_cell, y_cell)
                y_pred = gl.predict(X_cell)
                coefs = gl.coef_.flatten()
                _betas = np.hstack([gl.intercept_, coefs])
                r2 = r2_score(y_cell, y_pred)

            self.scores[cluster] = r2

            if r2 < 0.15:
                _model = CellularNicheNetwork(
                    n_modulators=len(self.modulators),
                    anchors=_betas * 0,
                    spatial_dim=self.spatial_dim,
                    n_clusters=self.n_clusters,
                ).to(self.device)

                self.models[cluster] = _model

                print(f"{cluster}: x.xxx* | {r2:.4f}")
                pbar.update(len(X_cell) * num_epochs)
                continue

            loader = DataLoader(
                RotatedTensorDataset(
                    sp_maps[mask],
                    X_cell,
                    y_cell,
                    cluster,
                    self.spatial_features.iloc[mask].values,
                    rotate_maps=True,
                ),
                batch_size=batch_size,
                shuffle=True,
            )

            assert _betas.shape[0] == len(self.modulators) + 1

            if self.vision_model == "cnn":
                model = CellularNicheNetwork(
                    n_modulators=len(self.modulators),
                    anchors=_betas,
                    spatial_dim=self.spatial_dim,
                    n_clusters=self.n_clusters,
                ).to(self.device)

            elif self.vision_model == "transformer":
                model = CellularViT(
                    n_modulators=len(self.modulators),
                    anchors=_betas,
                    spatial_dim=self.spatial_dim,
                    n_clusters=self.n_clusters,
                ).to(self.device)

            criterion = torch.nn.MSELoss()
            optimizer = torch.optim.Adam(
                model.parameters(), lr=learning_rate, weight_decay=0
            )

            self.loss_dict[cluster] = []

            for epoch in range(num_epochs):
                model.train()
                epoch_loss = 0

                all_y_true = []
                all_y_pred = []

                for batch in loader:
                    spatial_maps, inputs, targets, spatial_features = [
                        b.to(device) for b in batch
                    ]

                    optimizer.zero_grad()
                    outputs = model(spatial_maps, inputs, spatial_features)
                    loss = criterion(outputs, targets)
                    # loss += torch.mean(outputs.mean(0) - model.anchors) * 1e-5
                    loss.backward()

                    torch.nn.utils.clip_grad_norm_(
                        model.parameters(), max_norm=2.0, norm_type=2
                    )
                    optimizer.step()

                    epoch_loss += loss.item()
                    all_y_true.extend(targets.cpu().detach().numpy())
                    all_y_pred.extend(outputs.cpu().detach().numpy())

                    pbar.desc = f"{self.target_gene} | {cluster + 1}/{self.n_clusters}"
                    pbar.update(len(targets))

                    self.loss_dict[cluster].append(loss.item())

            if num_epochs:
                score = r2_score(all_y_true, all_y_pred)
                if score < score_threshold:
                    # no point in predicting betas if we do it poorly
                    model.anchors = model.anchors * 0.0
                    print(f"{cluster}: x.xxxx | {r2:.4f}")

                else:
                    print(f"{cluster}: {score:.4f} | {r2:.4f}")

            self.models[cluster] = model

    def export(self, save_dir="./models"):
        """Serialise the trained estimator to disk.

        PyTorch :class:`~torch.nn.Module` objects are not directly
        picklable in all configurations.  This method works around that by
        converting each cluster model to a plain ``dict`` of
        ``{state_dict, anchors}`` before pickling the estimator, then saves
        the result as ``<save_dir>/<target_gene>_estimator.pkl``.

        Usage::

            estimator.fit(num_epochs=50)
            estimator.export("./output/models")
            # → ./output/models/Myc_estimator.pkl

        Args:
            save_dir (str): Directory path to save the ``.pkl`` file.
                Created automatically if it does not exist.
                Default ``"./models"``.

        Returns:
            None
        """
        # Create a copy of self that we can modify
        export_obj = copy.copy(self)

        # Extract state dicts and anchors from models
        model_states = {}
        for cluster, model in self.models.items():
            if model is None:
                model_states[cluster] = None
            else:
                model_states[cluster] = {
                    "state_dict": model.state_dict(),
                    "anchors": model.anchors,
                }

        # Replace model objects with None before pickling
        export_obj.models = model_states

        # Save the modified object
        os.makedirs(save_dir, exist_ok=True)
        with open(
            os.path.join(save_dir, f"{self.target_gene}_estimator.pkl"), "wb"
        ) as f:
            pickle.dump(export_obj, f)

    def load(self, path):
        """Restore a previously exported estimator from disk.

        Reads the pickle file written by :meth:`export`, copies all
        non-model attributes back onto ``self``, then reconstructs each
        per-cluster :class:`~SpaceTravLR.models.pixel_attention.CellularNicheNetwork`
        from its saved state dict and anchors.

        Usage::

            estimator = SpatialCellularProgramsEstimator(adata, "Myc", grn=grn)
            estimator.load("./output/models/Myc_estimator.pkl")
            betas = estimator.get_betas()

        Args:
            path (str): Absolute or relative path to the ``.pkl`` file
                produced by :meth:`export`.

        Returns:
            None: Modifies ``self`` in-place.
        """
        with open(path, "rb") as f:
            loaded = pickle.load(f)

        # Copy all attributes except models
        for attr, val in loaded.__dict__.items():
            if attr != "models":
                setattr(self, attr, val)

        # Reconstruct models from state dicts
        self.models = {}
        for cluster, state in loaded.models.items():
            model = CellularNicheNetwork(
                n_modulators=len(self.modulators),
                anchors=state["anchors"],
                spatial_dim=self.spatial_dim,
                n_clusters=self.n_clusters,
            ).to(self.device)
            model.load_state_dict(state)
            self.models[cluster] = model
