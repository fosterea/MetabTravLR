from numba import jit, njit, prange
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import torch
import random
import functools
import inspect
import warnings
import pickle
from sklearn.neighbors import NearestNeighbors
from scipy.sparse import csr_matrix
from scipy import sparse
from tqdm import tqdm
import io
import networkx as nx
from sklearn.neighbors import NearestNeighbors



class CPU_Unpickler(pickle.Unpickler):
    def find_class(self, module, name):
        if module == 'torch.storage' and name == '_load_from_bytes':
            return lambda b: torch.load(io.BytesIO(b), map_location='cpu')
        else:
            return super().find_class(module, name)


def search(query, string_list):
    return [i for i in string_list if query.lower() in i.lower()]


def scale_adata(adata, cell_size=15):
    nbrs = NearestNeighbors(n_neighbors=2, algorithm='ball_tree').fit(adata.obsm['spatial'])
    distances, indices = nbrs.kneighbors(adata.obsm['spatial'])

    # nn_distance = np.percentile(distances[:, 1], 5).min() # maybe 5% cells are squished
    nn_distance = np.median(distances[:, 1])
    scale_factor = cell_size / nn_distance
    adata.obsm['spatial_unscaled'] = adata.obsm['spatial'].copy()
    adata.obsm['spatial'] *= scale_factor
    
    return adata


def knn_distance_matrix(data, metric=None, k=40, mode='connectivity', n_jobs=4):
    """Calculate a nearest neighbour distance matrix

    Notice that k is meant as the actual number of neighbors NOT INCLUDING itself
    To achieve that we call kneighbors_graph with X = None
    """
    if metric == "correlation":
        nn = NearestNeighbors(
            n_neighbors=k, metric="correlation", 
            algorithm="brute", n_jobs=n_jobs)
        nn.fit(data)
        return nn.kneighbors_graph(X=None, mode=mode)
    else:
        nn = NearestNeighbors(n_neighbors=k, n_jobs=n_jobs, )
        nn.fit(data)
        return nn.kneighbors_graph(X=None, mode=mode)

def connectivity_to_weights(mknn, axis=1):
    if type(mknn) is not sparse.csr_matrix:
        mknn = mknn.tocsr()
    return mknn.multiply(1. / sparse.csr_matrix.sum(mknn, axis=axis))

def convolve_by_sparse_weights(data, w):
    w_ = w.T
    assert np.allclose(w_.sum(0), 1)
    return sparse.csr_matrix.dot(data, w_)


def _adata_to_matrix(adata, layer_name, transpose=True):
    if isinstance(adata.layers[layer_name], np.ndarray):
        matrix = adata.layers[layer_name].copy()
    else:
        matrix = adata.layers[layer_name].todense().A.copy()

    if transpose:
        matrix = matrix.transpose()

    return matrix.copy(order="C")





class DeprecatedWarning(UserWarning):
    pass

def deprecated(instructions=''):
    """Flags a method as deprecated.

    Args:
        instructions: A human-friendly string of instructions, such
            as: 'Please migrate to add_proxy() ASAP.'
    """
    def decorator(func):
        '''This is a decorator which can be used to mark functions
        as deprecated. It will result in a warning being emitted
        when the function is used.'''
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            message = '{} is a deprecated function. {}'.format(
                func.__name__,
                instructions)

            frame = inspect.currentframe().f_back

            warnings.warn_explicit(message,
                                   category=DeprecatedWarning,
                                   filename=inspect.getfile(frame.f_code),
                                   lineno=frame.f_lineno)

            return func(*args, **kwargs)

        return wrapper

    return decorator

def set_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    random.seed(seed)
    
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    
    if torch.backends.mps.is_available():
        torch.mps.manual_seed(seed)
    
    # torch.backends.cudnn.deterministic = True
    # torch.backends.cudnn.benchmark = False
    # torch.use_deterministic_algorithms(True)
    
def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def clean_up_adata(adata, fields_to_keep):
    current_obs_fields = adata.obs.columns.tolist()
    excess_obs_fields = [field for field in current_obs_fields if field not in fields_to_keep]
    for field in excess_obs_fields:
        del adata.obs[field]
    
    current_var_fields = adata.var.columns.tolist()
    excess_var_fields = [field for field in current_var_fields 
        if field not in []]
    for field in excess_var_fields:
        del adata.var[field]

    for field in set(adata.uns.keys()) - set(fields_to_keep):
        del adata.uns[field]


@jit
def gaussian_kernel_2d(origin, xy_array, radius, eps=0.001):
    distances = np.sqrt(np.sum((xy_array - origin)**2, axis=1))
    sigma = radius / np.sqrt(-2 * np.log(eps))
    weights = np.exp(-(distances**2) / (2 * sigma**2))
    weights[distances > radius] = 0
    # weights[0] = 0
    return weights


def min_max_df(df):
    return pd.DataFrame(
        MinMaxScaler().fit_transform(df),
        columns=df.columns,
        index=df.index
    )


def prune_neighbors(dsi, dist, maxl):
    num_samples = dsi.shape[0]

    rows = np.repeat(np.arange(num_samples), dsi.shape[1])
    cols = dsi.flatten()
    weights = dist.flatten()

    adjacency = np.zeros((num_samples, num_samples), dtype=weights.dtype)
    adjacency[rows, cols] = weights
    np.fill_diagonal(adjacency, 0) 

    for i in range(num_samples):
        row = adjacency[i]
        non_zero_indices = np.nonzero(row)[0]
        if len(non_zero_indices) > maxl:
            sorted_indices = non_zero_indices[np.argsort(row[non_zero_indices])] # indices sorted by weight
            to_remove = sorted_indices[maxl:]  # set all connections with high weight to 0
            adjacency[i, to_remove] = 0

    adjacency = np.minimum(adjacency, adjacency.T)
    bknn = csr_matrix(adjacency)
    return bknn


def lR_to_l(adata, mapper={'leiden_R': 'leiden'}):
    '''
    Map a current column name to a new column name. By default,
    maps `leiden_R` to `leiden`, typically run after using 
    `sc.tl.leiden(restrict_to=)`.
    
    `adata`: annotated data matrix
    
    returns: None, modifies in-place
    '''
    for current_col_name in mapper:
        new_col_name = mapper[current_col_name]
        current_col = adata.obs[current_col_name].copy()
        adata.obs.drop(columns=current_col_name)
        adata.obs[new_col_name] = current_col
    return

def reset_colors(adata, key='leiden', use_plt=True):
    if use_plt:
        try:
            del(adata.uns['plt']['color'][key])
        except:
            pass
    else:
        # Fall back to scanpy color storage
        try:
            del(adata.uns['%s_colors' % key])
        except:
            pass
    return

def relabel_clusts(adata, key='leiden'):
    '''
    Relabel the values in `key` as ordered categories numbering from 0 to _n_.
    
    `adata`: annotated data matrix
    `key`: name of column in `adata.obs` with the clusters
    
    returns: None, modifies in-place
    ''' 
    try:
        adata.obs[key].cat
    except AttributeError:
        adata.obs[key] = adata.obs[key].astype('category')
        
    cats = adata.obs[key].cat.categories
    new_cats = [str(i) for i in range(len(cats))]
    adata.obs[key] = adata.obs[key].map(dict(zip(cats, new_cats)))
    adata.obs[key] = adata.obs[key].astype('category')
    
    reset_colors(adata, key=key)
    
    return

def clean_leiden(adata):
    '''
    Convenience function to clean up the `leiden` column in `adata.obs`.

    `adata`: annotated data matrix
    '''
    lR_to_l(adata)
    relabel_clusts(adata)
    

    
def is_mouse_data(adata):
    """
    Determine if an AnnData object contains mouse or human data based on gene names.
    
    This function examines gene names to determine if the data is from mouse (capitalized first letter only)
    or human (all caps gene symbols). It samples a subset of genes to make the determination.
    
    Parameters
    ----------
    adata : AnnData
        The annotated data matrix to check
        
    Returns
    -------
    bool
        True if the data appears to be from mouse, False if it appears to be from human
    """
    # Get a sample of gene names to check (up to 100)
    gene_sample = adata.var_names[:1000]
    
    # Count genes that follow mouse naming convention (only first letter capitalized)
    mouse_pattern_count = sum(1 for gene in gene_sample if 
                             gene[0].isupper() and 
                             all(not c.isupper() for c in gene[1:]) and
                             len(gene) > 1)
    
    # Count genes that follow human naming convention (all uppercase)
    human_pattern_count = sum(1 for gene in gene_sample if 
                             all(c.isupper() or not c.isalpha() for c in gene) and
                             any(c.isupper() for c in gene))
    
    # Return True if more genes match mouse pattern than human pattern
    return mouse_pattern_count > human_pattern_count
