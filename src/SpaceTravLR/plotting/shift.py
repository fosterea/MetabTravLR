from functools import cache
import numpy as np 
from scipy.stats import pearsonr
from sklearn.neighbors import NearestNeighbors

from tqdm import tqdm
from pqdm.processes import pqdm
from velocyto.estimation import colDeltaCorpartial, colDeltaCor


def compute_probability(i, d_i, gene_mtx, indices, n_cells, T):
    exp_corr_sum = 0
    row_probs = np.zeros(n_cells)
    
    for j in indices[i]:
        r_ij = gene_mtx[i] - gene_mtx[j]
        corr, _ = pearsonr(r_ij, d_i)
        if np.isnan(corr):
            corr = 1
        exp_corr = np.exp(corr / T)
        exp_corr_sum += exp_corr
        row_probs[j] = exp_corr

    if exp_corr_sum != 0:
        row_probs /= exp_corr_sum

    return np.array(row_probs)

## CellOracle uses adapted Velocyto code
## This function is coded exactly as described in CellOracle paper
def estimate_transition_probabilities(adata, delta_X, embedding=None, n_neighbors=200, 
random_neighbors=False, annot=None, T=0.05, n_jobs=1):

    n_cells, n_genes = adata.shape
    delta_X = np.array(delta_X)
    gene_mtx = adata.layers['imputed_count'].astype(np.float64)
    delta_X = delta_X.astype(np.float64)

    if n_neighbors is None:

        P = np.ones((n_cells, n_cells))
        corr = colDeltaCor(
            np.ascontiguousarray(gene_mtx.T), 
            np.ascontiguousarray(delta_X.T), 
            threads=n_jobs
            )
    
    else:

        P = np.zeros((n_cells, n_cells))

        n_neighbors = min(n_cells, n_neighbors)
        
        if random_neighbors == 'even':

            cts = np.unique(adata.obs[annot])
            ct_dict = {ct: np.where(adata.obs[annot] == ct)[0] for ct in cts}
            cells_per_ct = round(n_neighbors / len(cts))

            indices = []

            for i in range(n_cells):
                i_indices = []

                for ct, ct_cells in ct_dict.items():

                    sample = np.random.choice(ct_cells[ct_cells != i], size=cells_per_ct, replace=False)
                    i_indices.extend(sample)

                i_indices = np.array(i_indices)
                P[i, i_indices] = 1
                indices.append(i_indices)

            indices = np.array(indices)

        elif random_neighbors:
            
            indices = []
            cells = np.arange(n_cells)
            for i in range(n_cells):
                i_indices = np.random.choice(np.delete(cells, i), size=n_neighbors, replace=False)
                P[i, i_indices] = 1
                indices.append(i_indices)
            
            indices = np.array(indices)

        else: 

            nn = NearestNeighbors(n_neighbors=n_neighbors, n_jobs=n_jobs)
            nn.fit(embedding)
            _, indices = nn.kneighbors(embedding)

            rows = np.repeat(np.arange(n_cells), n_neighbors)
            cols = indices.flatten()
            P[rows, cols] = 1
        
        gene_mtx = gene_mtx.astype('float64')
        corr = colDeltaCorpartial(
            np.ascontiguousarray(gene_mtx.T), 
            np.ascontiguousarray(delta_X.T), 
            indices, threads=n_jobs
        )

    corr = np.nan_to_num(corr, nan=1)

    np.fill_diagonal(P, 0)
    P *= np.exp(corr / T)   
    P /= P.sum(1)[:, None]

    # args = [[i, delta_X[i], gene_mtx, indices, n_cells, T] for i in range(n_cells)]
    # results = pqdm(
    #     args,
    #     compute_probability,
    #     n_jobs=n_jobs,
    #     argument_type='args',
    #     tqdm_class=tqdm,
    #     desc='Estimating cell transition probabilities',
    # )

    # for i, row_probs in enumerate(results):
    #     P[i] = np.array(row_probs)

    return P


def project_probabilities(P, embedding, normalize=True):
    if normalize: 
        embed_dim = embedding.shape[1]

        embedding_T = embedding.T # shape (m, n_cells)
        unitary_vectors = embedding_T[:, None, :] - embedding_T[:, :, None]  # shape (m, n_cells, n_cells)
        unitary_vectors = unitary_vectors.astype(np.float64)
        
        # Normalize the difference vectors (L2 norm)
        with np.errstate(divide='ignore', invalid='ignore'):
            norms = np.linalg.norm(unitary_vectors, ord=2, axis=0)  # shape (n_cells, n_cells)
            unitary_vectors /= norms
            for m in range(embed_dim):
                np.fill_diagonal(unitary_vectors[m, ...], 0)   
        
        delta_embedding = (P * unitary_vectors).sum(2)  # shape (m, n_cells)
        delta_embedding = delta_embedding.T
        
        return delta_embedding
    
    else:
        embed_diffs = embedding[np.newaxis, :, :] - embedding[:, np.newaxis, :]
        
        # masked = embed_diffs * P[:, :, np.newaxis]
        # V_simulated = np.sum(masked, axis=1)
        V_simulated = np.einsum('ij,ijk->ik', P, embed_diffs)
        
        return V_simulated
    

