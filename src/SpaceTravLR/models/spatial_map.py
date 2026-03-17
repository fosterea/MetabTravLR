from numba import jit, prange
from tqdm import tqdm
import numpy as np
from scipy.ndimage import gaussian_filter

from ..tools.utils import deprecated

@jit
def generate_grid_centers(m, n, xmin, xmax, ymin, ymax):
    centers = []
    cell_width = (xmax - xmin) / n
    cell_height = (ymax - ymin) / m
    
    for i in range(m):
        for j in range(n):
            x = xmin + (j + 0.5) * cell_width
            y = ymax - (i + 0.5) * cell_height
            centers.append((x, y))    
    return centers

@jit
def distance(point1, point2):
    x1, y1 = point1
    x2, y2 = point2
    return np.sqrt((x2 - x1)**2 + (y2 - y1)**2)

# @deprecated('`xyc2spatial` is deprecated. Use `xyc2spatial_fast` instead to save the trees 🌴️')
def xyc2spatial(x, y, c, m, n, split_channels=True, disable_tqdm=True):
    
    assert len(x) == len(y) == len(c)
    xmin, xmax, ymin, ymax = np.min(x), np.max(x), np.min(y), np.max(y)
    xyc = np.column_stack([x, y, c]).astype(float)
    
    centers = generate_grid_centers(m, n, xmin, xmax, ymin, ymax)
    clusters = np.unique(c).astype(int)
    
    spatial_maps = np.zeros((len(x), m, n))
    mask = np.zeros((len(clusters), m, n))

    # mask = np.ones((len(clusters), m, n)) * -1*np.inf


    with tqdm(total=len(xyc), disable=disable_tqdm, desc=f'🌍️ Generating {m}x{n} spatial maps') as pbar:
        
        for s, coord in enumerate(xyc):
            x_, y_, cluster = coord
            
            dist_map = np.array([np.float32(distance((x_, y_), c)) for c in centers]).reshape(m, n).astype(np.float32)
            
            nearest_center_idx = np.argmin(dist_map)
            u, v = np.unravel_index(nearest_center_idx, (m, n))
            mask[int(cluster)][u, v] = 1

            spatial_maps[s] = dist_map
            
            pbar.update()
    
    
    spatial_maps = np.repeat(np.expand_dims(spatial_maps, axis=1), len(clusters), axis=1)
    mask = np.repeat(np.expand_dims(mask, axis=0), spatial_maps.shape[0], axis=0)


    # max_vals = np.max(spatial_maps, axis=(2, 3), keepdims=True)
    # channel_wise_maps = max_vals/spatial_maps*mask 
    channel_wise_maps = spatial_maps*mask 


    # channel_wise_maps = 1.0/channel_wise_maps

    # mean = np.mean(channel_wise_maps, axis=(2, 3), keepdims=True)
    # std = np.std(channel_wise_maps, axis=(2, 3), keepdims=True)
    # epsilon = 1e-8 
    # channel_wise_maps_norm = (channel_wise_maps - mean) / (std + epsilon)


    min_vals = np.min(channel_wise_maps, axis=(2, 3), keepdims=True)
    max_vals = np.max(channel_wise_maps, axis=(2, 3), keepdims=True)
    denominator = np.maximum(max_vals - min_vals, 1e-15)
    channel_wise_maps_norm = (channel_wise_maps - min_vals) / denominator

    # channel_wise_maps = (1+(channel_wise_maps_norm*-1)) * mask


    # channel_wise_maps = channel_wise_maps_norm


    # channel_wise_maps = channel_wise_maps_norm
    # channel_wise_maps = 1.0/channel_wise_maps

    # channel_wise_maps = np.where(mask!=0, channel_wise_maps_norm, channel_wise_maps_norm.max())

    # channel_wise_maps = 1.0/(channel_wise_maps+1)

    # channel_wise_maps = (1.0/spatial_maps)*mask
    # channel_wise_maps = (spatial_maps.max()/spatial_maps)*mask  
    # channel_wise_maps = gaussian_filter(channel_wise_maps, sigma=0.5)
        
    assert channel_wise_maps.shape == (len(x), len(clusters), m, n)
    
    if split_channels:
        return channel_wise_maps
    else:
        return channel_wise_maps.sum(axis=1)
    
    
@jit(nopython=True, parallel=True)
def xyc2spatial_fast(xyc, m, n):
    """
    Converts spatial coordinates (x, y) and cluster labels (c) to a spatial \
        distance map with grid sizes m x n. 
    Each channels encodes the distance map for a unique cluster.

    Return (n_samples, n_clusters, m, n)
    """

    # print(f'🌍️ Generating spatial {m}x{n} maps...*')

    x, y, c = xyc[:, 0], xyc[:, 1], xyc[:, 2]
    xmin, xmax, ymin, ymax = np.min(x), np.max(x), np.min(y), np.max(y)
    
    centers = generate_grid_centers(m, n, xmin, xmax, ymin, ymax)
    clusters = np.unique(c).astype(np.int32)
    num_clusters = len(clusters)
    
    spatial_maps = np.zeros((len(xyc), num_clusters, m, n), dtype=np.float32)
    # mask = np.zeros((num_clusters, m, n), dtype=np.float32)
    mask = np.ones((num_clusters, m, n), dtype=np.float32)

    
    for s in prange(len(xyc)):
        x_, y_, cluster = xyc[s]
        dist_map = np.array([distance((x_, y_), c) for c in centers]).reshape(m, n)
        
        nearest_center_idx = np.argmin(dist_map)
        u, v = nearest_center_idx // n, nearest_center_idx % n
        mask[int(cluster), u, v] = 1
        
        for i in range(num_clusters):
            spatial_maps[s, i] = dist_map
    max_val = np.max(spatial_maps)
    channel_wise_maps = np.zeros_like(spatial_maps)
    
    for s in prange(len(xyc)):
        for i in range(num_clusters):
            for j in range(m):
                for k in range(n):
                    # channel_wise_maps[s, i, j, k] = (max_val / spatial_maps[s, i, j, k]) * mask[i, j, k]
                    channel_wise_maps[s, i, j, k] = spatial_maps[s, i, j, k] * mask[i, j, k]



    min_vals = np.zeros((len(xyc), num_clusters, 1, 1), dtype=np.float32)
    max_vals = np.zeros((len(xyc), num_clusters, 1, 1), dtype=np.float32)
    for s in prange(len(xyc)):
        for i in range(num_clusters):
            min_vals[s, i, 0, 0] = np.min(channel_wise_maps[s, i])
            max_vals[s, i, 0, 0] = np.max(channel_wise_maps[s, i])
    
    denominator = np.maximum(max_vals - min_vals, 1e-15)
    channel_wise_maps_norm = np.zeros_like(channel_wise_maps)
    for s in prange(len(xyc)):
        for i in range(num_clusters):
            for j in range(m):
                for k in range(n):
                    channel_wise_maps_norm[s, i, j, k] = (channel_wise_maps[s, i, j, k] - min_vals[s, i, 0, 0]) / denominator[s, i, 0, 0]


    # channel_wise_maps = 1.0/channel_wise_maps
    return channel_wise_maps_norm