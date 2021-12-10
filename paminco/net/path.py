"""Module contaning path related methods for a network."""

import psutil
from functools import partial

import multiprocessing as mp
import numpy as np
from numpy.ctypeslib import as_ctypes_type
import scipy.sparse as sps


def csr_dijkstra(data, indices=None, **kwargs):
    csr = sps.csr_matrix(data, copy=False)
    return sps.csgraph.dijkstra(csr, indices=indices, return_predecessors=True, **kwargs)


def csr_dijkstra_mp(data, indices, num_cpus=None, chunks_per_cpu: int = 1, **kwargs):
    # TODO: to be tested on unix where fork(), does not work on windoof
    # due to child process creation
    spmat = sps.csr_matrix(data, copy=False)
    
    # Shared csr data
    coo_data_s = mp.Array(as_ctypes_type(spmat.data.dtype), len(spmat.data))
    coo_data = np.frombuffer(coo_data_s.get_obj(), dtype=spmat.data.dtype)
    coo_data[:] = spmat.data[:]
    
    coo_indices_s = mp.Array(as_ctypes_type(spmat.indices.dtype), len(spmat.indices))
    coo_indices = np.frombuffer(coo_indices_s.get_obj(), dtype=spmat.indices.dtype)
    coo_indices[:] = spmat.indices[:]
    
    coo_indptr_s = mp.Array(as_ctypes_type(spmat.indptr.dtype), len(spmat.indptr))
    coo_indptr = np.frombuffer(coo_indptr_s.get_obj(), dtype=spmat.indptr.dtype)
    coo_indptr[:] = spmat.indptr[:]
    
    # Init pool
    if num_cpus is None:
        num_cpus = psutil.cpu_count(logical=False) - 1
    pool = mp.Pool(num_cpus)
    
    # Split indices for which to compute dijekstra into chunks
    idx_chunks = np.array_split(indices.astype(int), num_cpus * chunks_per_cpu)
    
    # Add matrix to func and start computation
    func = partial(csr_dijkstra, (coo_data_s, coo_indices_s, coo_indptr_s))
    data = pool.map(func, idx_chunks)
    
    # Transform data
    # D, Pr, indices = zip(*data)
    D, Pr = zip(*data)
    D, Pr = np.vstack((D)), np.vstack((Pr))
    
    return D, Pr


def get_path_edges(
        Pr: np.ndarray,
        t: int,
        s: int,
        lookup_id: dict,
        reversed: bool = False
        ) -> np.ndarray:
    """Get edges on path from ``s`` -> ``t``.
    
    Parameters
    ----------
    Pr : ndarray
        Predecessor matrix.
    s : int
        Index of source.
    t : int
        Index of target.
    lookup_id : dict
        Mapping of (node_id, node_id) -> edge_id.
    reversed : bool, default=False
        Whether to reverse the path from ``s`` to ``t``.
    
    Returns
    -------
    ndarray
        Indices of path edges.
    """
    edges = []
    w = t
    # TODO: pass proper Pr here to not worry about indices.
    while True:
        try:
            v = Pr[s, w]
        except IndexError:
            v = Pr[w]
        if v == -9999:
            break
        edges.append(lookup_id[(v, w)])
        w = v
    edges = np.array(edges)
    
    if reversed is True:
        edges = np.flip(edges)
    return edges
