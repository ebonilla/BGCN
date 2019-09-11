# -*- coding: utf-8 -*-

"""Helper functions and classes."""
import numpy as np
import scipy.sparse as sps
import networkx as nx


from sklearn.model_selection import StratifiedShuffleSplit

def sparse_to_tuple(m):

    if not sps.isspmatrix_coo(m):
        m = m.tocoo()

    indices = np.vstack((m.row, m.col)).transpose()
    values = np.float32(m.data)
    dense_shape = m.shape

    return indices, values, dense_shape


def recursive_stratified_shuffle_split(sizes, random_state=None):
    """
    For usage examples please see:
    https://github.com/stellargraph/gcn-latent-net/blob/master/notebooks/Recursive%20Stratified%20Sampling-Karate%20Club.ipynb
    """
    head, *tail = sizes
    sss = StratifiedShuffleSplit(n_splits=1, test_size=head,
                                 random_state=random_state)

    def split(X, y):

        a_index, b_index = next(sss.split(X, y))

        yield a_index

        if tail:

            split_tail = recursive_stratified_shuffle_split(sizes=tail,
                                                            random_state=random_state)

            for ind in split_tail(X[b_index], y[b_index]):

                yield b_index[ind]

        else:

            yield b_index

    return split


def indices_to_mask(indices, size):

    mask = np.zeros(size, dtype=np.bool)
    mask[indices] = True

    return mask


def mask_values(a, mask, fill_value=0):

    a_masked = np.full_like(a, fill_value, dtype=np.int32)
    a_masked[mask] = a[mask]

    return a_masked


def load_adjacency_from_file(adjacency_matrix):
    g_ = nx.read_gpickle(adjacency_matrix)
    # A = nx.adjacency_matrix(g_).toarray()
    A = nx.adjacency_matrix(g_) # need get a SRC format

    print("Adjacency loaded from " + adjacency_matrix)
    return A



