import os.path

import numpy as np
import scipy.sparse as sps
import tensorflow as tf

import pandas as pd
import networkx as nx
import pickle as pkl

from functools import partial

from sklearn.preprocessing import LabelBinarizer

from sklearn.preprocessing import normalize

from src.base import (
    recursive_stratified_shuffle_split,
    indices_to_mask,
    mask_values,
    load_adjacency_from_file,
)

from src.utils_gcn import load_data
from sklearn.neighbors import kneighbors_graph

import scipy as sp

ADJ_FORMAT = "csr"  # the default format for a BGCN models


def load_pickle(name, ext, data_home="datasets", encoding="latin1"):

    path = os.path.join(data_home, name, "ind.{0}.{1}".format(name, ext))

    with open(path, "rb") as f:

        return pkl.load(f, encoding=encoding)


def load_test_indices(name, data_home="datasets"):

    indices_df = pd.read_csv(
        os.path.join(data_home, name, "ind.{0}.test.index".format(name)), header=None
    )
    indices = indices_df.values.squeeze()

    return indices


def load_dataset(name, data_home="datasets"):

    exts = ["tx", "ty", "allx", "ally", "graph"]

    (X_test, y_test, X_rest, y_rest, G_dict) = map(
        partial(load_pickle, name, data_home=data_home), exts
    )

    _, D = X_test.shape
    _, K = y_test.shape

    ind_test_perm = load_test_indices(name, data_home)
    ind_test = np.sort(ind_test_perm)

    num_test = len(ind_test)
    num_test_full = ind_test[-1] - ind_test[0] + 1

    # TODO: Issue warning if `num_isolated` is non-zero.
    num_isolated = num_test_full - num_test

    # normalized zero-based indices
    ind_test_norm = ind_test - np.min(ind_test)

    # features
    X_test_full = sps.lil_matrix((num_test_full, D))
    X_test_full[ind_test_norm] = X_test

    X_all = sps.vstack((X_rest, X_test_full)).toarray()
    X_all[ind_test_perm] = X_all[ind_test]

    # targets
    y_test_full = np.zeros((num_test_full, K))
    y_test_full[ind_test_norm] = y_test

    y_all = np.vstack((y_rest, y_test_full))
    y_all[ind_test_perm] = y_all[ind_test]

    # graph
    G = nx.from_dict_of_lists(G_dict)
    A = nx.to_scipy_sparse_matrix(G, format=ADJ_FORMAT)

    return X_all, y_all, A


def load_cora(data_home="datasets/legacy/cora", cites_filename="cora.cites"):

    df = pd.read_csv(
        os.path.join(data_home, "cora.content"), sep=r"\s+", header=None, index_col=0
    )
    features_df = df.iloc[:, :-1]
    labels_df = df.iloc[:, -1]

    X_all = features_df.values

    y_all = LabelBinarizer().fit_transform(labels_df.values)

    fname_ext = cites_filename.split(".")[-1]

    if fname_ext == "adjlist":
        g_ = nx.read_adjlist(os.path.join(data_home, cites_filename))
        A = nx.to_scipy_sparse_matrix(g_, nodelist=sorted(df.index), format=ADJ_FORMAT)
    elif fname_ext == "gpickle":
        g_ = nx.read_gpickle(os.path.join(data_home, cites_filename))
        A = nx.to_scipy_sparse_matrix(g_, format=ADJ_FORMAT)
    else:
        edge_list_df = pd.read_csv(
            os.path.join(data_home, cites_filename), sep=r"\s+", header=None
        )

        idx_map = {j: i for i, j in enumerate(df.index)}

        H = nx.from_pandas_edgelist(edge_list_df, 0, 1)
        G = nx.relabel.relabel_nodes(H, idx_map)

        A = nx.to_scipy_sparse_matrix(G, nodelist=sorted(G.nodes()), format=ADJ_FORMAT)

    return (X_all, y_all, A)


def load_citeseer(data_home="datasets/legacy/citeseer"):

    df = pd.read_csv(
        os.path.join(data_home, "citeseer.content"),
        sep=r"\s+",
        header=None,
        index_col=0,
    )
    df.index = df.index.map(str)

    features_df = df.iloc[:, :-1]
    labels_df = df.iloc[:, -1]

    X_all = features_df.values

    y_all = LabelBinarizer().fit_transform(labels_df.values)

    edge_list_df = pd.read_csv(
        os.path.join(data_home, "citeseer.cites"), sep=r"\s+", dtype=str, header=None
    )

    idx_map = {j: i for i, j in enumerate(df.index)}

    H = nx.from_pandas_edgelist(edge_list_df, 0, 1)
    G = nx.relabel.relabel_nodes(H, idx_map)

    # This dataset has about 15 nodes in the edge list that don't have corresponding entries
    # in citeseer.content, that is don't have features. We need to identify them and then remove
    # them from the graph along with all the edges to/from them.
    nodes_to_remove = [n for n in G.nodes() if type(n) == str]
    G.remove_nodes_from(nodes_to_remove)

    A = nx.to_scipy_sparse_matrix(G, nodelist=sorted(G.nodes()), format=ADJ_FORMAT)

    return (X_all, y_all, A)


def load_twitter(data_home="datasets/twitter"):

    # Load the node features. The index is node IDs
    df = pd.read_csv(
        os.path.join(data_home, "twitter_node_features.csv"), header=0, index_col=0
    )

    print("Loaded node features {}".format(df.shape))
    # Extract the target values

    df_y = pd.DataFrame({"hate": df["hate"], "not-hate": 1 - df["hate"]})

    y_all = df_y.values
    # y_all = LabelBinarizer().fit_transform(y_all)
    # Drop the column with the target values
    df.drop(columns=["hate"], inplace=True)

    X_all = df.values

    print("y_all shape {}".format(y_all.shape))
    print("X_all shape {}".format(X_all.shape))

    # Load the graph
    print("Loading graph")
    g_nx = nx.read_adjlist(os.path.join(data_home, "twitter.edges"), nodetype=int)
    print("...Done")
    print(
        "Graph num nodes {} and edges {}".format(
            g_nx.number_of_nodes(), g_nx.number_of_edges()
        )
    )
    # Extract the adjacency matrix.
    # A should be scipy sparse matrix
    print("Extracting A from networkx graph")
    A = nx.adjacency_matrix(g_nx, nodelist=df.index, weight=None)
    # A = nx.adjacency_matrix(g_nx)
    print("...Done")
    return (X_all, y_all, A)


DATASET_LOADERS = dict(
    cora=partial(load_dataset, "cora"),
    citeseer=partial(load_dataset, "citeseer"),
    pubmed=partial(load_dataset, "pubmed"),
    twitter=load_twitter,
)


def load_split(name, val_size=500, data_home="datasets"):

    y = load_pickle(name, ext="y", data_home=data_home)

    train_size = len(y)

    ind_train = range(train_size)
    ind_val = range(train_size, train_size + val_size)

    ind_test_perm = load_test_indices(name, data_home)
    ind_test = np.sort(ind_test_perm)

    return ind_train, ind_val, ind_test


def balanced_data_split(X, y, samples_per_class, random_state):
    indices = []

    num_classes = y.shape[1]
    train_indices = []

    for c in range(num_classes):
        ind = np.where(y[:, c] > 0.5)[0]
        #train_indices.extend(list(np.random.choice(ind, samples_per_class, replace=False)))
        train_indices.extend(list(random_state.choice(ind, samples_per_class, replace=False)))
        #print(ind)

    indices.append(train_indices)

    all_indices = set(range(y.shape[0]))
    # remove the indices of the datasets in the training set.
    unused_indices = all_indices - set(train_indices)

    # Now split the remaining datasets into validation and test sets using
    # uniform sampling such that validation datasets are 25% of the remaining datasets
    # and test datasets is 75% of the remaining datasets.
    val_size = int(len(unused_indices)*0.30)
    #val_indices = list(np.random.choice(list(unused_indices), val_size, replace=False))
    val_indices = list(random_state.choice(list(unused_indices), val_size, replace=False))

    test_indices = list(unused_indices-set(val_indices))

    # some sanity checks
    assert (len(train_indices)+len(val_indices)+len(test_indices)) == y.shape[0]
    assert len(set(train_indices).intersection(set(val_indices))) == 0
    assert len(set(val_indices).intersection(set(test_indices))) == 0

    indices.append(val_indices)
    indices.append(test_indices)


    return indices


def add_val_to_train(mask_train, mask_val, seed_val, p):
    """
    Add a percentage of the validation set to the training set
    :param mask_train:
    :param mask_val:
    :param seed_val:
    :param p: Probability of a point in validation to be addded in training
    :return:
    """
    print("Adding some validation datasets to training")
    rnd_val = np.random.RandomState(seed_val)
    chs = rnd_val.choice([True, False], size=np.sum(mask_val), p=[p, 1.0 - p])
    mask_val_new = np.array(mask_val)
    mask_train_new = np.array(mask_train)
    mask_val_new[mask_val_new] = chs
    mask_train_new[mask_val] = ~chs
    return mask_train_new, mask_val_new


def get_data(dataset_name, random_split, split_sizes, random_split_seed,
             add_val=True, add_val_seed=1, p_val=0.5,
             adjacency_filename=None,
             use_knn_graph=False,
             knn_metric=None,
             knn_k=None,
             balanced_split=False,
             samples_per_class=20):

    A, X, y_train, y_val, y_test, mask_train, mask_val, mask_test, y = load_data(dataset_name, "datasets")

    if use_knn_graph:
        tf.logging.info("Using KNN graph")
        A = kneighbors_graph(X, knn_k, metric=knn_metric)

        # consistent with our implementation of only considering the lower triangular
        A = sp.sparse.tril(A, k=-1)
        A = A + np.transpose(A)

    if adjacency_filename:
        A = load_adjacency_from_file(adjacency_filename)

    n, d = X.shape
    _, k = y.shape

    tf.logging.info("Dataset has {} samples, dimensionality {}".format(n, d))
    tf.logging.info("Targets belong to {} classes".format(k))

    if random_split:
        print("Using a random split")
        random_state = np.random.RandomState(random_split_seed)
        split = recursive_stratified_shuffle_split(
            sizes=split_sizes, random_state=random_state
        )
        indices = list(split(X, y))
    elif balanced_split:
        indices = balanced_data_split(X, y, samples_per_class, random_state=random_split_seed)
    else:  # fixed plit
        indices = load_split(dataset_name)

    tf.logging.info(
        "Split resulted in "
        "{} training, "
        "{} validation, "
        "{} test samples.".format(*map(len, indices))
    )

    [mask_train, mask_val, mask_test] = masks = list(
        map(partial(indices_to_mask, size=n), indices)
    )

    y_train, y_val, y_test = map(partial(mask_values, y), masks)

    # A = A.toarray()

    if (add_val):
        mask_train, mask_val = add_val_to_train(mask_train, mask_val, add_val_seed, p_val)
        masks = [mask_train, mask_val, mask_test]
        y_train, y_val, y_test = map(partial(mask_values, y), masks)


    print("**********************************************************************************************")
    print("train size: {} val size: {} test size: {}".format(np.sum(mask_train), np.sum(mask_val), np.sum(mask_test)))
    print("**********************************************************************************************")

    return X, y, A, mask_train, mask_val, mask_test, y_train, y_val, y_test



# def get_data_incompatible(dataset_name, random_split, split_sizes, random_split_seed,
#              add_val=True, add_val_seed=1, p_val=0.5,
#              adjacency_filename=None,
#              balanced_split=False, samples_per_class=20):
#
#     tf.logging.info("Loading '{}' dataset...".format(dataset_name))
#     loader = DATASET_LOADERS[dataset_name]
#     X, y, A = loader()
#
#     if adjacency_filename:
#         A = load_adjacency_from_file(adjacency_filename)
#
#     X = normalize(X, norm="l1", axis=1)
#
#     n, d = X.shape
#     _, k = y.shape
#
#     tf.logging.info("Dataset has {} samples, dimensionality {}".format(n, d))
#     tf.logging.info("Targets belong to {} classes".format(k))
#
#     if random_split:
#         random_state = np.random.RandomState(random_split_seed)
#         split = recursive_stratified_shuffle_split(
#             sizes=split_sizes, random_state=random_state
#         )
#         indices = list(split(X, y))
#     elif balanced_split:
#         indices = balanced_data_split(X, y, samples_per_class, random_state=random_split_seed)
#     else: # fixed plit
#         indices = load_split(dataset_name)
#
#     tf.logging.info(
#         "Split resulted in "
#         "{} training, "
#         "{} validation, "
#         "{} test samples.".format(*map(len, indices))
#     )
#
#     [mask_train, mask_val, mask_test] = masks = list(
#         map(partial(indices_to_mask, size=n), indices)
#     )
#
#     y_train, y_val, y_test = map(partial(mask_values, y), masks)
#
#     # A = A.toarray()
#
#     if (add_val):
#         mask_train, mask_val = add_val_to_train(mask_train, mask_val, add_val_seed, p_val)




    print("**********************************************************************************************")
    print("train size: {} val size: {} test size: {}".format(np.sum(mask_train), np.sum(mask_val), np.sum(mask_test)))
    print("**********************************************************************************************")

    return X, y, A, mask_train, mask_val, mask_test, y_train, y_val, y_test
