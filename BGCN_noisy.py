'''
Copyright (C) 2019. Huawei Technologies Co., Ltd and McGill University. All rights reserved.
This program is free software; you can redistribute it and/or modify
it under the terms of the MIT License.
This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
MIT License for more details.
'''

# Modified by EVB to handle noisy datasets and other splits

import tensorflow as tf
import numpy as np
from src.GNN_models import GnnModel
import os
from src.flags import flags
import click
from src.datasets import get_data
import csv

import src.config as cfg

code_path = os.path.abspath('')
FLAGS = flags()


def save_parameters(params, results_dir):
    if results_dir is not None:
        if not os.path.exists(os.path.expanduser(results_dir)):
            print("Results dir does not exist.")
            print("Creating results dir at {}".format(os.path.expanduser(results_dir)))
            os.makedirs(os.path.expanduser(results_dir))
            print(
                "Created results directory: {}".format(os.path.expanduser(results_dir))
            )
        else:
            print("Results directory already exists.")

        # write parameters file
        params_filename = os.path.join(os.path.expanduser(results_dir),  "params.csv")
        try:
            with open(params_filename, "w", buffering=1) as fh_params:
                w = csv.DictWriter(fh_params, params.keys())
                w.writeheader()
                w.writerow(params)

        except IOError:
            print("Could not open results file {}".format(params_filename))
    return


@click.command()
@click.argument("name")
@click.option(
    "--dataset",
    default="cora",
    type=click.STRING,
    help="datasets set name [cora|citeseer|pubmed].",
)
@click.option(
    "--epochs",
    default=FLAGS.epochs,
    type=click.INT,
    help="Number of epochs [int].",
)
@click.option(
    "--adjacency",
    type=click.STRING,
    help="name of adjacency matrix file [string]",
)
@click.option(
    "--random-seed-np",
    type=click.INT,
    help="global numpy random seed [integer]",
)
@click.option(
    "--random-seed-tf",
    type=click.INT,
    help="global tensorflow random seed [integer]",
)
@click.option(
    "--random-split/--fixed-split",
    default=False,
    help="Use random split (true) or fixed split (false)",
)
@click.option(
    "--split-sizes",
    default=[0.9, 0.75],
    nargs=2,
    type=click.FLOAT,
    help="size of random splits",
)
@click.option(
    "--random-split-seed",
    type=click.INT,
    help="random split seed [integer]",
)

@click.option(
    "--use-half-val-to-train/--no-use-half-val-to-train",
    default=cfg.USE_HALF_VAL_TO_TRAIN,
    help="Use half the validation data in training.",
)
@click.option("-s", "--seed", default=cfg.SEED, type=click.INT, help="Random seed")
@click.option(
    "-s", "--seed-np", default=cfg.SEED_NP, type=click.INT, help="Random seed for numpy"
)
@click.option(
    "--seed-val",
    default=cfg.SEED_VAL,
    type=click.INT,
    help="Random seed for splitting the validation set",
)

@click.option(
    "--use-knn-graph/--no-use-knn-graph",
    default=False,
    help="Use half the validation data in training.",
)
@click.option(
    "--knn-metric",
    default=cfg.DEFAULT_KNN_METRIC,
    type=click.Choice(cfg.KNN_METRICS),
    help="Default knn metric to use for building prior"
)
@click.option(
    "--knn-k",
    default=cfg.DEFAULT_KNN_K,
    type=click.INT,
    help="Default number of neighbours for KNN prior.",
)
@click.option(
    "--results-dir",
    type=click.STRING,
    help="name of results directory [string]",
    default="./results",
)
def main(name,
         dataset,
         epochs,
         adjacency,
         random_seed_np,
         random_seed_tf,
         random_split,
         split_sizes,
         random_split_seed,
         use_half_val_to_train,
         seed,
         seed_np,
         seed_val,
         use_knn_graph,
         knn_metric,
         knn_k,
         results_dir):
    """

    :param dataset:
    :param epochs:
    :param adjacency:
    :param random_seed_np:
    :param random_seed_tf:
    :param random_split:
    :param split_sizes:
    :param random_split_seed:
    :param use_half_val_to_train:
    :param seed_val:
    :param use_knn_graph:
    :param knn_metric:
    :param knn_k:
    :param results_dir:
    :return:
    """
    tf.random.set_random_seed(seed)
    random_state = np.random.RandomState(seed_np)

    FLAGS.epochs = epochs
    params = click.get_current_context().params
    print(params)
    save_parameters(params, results_dir)

    tf.set_random_seed(random_seed_tf)
    np.random.seed(random_seed_np)

    features, labels, adj, mask_train, mask_val, mask_test, y_train, y_val, y_test = get_data(dataset_name=dataset,
                                                                                              random_split=random_split,
                                                                                              split_sizes=split_sizes,
                                                                                              random_split_seed=random_split_seed,
                                                                                              add_val=use_half_val_to_train,
                                                                                              add_val_seed=seed_val,
                                                                                              p_val=0.5,
                                                                                              adjacency_filename=adjacency,
                                                                                              use_knn_graph=use_knn_graph,
                                                                                              knn_metric=knn_metric,
                                                                                              knn_k=knn_k)


    # ==================================Train Model===========================================

    GNN_Model = GnnModel(FLAGS, features, labels, adj, y_train, y_val, y_test, mask_train, mask_val, mask_test,
                         checkpt_name='model_1', model_name="BGCN", results_dir=results_dir)
    GNN_Model.model_initialization()
    GNN_Model.train()


if __name__ == "__main__":
    exit(main())  # pragma: no cover
