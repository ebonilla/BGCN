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
    "--add-val/--no-add-val",
    default=False,
    help="Add 50% of validation for training (true) or not",
)
@click.option(
    "--add-val-seed",
    type=click.INT,
    help="Seed for including validation datasets in training [integer]",
    default=1,
)
@click.option(
    "--results-dir",
    type=click.STRING,
    help="name of results directory [string]",
    default="./results",
)
def main(dataset,
         epochs,
         adjacency,
         random_seed_np,
         random_seed_tf,
         random_split,
         split_sizes,
         random_split_seed,
         add_val,
         add_val_seed,
         results_dir):
    """
    Run BGCN
    :param dataset:
    :param epochs:
    :param adjacency:
    :param random_seed_np:
    :param random_seed_tf:
    :param random_split:
    :param split_sizes:
    :param random_split_seed:
    :param add_val:
    :param add_val_seed:
    :param results_dir:
    :return:
    """

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
                                                                                              add_val=add_val,
                                                                                              add_val_seed=add_val_seed,
                                                                                              p_val=0.5,
                                                                                              adjacency_filename=adjacency)


    # ==================================Train Model===========================================

    GNN_Model = GnnModel(FLAGS, features, labels, adj, y_train, y_val, y_test, mask_train, mask_val, mask_test,
                         checkpt_name='model_1', model_name="BGCN", results_dir=results_dir)
    GNN_Model.model_initialization()
    GNN_Model.train()


if __name__ == "__main__":
    exit(main())  # pragma: no cover
