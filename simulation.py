# Databricks notebook source
import os
import numpy as np
import pandas as pd
from scipy.stats import norm
import torch
import copy
from torch import nn, optim
from torch.nn import functional as F
from torch.distributions import Normal 
from torch.utils.data import DataLoader, Dataset
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt

from functools import partial
from src.GEEN import *
from src.simulationDataGen import get_simulation_data

# COMMAND ----------

torch.set_default_dtype(torch.float64)

# COMMAND ----------

from types import SimpleNamespace
init_config = {
               'device': 'cuda',
               'activation':torch.nn.ReLU(),
               'clip':0.,
               'drop_out':0.,
               'epochs':30,
               'N_sample':10000,
               'patience':3,
               'batch_size': 64,
               'input_size': 4,
               'sample_size': 500,
               'num_layer': 6,
               'hidden_size': 10,
               'learning_rate': 0.05,
               'number_simulation':25,
               'dir': '/dbfs/mnt/gti/GEEN' #TODO
               }
init_config = SimpleNamespace(**init_config)

# COMMAND ----------

try:
    with_xstar = dbutils.widgets.get("with_xstar") == 'True'
except:
    with_xstar = True

try:
    discrete_xstar = dbutils.widgets.get("discrete_xstar") == 'True'
except:
    discrete_xstar = False

try:
    x4_distinct = dbutils.widgets.get("x4_distinct") == 'True'
except:
    x4_distinct = True

try:
    loss_fun = dbutils.widgets.get("loss_fun")
except:
    loss_fun = 'kl_loss_from_avg'

try:
    is_normal = dbutils.widgets.get("is_normal") == 'True'
except:
    is_normal = True

try:
    batch_norm = dbutils.widgets.get("batch_norm") == 'True'
except:
    batch_norm = True

try:
    window_size_base = float(dbutils.widgets.get("window_size_base"))
except:
    window_size_base = 1.1247613400258931
try:
    lambda_base = float(dbutils.widgets.get("lambda_base"))
except:
    lambda_base = 0.9491195387601212

try:
    window_size_double = float(dbutils.widgets.get("window_size_double"))
except:
    window_size_double = 1.2694011671387606
try:
    lambda_double = float(dbutils.widgets.get("lambda_double"))
except:
    lambda_double = 0.3243454656783943

try:
    window_size_linear = float(dbutils.widgets.get("window_size_linear"))
except:
    window_size_linear = 2.2053551506747415
try:
    lambda_linear = float(dbutils.widgets.get("lambda_linear"))
except:
    lambda_linear = 0.03518811110691017

# COMMAND ----------

simulations = {
    "Basement": {
        "with_xstar": with_xstar,
        "x4_distinct": x4_distinct,
        "discrete_xstar": discrete_xstar,
        'loss_fun': loss_fun,
        "is_normal": is_normal,
        "batch_norm": batch_norm,
        "error_scale": 1,
        "linear_error": False,
        "window_size":window_size_base,
        "lambda":lambda_base,
    },
    "doubleError": {
        "with_xstar": with_xstar,
        "x4_distinct": x4_distinct,
        "discrete_xstar": discrete_xstar,
        "loss_fun": loss_fun,
        "is_normal": is_normal,
        "batch_norm": batch_norm,
        "error_scale": 2,
        "linear_error": False,
        "window_size": window_size_double,
        "lambda":lambda_double,
    },
    "linearError": {
        "with_xstar": with_xstar,
        "x4_distinct": x4_distinct,
        "discrete_xstar": discrete_xstar,
        "loss_fun": loss_fun,
        "is_normal": is_normal,
        "batch_norm": batch_norm,
        "error_scale": 1,
        "linear_error": True,
        "window_size":window_size_linear,
        "lambda":lambda_linear,
    },
}

# COMMAND ----------

def simulation(config):
    (
        train_dataloader,
        val_dataloader,
        test_dataloader,
        x_train,
        x_star_train,
        x_val,
        x_star_val,
        x_test,
        x_star_test,
    ) = get_simulation_data(config)

    genModel = LitGenDNN(config).double()
    genModel = genModel.to(config.device)

    return (
        x_train,
        x_star_train,
        x_gen_train,
        x_val,
        x_star_val,
        x_gen_val,
        x_test,
        x_star_test,
        x_gen_test,
        test_result[0]["test_divergence_loss"],
        test_result[0]["test_loss"],
        test_result[0]["test_normalization"],
    )

# COMMAND ----------

for k, v in simulations.items():
    simulation_name = k
    if v['discrete_xstar']:
        simulation_name = simulation_name + '-discrete_xstar'
    if v['x4_distinct']:
        simulation_name = simulation_name + '-x4_distinct'
    config = copy.deepcopy(init_config)
    config = config | v

    test_mse_list = []
    test_corr_list = []
    min_loss_res = []
    min_loss = np.inf
    min_mse = np.inf
    max_corr = -np.inf
    x = None
    x_star = None
    x_star_gen = None

    for i in range(config["number_simulation"]):
        (
            x_train,
            x_star_train,
            x_gen_train,
            x_val,
            x_star_val,
            x_gen_val,
            x_test,
            x_star_test,
            x_gen_test,
            test_divergence_loss,
            test_loss,
            test_normalization,
        ) = simulation(config)

        mse = mean_squared_error(x_star_test, x_gen_test, squared=False)
        corr = np.corrcoef(x_star_test, x_gen_test)[0, 1]

        test_mse_list.append(mse)
        test_corr_list.append(corr)

        #min_mse = min(min_mse, mse)
        max_corr = max(max_corr, corr)
        min_loss = min(min_loss, test_loss)

        if min_mse > mse:
            min_mse = mse

            x = np.concatenate([x_train, x_val, x_test], axis=0)
            x_star = np.concatenate([x_star_train, x_star_val, x_star_test], axis=0)
            x_star_gen = np.concatenate([x_gen_train, x_gen_val, x_gen_test], axis=0)

    record = {}
    record["mean_mse"] = np.mean(test_mse_list)
    record["median_mse"] = np.median(test_mse_list)
    record["min_mse"] = min(test_mse_list)
    record["max_mse"] = max(test_mse_list)
    record["std_mse"] = np.std(test_mse_list)

    record["mean_corr"] = np.mean(test_corr_list)
    record["median_corr"] = np.median(test_corr_list)
    record["min_corr"] = min(test_corr_list)
    record["max_corr"] = max(test_corr_list)
    record["std_corr"] = np.std(test_corr_list)

    df = pd.DataFrame.from_records([record])
    df.to_csv(
        os.path.join(
            config["dir"], "results", "%s_simulation_res.csv" % simulation_name
        ),
        index=False,
    )

    df_cor = pd.DataFrame(data=test_corr_list, columns=["corr"])
    df_cor.to_csv(
        os.path.join(
            config["dir"], "results", "%s_simulation_corr.csv" % simulation_name
        ),
        index=False,
    )

    df_data = pd.DataFrame(
        data=np.concatenate(
            [
                x,
                x_star.reshape(-1, 1),
                x_star_gen.reshape(-1, 1),
            ],
            axis=1,
        ),
        columns=["x1", "x2", "x3", "x4", "x_star", "x_star_gen"],
    )
    df_data.to_csv(
        os.path.join(
            config["dir"], "results", "%s_simulation_res_data.csv" % simulation_name
        ),
        index=False,
    )
