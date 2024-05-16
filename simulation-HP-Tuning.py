# Databricks notebook source
import os
import sys
import numpy as np
from scipy.stats import norm
import pandas as pd
import torch
import copy
from torch import nn, optim
from torch.nn import functional as F
from torch.distributions import Normal 
from torch.utils.data import DataLoader, Dataset
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt

import mlflow
from hyperopt import fmin, rand, tpe, hp, SparkTrials, Trials, STATUS_OK, space_eval
from functools import partial

from src.GEEN import *
from src.simulationDataGen import get_simulation_data

# COMMAND ----------

torch.set_default_dtype(torch.float64)

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
    error_scale = int(dbutils.widgets.get("error_scale"))
except:
    error_scale = 1

try:
    linear_error = dbutils.widgets.get("linear_error") == 'True'
except:
    linear_error = False


# COMMAND ----------

experiment_name = "/Users/yliu10@imf.org/GEEN/GEEN-HP-Tuning"
if discrete_xstar:
    experiment_name = experiment_name + '-discrete_xstar'
if x4_distinct:
    experiment_name = experiment_name + '-x4_distinct'
if is_normal:
    experiment_name = experiment_name + '-is_normal'
if batch_norm:
    experiment_name = experiment_name + '-batch_norm'
if error_scale > 1:
    experiment_name = experiment_name + '-doubleError'
if linear_error:
    experiment_name = experiment_name + '-linear_error'
mlflow.set_experiment(experiment_name) 

# COMMAND ----------

from types import SimpleNamespace
max_evals = 128
init_config = {
               'with_xstar': with_xstar,
               'discrete_xstar': discrete_xstar,
               'x4_distinct': x4_distinct,
               'loss_fun': loss_fun,
               'is_normal': is_normal,
               'batch_norm': batch_norm,
               'error_scale': error_scale,
               'linear_error': linear_error,
               'device': 'cuda',
               'activation':torch.nn.ReLU(),
               'clip':0.,
               'drop_out':0.,
               'epochs':30,
               'N_sample':10000,
               'patience':3,
               'batch_size': 64,
               'num_measurement': 4,
               'sample_size': 500,
               'num_layer': 6,
               'hidden_size': 10,
               'learning_rate': 0.05,
               'number_simulation':5,
               }
init_config = SimpleNamespace(**init_config)

# COMMAND ----------

def train_GEEN(config, verbose=False):
    def train_one_epoch(epoch, model, train_dataloader, optimizer):
        model.train()
        train_loss = 0

        for batch_idx, batch in enumerate(train_dataloader):
            if config.with_xstar:
                x, x_star = batch
                x = x.to(config.device).double()
                x_star = x_star.to(config.device).double()
            else:
                x, x_index = batch
                x = x.to(config.device).double()

            optimizer.zero_grad()
            x_star_gen = model(x)

            if config.is_normal:
                divergence_loss, normalization = divergenceLoss_wNormal_wKernelDensity(
                    x, x_star_gen, config
                )
            else:
                divergence_loss = divergenceLoss_woNormal_wKernelDensity(x, x_star_gen, config)
                normalization = 0

            loss = torch.mean(divergence_loss + normalization)
            loss.backward()
            train_loss += loss.item()
            
            if config.clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), config.clip)
            optimizer.step()

            #if verbose:
            #    if batch_idx % 1000 == 0:
            #        print(
            #            "Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6e}".format(
            #                epoch,
            #                batch_idx * x.shape[0],
            #                len(train_dataloader.dataset),
            #                100.0 * batch_idx / len(train_dataloader),
            #                loss.item() / x.shape[0],
            #            )
            #        )

        return train_loss / len(train_dataloader.dataset)

    def test_one_epoch(epoch, model, test_dataloader):
        model.eval()
        test_loss = 0

        with torch.no_grad():
            for i, batch in enumerate(test_dataloader):
                if config.with_xstar:
                    x, x_star = batch
                    x = x.to(config.device).double()
                    x_star = x_star.to(config.device).double()
                else:
                    x, x_index = batch
                    x = x.to(config.device).double()
                x_star_gen = model(x)
                if config.is_normal:
                    divergence_loss, normalization = divergenceLoss_wNormal_wKernelDensity(
                        x, x_star_gen, config
                    )
                else:
                    divergence_loss = divergenceLoss_woNormal_wKernelDensity(x, x_star_gen, config)
                    normalization = 0
                loss = torch.mean(divergence_loss + normalization)
                test_loss += loss.item()

        return test_loss / len(test_dataloader.dataset)

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

    genModel = GenDNN(config).to(config.device).double()
    optimizer = optim.Adam(genModel.parameters(), lr=config.learning_rate)

    patience = config.patience
    best_val_loss = np.inf
    epoch_no = 0
    for epoch in range(config.epochs):
        train_loss = train_one_epoch(
            epoch_no, genModel, train_dataloader, optimizer
        )
        val_loss = test_one_epoch(epoch_no, genModel, val_dataloader)
        test_loss = test_one_epoch(epoch_no, genModel, test_dataloader)
        if best_val_loss > val_loss:
            best_val_loss = val_loss
        else:
            patience -= 1

        if patience < 0:
            break
            
        if verbose:
            x_gen_test = (
                genModel(
                    torch.tensor(x_test, device=config.device).view(
                        -1, 1, config.num_measurement
                    )
                )
                .flatten()
                .detach()
                .cpu()
                .numpy()
            )   
            mse = mean_squared_error(x_star_test, x_gen_test, squared=False)
            corr = np.corrcoef(x_star_test, x_gen_test)[0, 1]   
            print(
                "====> Epoch: {} Test corr: {:.4f} Test mse: {:.4E} Test loss: {:.4E}".format(
                    epoch_no, corr, mse, test_loss
                )
            )
        epoch_no += 1

    x_gen_train = (
        genModel(
            torch.tensor(x_train, device=config.device, dtype=torch.float64).view(
                -1, 1, config.num_measurement
            )
        )
        .flatten()
        .detach()
        .cpu()
        .numpy()
        )
    x_gen_val = (
        genModel(
            torch.tensor(x_val, device=config.device, dtype=torch.float64).view(
                -1, 1, config.num_measurement
            )
        )
        .flatten()
        .detach()
        .cpu()
        .numpy()
        )
    x_gen_test = (
        genModel(
            torch.tensor(x_test, device=config.device, dtype=torch.float64).view(
                -1, 1, config.num_measurement
            )
        )
        .flatten()
        .detach()
        .cpu()
        .numpy()
        )

    mse = mean_squared_error(x_star_test, x_gen_test, squared=False)
    corr = np.corrcoef(x_star_test, x_gen_test)[0, 1]

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
        test_loss,
        mse,
        corr,
    )

# COMMAND ----------

# DBTITLE 1,To test training
if False:
    config = copy.deepcopy(init_config)
    config.lm = 0.1
    config.window_size = float(1 * 1.06)
    config.window_size_star = float(1 * 1.06)
    train_GEEN(config, verbose=True)

# COMMAND ----------

def train_GEEN_whyperopt(params, init_config):
    config = copy.deepcopy(init_config)
    if config.is_normal:
        lm = params["lambda"]
    else:
        lm = 0
    window_size = params["window_size"]
    config.lm = float(lm)
    config.window_size = float(window_size * 1.06)
    config.window_size_star = float(window_size * 1.06)

    min_loss = np.inf
    min_mse = np.inf
    max_corr = -np.inf

    with mlflow.start_run(nested=True):
        for i in range(config.number_simulation):
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
                test_loss,
                mse,
                corr
            ) = train_GEEN(config)
            min_loss = min(min_loss, test_loss)
            max_corr = max(corr, max_corr)
            min_mse = min(min_mse, mse)

        mlflow.log_metrics(
                {
                    "mse": min_mse,
                    "corr": max_corr,
                    "test_loss": min_loss,
                }
            )

    loss = - max_corr
    return {
        "loss": loss,
        "status": STATUS_OK,
        "corr": max_corr,
        'mse': min_mse,
        "test_loss": min_loss,
    }

# COMMAND ----------

 if init_config.is_normal:
    hp_space =\
    {
            'lambda': hp.uniform('lambda', 0, 1),
            'window_size': hp.uniform('window_size', 0.5, 5),
        }
else:
    hp_space = {'window_size': hp.uniform('window_size', 0.5, 5)}

# COMMAND ----------

par_train_GEEN_whyperopt = partial(
    train_GEEN_whyperopt,
    init_config=init_config,
)

tpe_algo = tpe.suggest
rand_algo = rand.suggest
spark_trials = SparkTrials(parallelism=None)

with mlflow.start_run():
    best_params = fmin(
        fn=par_train_GEEN_whyperopt,
        space=hp_space,
        algo=tpe_algo,
        trials=spark_trials,
        max_evals=max_evals,
    )
    mlflow.set_tag("discrete_xstar", init_config['discrete_xstar'])
    mlflow.set_tag("is_normal", init_config["is_normal"])
    mlflow.set_tag("linear_error", init_config["linear_error"])
    mlflow.set_tag("error_scale", init_config["error_scale"])
    mlflow.set_tag("x4_distinct", init_config["x4_distinct"])
    mlflow.set_tag("batch_norm", init_config["batch_norm"])
    mlflow.set_tag("loss_fun", init_config["loss_fun"])
