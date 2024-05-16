# Databricks notebook source
import numpy as np
import torch
import copy
from torch import optim
from sklearn.metrics import mean_squared_error

import mlflow
from hyperopt import fmin, rand, tpe, hp, SparkTrials, STATUS_OK
from functools import partial

from src.GEEN import *
from src.simulationDataGen import get_simulation_data
from src.train_validation import train_and_val_GEEN

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

    with mlflow.start_run(nested=True):
        for i in range(config.number_simulation):
            genModel = GenDNN(config).to(config.device).double()
            
            genModel, test_loss = train_and_val_GEEN(config, genModel, train_dataloader, val_dataloader, test_dataloader, (x_test, x_star_test))
            
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
