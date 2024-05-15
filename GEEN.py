# Databricks notebook source
!pip install lightning

# COMMAND ----------

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
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning import Trainer
from pytorch_lightning import LightningModule
from lightning.pytorch.loggers import TensorBoardLogger

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
               'dir': '/dbfs/mnt/gti/GEEN'
               }

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

class GenDataset(Dataset):
    def __init__(self, data, data_star):
        ## Fetch dataset
        self.data = data
        self.data_star = data_star
        self.len = data.shape[0]
    def __len__(self):
        return self.len
    def __getitem__(self, idx):
        return self.data[idx], self.data_star[idx]
    
def m1(x):
    return x
def m1_wonormal(x):
    return x**2 + x
def m2(x):
    return 1/(1+np.exp(x))
def m3(x):
    return x**2
def m4(x):
    return np.log(1+np.exp(x))

def x4_distinct(x):
    #general error term
    #eps_4: binary
    #x_4 = norm.cdf(x)*(-1)^(eps_5>0.5)
    eps_4 = np.random.binomial(1, 0.5, len(x))
    x_4 = norm.cdf(x/3)*(-1)**eps_4
    return x_4

def get_data(N_sample, config, random_seed=0):
    '''
    eps_1: normal distribution
    eps_2: beta distribution - alpha and beta
    eps_3: laplace distribution - loc and scale
    eps_4: uniform - a and b
    '''
    # set seed
    np.random.seed(random_seed)

    mu_star = 0
    sigma_star = 2

    if config['discrete_xstar']:
        x_star = np.random.multinomial(10, [0.5]*2, N_sample)[:,0]
    else:
        x_star = np.random.normal(mu_star, sigma_star, N_sample)     

    # linear error experiment in the paper
    if config['linear_error']:
        mu_1, scale_1 = 0, 0.5
        alpha, beta = 2, 2
        loc, scale = 0, 0.5
        a, scale_b = 0, 0.5 

        scale_1, beta, scale, scale_b = \
            config['error_scale']*scale_1, config['error_scale']*beta, config['error_scale']*scale, config['error_scale']*scale_b

        eps_1 = np.random.normal(mu_1, scale_1*np.abs(x_star)) 
        eps_2 = np.random.beta(alpha, beta, N_sample)-1/(1+beta/alpha)
        eps_3 = np.random.laplace(loc, scale*np.abs(x_star), N_sample)-loc
        eps_4 = np.random.uniform(a, scale_b*np.abs(x_star))-(a+scale_b*np.abs(x_star))/2
        
    else:
        mu_1, sigma_1 = 0, 1
        alpha, beta = 2, 2
        loc, scale = 0, 1
        a,b = 0, 1

        sigma_1, beta, scale, b = \
            config['error_scale']*sigma_1, config['error_scale']*beta, config['error_scale']*scale, config['error_scale']*b

        eps_1 = np.random.normal(mu_1, sigma_1, N_sample)  
        eps_2 = np.random.beta(alpha, beta, N_sample)-1/(1+beta/alpha)
        eps_3 = np.random.laplace(loc, scale, N_sample)-loc
        eps_4 = np.random.uniform(a, b, N_sample)-(a+b)/2

    x_star_vec = x_star

    if config['is_normal']:
        x1 = m1(x_star)+eps_1
    else:
        x1 = m1_wonormal(x_star)+eps_1
    x2 = m2(x_star)+eps_2
    x3 = m3(x_star)+eps_3
    if config['x4_distinct']:
        x4 = x4_distinct(x_star)
    else:
        x4 = m4(x_star)+eps_4
    
    x_vec = np.stack((x1,x2,x3,x4), axis=1)

    n_train, n_val, n_test = int(N_sample*0.8), int(N_sample*0.1), int(N_sample*0.1)
    x_train = x_vec[:n_train, :]
    x_star_train = x_star_vec[:n_train]

    x_val = x_vec[n_train:n_train+n_val, :]
    x_star_val = x_star_vec[n_train:n_train+n_val]

    x_test = x_vec[n_train+n_val:, :]
    x_star_test = x_star_vec[n_train+n_val:]

    #draw m points from train/dev/test data points to construct datasets
    sample_train = np.random.choice(n_train, size=(n_train, config['sample_size']), replace=True, p=None)
    sample_val = np.random.choice(n_val, size=(n_val, config['sample_size']), replace=True, p=None)
    sample_test = np.random.choice(n_test, size=(n_test, config['sample_size']), replace=True, p=None)

    x_train_resampled = np.zeros((n_train, config['sample_size'], 4,))
    x_val_resampled = np.zeros((n_val, config['sample_size'], 4,))
    x_test_resampled = np.zeros((n_test, config['sample_size'], 4,))

    x_star_train_resampled = np.zeros((n_train, config['sample_size']))
    x_star_val_resampled = np.zeros((n_val, config['sample_size']))
    x_star_test_resampled = np.zeros((n_test, config['sample_size']))

    for i in range(0,n_train):
        x_train_resampled[i,:,:] = x_train[sample_train[i, :], :]
        x_star_train_resampled[i, :] = x_star_train[sample_train[i, :]]
        if i < n_val:
            x_val_resampled[i,:,:] = x_val[sample_val[i, :], :]
            x_star_val_resampled[i, :] = x_star_val[sample_val[i, :]]
        if i < n_test:
            x_test_resampled[i,:,:] = x_test[sample_test[i, :], :]
            x_star_test_resampled[i, :] = x_star_test[sample_test[i, :]]

    train_dataset = GenDataset(x_train_resampled, x_star_train_resampled)
    val_dataset = GenDataset(x_val_resampled, x_star_val_resampled)
    test_dataset = GenDataset(x_test_resampled, x_star_test_resampled)

    train_dataloader = DataLoader(
            dataset=train_dataset,
            batch_size=config['batch_size'],
            shuffle=True)
    val_dataloader = DataLoader(
            dataset=val_dataset,
            batch_size=config['batch_size'],
            shuffle=False)
    test_dataloader = DataLoader(
            dataset=test_dataset,
            batch_size=config['batch_size'],
            shuffle=False)

    return train_dataloader, val_dataloader, test_dataloader,\
           x_train, x_star_train, \
           x_val, x_star_val,\
           x_test, x_star_test

# COMMAND ----------

class LitGenDNN(LightningModule):
    def __init__(self, config):

        super().__init__()

        # Set our init args as class attributes
        self.batch_size = config["batch_size"]
        self.window_size = config["window_size"]

        if "window_size_star" in config:
            self.window_size_star = config["window_size_star"]
        else:
            if type(config["window_size"]) is list:
                self.window_size_star = config["window_size"][0]
            else:
                self.window_size_star = config["window_size"]

        self.normalization_multiplier = None  # lambda in the paper
        if "lambda" in config:
            self.normalization_multiplier = config["lambda"]

        self.input_size = input_size = config["input_size"]
        self.sample_size = config["sample_size"]
        self.num_layer = num_layer = config["num_layer"]
        self.hidden_size = hidden_size = config["hidden_size"]
        self.learning_rate = config["learning_rate"]
        self.drop_out = drop_out = config["drop_out"]
        self.activation = activation = config["activation"]
        self.com_device = config["device"]
        self.is_normal = config["is_normal"]
        self.with_xstar = config["with_xstar"]
        self.loss_fun = config["loss_fun"]
        self.batch_norm = config['batch_norm']

        # Define PyTorch model
        if self.batch_norm:
            self.layer_list = [
                               nn.BatchNorm1d(input_size),
                               nn.Linear(input_size, hidden_size),
                               activation,
                               nn.BatchNorm1d(hidden_size),
                               nn.Dropout(drop_out),
                              ]
        else:
            self.layer_list = [
                               nn.Linear(input_size, hidden_size),
                               activation,
                               nn.Dropout(drop_out),
                              ]
        for i in range(num_layer-1):
            if self.batch_norm:
                self.layer_list.extend([nn.Linear(hidden_size, hidden_size),
                                        activation,
                                        nn.BatchNorm1d(hidden_size),
                                        nn.Dropout(drop_out),
                                        ])
            else:
                self.layer_list.extend([nn.Linear(hidden_size, hidden_size),
                                        activation,
                                        nn.Dropout(drop_out),
                                        ])
        self.layer_list.append(nn.Linear(hidden_size, 1))
        self.dnn = nn.Sequential(*self.layer_list)

    def forward(self, x):
        sample_size = x.shape[1]
        input_size = x.shape[2]
        x = x.reshape(-1, input_size)
        x = self.dnn(x)
        return x.reshape(-1, sample_size, 1)

    def divergence_loss_normal(self, x, x_star_gen):
        # kernel function
        def K_func(x):
            return torch.exp(Normal(0, 1).log_prob(x))

        def f_x1_cond_xstar(
            x1_loop, x_star_loop, x1_vec_loop, x_star_vec_loop, h1_loop
        ):
            # equation 13 in the paper. X1 is special since it is used to normalize generalized X*
            # x1_loop: x1, batch_size*N*1
            # x_star_loop: generalized x_star, batch_size*N*1
            # x1_vec_loop: X1; a vector of x1 for kenel density proximation, batch_size*N*N
            # x_star_vec_loop: generalized X*; a vector of generalized x* for kenel density proximation, batch_size*N*N*1
            # h1_loop: h1; bandwith for x1, batch_size*N*1
            N = x_star_vec_loop.shape[2]
            x_prime_loop = x1_loop - x_star_loop  # batch*N*1
            arg = (
                K_func(
                    (
                        (x1_vec_loop.unsqueeze(dim=3) - x_star_vec_loop).squeeze(dim=3)
                        - x_prime_loop
                    )
                    / h1_loop
                )
                / h1_loop
            )
            return torch.mean(arg, dim=2)

        def f_xj_xstar(
            x_j_loop, x_star_loop, x_j_vec_loop, x_star_vec_loop, h_j_loop, h_star_loop
        ):
            # equation 14 in the paper
            # x_j_loop: xj, batch_size*N*1
            # x_star_loop: generalized x_star, batch_size*N*1
            # x_j_vec_loop: Xj; a vector of xj for kenel density proximation, batch_size*N*N
            # x_star_vec_loop: generalized X*; a vector of generalized x* for kenel density proximation, batch_size*N*N*1
            # h_j_loop: hj; bandwith for xj, batch_size*N*1
            # h_star_loop: h*; bandwith for generalized x*, batch_size*N*1
            N = x_j_vec_loop.shape[2]
            arg1 = (
                K_func((x_j_vec_loop - x_j_loop) / h_j_loop) / h_j_loop
            )  # batch_size*N*N
            arg2 = (
                K_func((x_star_vec_loop.squeeze(dim=3) - x_star_loop) / h_star_loop)
                / h_star_loop
            )  # batch_size*N*N
            return torch.sum(arg1 * arg2 / N, dim=2)

        def f_xstar(x_star_loop, x_star_vec_loop, h_star_loop):
            # equation 15 in the paper
            # x_star_loop: generalized x_star, batch_size*N*1
            # x_star_vec_loop: generalized X*; a vector of generalized x* for kenel density proximation, batch_size*N*N*1
            # h_star_loop: h*; bandwith for generalized x*, batch_size*N*1
            N = x_star_vec_loop.shape[2]
            arg = (
                K_func((x_star_vec_loop.squeeze(dim=3) - x_star_loop) / h_star_loop)
                / h_star_loop
            )  # batch_size*N*N
            return torch.sum(arg, dim=2) / N

        def f_xj_cond_xstar(
            x_j_loop, x_star_loop, x_j_vec_loop, x_star_vec_loop, h_j_loop, h_star_loop
        ):
            # equation 13 in the paper
            # x_j_loop: xj, batch_size*N*1
            # x_star_loop: generalized x_star, batch_size*N*1
            # x_j_vec_loop: Xj; a vector of xj for kenel density proximation, batch_size*N*N
            # x_star_vec_loop: generalized X*; a vector of generalized x* for kenel density proximation, batch*N*N*1
            # h_j_loop: hj; bandwith for xj, batch_size*N*1
            # h_star_loop: h*; bandwith for generalized x*, batch_size*N*1
            return f_xj_xstar(
                x_j_loop,
                x_star_loop,
                x_j_vec_loop,
                x_star_vec_loop,
                h_j_loop,
                h_star_loop,
            ) / f_xstar(x_star_loop, x_star_vec_loop, h_star_loop)

        def f_joint(x, x_star, h_loop, h_star_loop):
            # equation 16 in the paper
            # x: x1, x2, ... xk, batch_size * N * k
            # x_star: generalized x_star, batch_size * N * 1
            # h_loop: h1, h2, ... hk; bandwith for xj, batch_size*N *1 * k
            # h_star_loop: h*; bandwith for generalized x*, batch_size*N*1
            batch_size = x.shape[0]
            k = x.shape[2]
            N = x.shape[1]

            x_vec_loop = x.unsqueeze(dim=1).expand(-1, N, -1, -1)  # X1, X2, ... Xk
            x_loop = x.unsqueeze(dim=2)  # x1, x2, ... xk
            x_star_vec_loop = x_star.unsqueeze(dim=1).expand(-1, N, -1, -1)
            x_star_loop = x_star
            arg = (
                K_func((x_star_vec_loop.squeeze(dim=3) - x_star_loop) / h_star_loop)
                / h_star_loop
            )

            for j in range(k):
                arg = (
                    arg
                    * K_func(
                        (x_vec_loop[:, :, :, j] - x_loop[:, :, :, j])
                        / h_loop[:, :, :, j]
                    )
                    / h_loop[:, :, :, j]
                )
            return torch.mean(arg, dim=2)

        def loss(x, x_star_gen):
            # x: measurement of true x_star
            # x_star_gen: generated x_star
            k = x.shape[2]
            N = x.shape[1]
            batch_size = x.shape[0]

            x_loop = x.unsqueeze(dim=1).expand(-1, N, -1, -1)
            x_obs_loop = x.unsqueeze(dim=2)
            x_star_loop = x_star_gen.unsqueeze(dim=1).expand(-1, N, -1, -1)

            h_vec_loop = torch.zeros(
                (batch_size, N, 1, k), device=self.com_device, dtype=torch.float64
            )
            for j in range(k):
                if j == 0:
                    if type(self.window_size) is list:
                        h_vec_loop[:, :, :, [j]] = (
                            (
                                self.window_size[j]
                                * torch.std(
                                    x[:, :, [j]] - x_star_gen, dim=1, keepdim=True
                                )
                                * N ** (-1 / 5)
                            )
                            .unsqueeze(dim=1)
                            .expand(-1, N, -1, -1)
                        )
                    else:
                        h_vec_loop[:, :, :, [j]] = (
                            (
                                self.window_size
                                * torch.std(
                                    x[:, :, [j]] - x_star_gen, dim=1, keepdim=True
                                )
                                * N ** (-1 / 5)
                            )
                            .unsqueeze(dim=1)
                            .expand(-1, N, -1, -1)
                        )
                else:
                    if type(self.window_size) is list:
                        h_vec_loop[:, :, :, [j]] = (
                            (
                                self.window_size[j]
                                * torch.std(x[:, :, [j]], dim=1, keepdim=True)
                                * N ** (-1 / 5)
                            )
                            .unsqueeze(dim=1)
                            .expand(-1, N, -1, -1)
                        )
                    else:
                        h_vec_loop[:, :, :, [j]] = (
                            (
                                self.window_size
                                * torch.std(x[:, :, [j]], dim=1, keepdim=True)
                                * N ** (-1 / 5)
                            )
                            .unsqueeze(dim=1)
                            .expand(-1, N, -1, -1)
                        )
            h_star = (
                self.window_size_star * torch.std(x_star_gen, dim=1) * N ** (-1 / 5)
            )
            h_star_loop = h_star.unsqueeze(dim=1).expand(-1, N, -1)

            arg1_loop = f_joint(x, x_star_gen, h_vec_loop, h_star_loop)
            arg1_loop_log = torch.log(arg1_loop)

            arg2_loop = torch.ones(
                batch_size, N, device=self.com_device, dtype=torch.float64
            )
            x_star_obs_loop = x_star_gen

            for j in range(k):
                x_j_obs_loop = x_obs_loop[:, :, :, j]
                x_j_vec_loop = x_loop[:, :, :, j]
                h_j_loop = h_vec_loop[:, :, :, j]
                if j == 0:
                    arg2_loop *= f_x1_cond_xstar(
                        x_j_obs_loop,
                        x_star_obs_loop,
                        x_j_vec_loop,
                        x_star_loop,
                        h_j_loop,
                    )
                else:
                    arg2_loop *= f_xj_cond_xstar(
                        x_j_obs_loop,
                        x_star_obs_loop,
                        x_j_vec_loop,
                        x_star_loop,
                        h_j_loop,
                        h_star_loop,
                    )
            arg2_loop *= f_xstar(x_star_obs_loop, x_star_loop, h_star_loop)
            arg2_loop_log = torch.log(arg2_loop)

            if self.loss_fun == 'dl_hellinger':
                hellinger_distance = torch.square(torch.sqrt(arg1_loop) - torch.sqrt(arg2_loop))
                divergence_loss = torch.mean(hellinger_distance, dim=1)
            elif self.loss_fun == 'kl_loss':
                # KL_divergence = torch.abs(arg1_loop*(arg1_loop_log-arg2_loop_log))   ## PyTorch and Tensorflow implementation
                KL_divergence = arg1_loop * (arg1_loop_log - arg2_loop_log)
                divergence_loss = torch.mean(KL_divergence, dim=1)
            else:
                arg1 = torch.mean(arg1_loop_log, dim=1)  ##take as sample average
                arg2 = torch.mean(arg2_loop_log, dim=1)
                divergence_loss = torch.abs(arg1 - arg2)
                # divergence_loss = arg1 - arg2

            if self.normalization_multiplier:
                normalization = self.normalization_multiplier * torch.square(
                    torch.mean(x_star_gen.squeeze(dim=2), dim=1)
                    - torch.mean(x[:, :, 0], dim=1)
                )
            else:
                normalization = 0

            return divergence_loss, normalization

        return loss(x, x_star_gen)

    # this is only used for non-normalization case
    def divergence_loss_wo_normal(self, x, x_star_gen):
        def K_func(x):
            return torch.exp(Normal(0, 1).log_prob(x))

        def f_xj_xstar(
            x_j_loop, x_star_loop, x_j_vec_loop, x_star_vec_loop, h_j_loop, h_star_loop
        ):
            # equation 14 in the paper
            # x_j_loop: xj, small x
            # x_star_loop: generalized x_star, small x*
            # x_j_vec_loop: Xj, capitalized X; a vector of xj for kenel density proximation
            # x_star_vec_loop: generalized X*, capitalized X*; a vector of generalized x* for kenel density proximation
            # h_j_loop: hj; bandwith for xj
            # h_star_loop: h*; bandwith for generalized x*
            N = x_j_vec_loop.shape[2]
            arg1 = (
                K_func((x_j_vec_loop - x_j_loop) / h_j_loop) / h_j_loop
            )  # batch_size*N*N
            arg2 = (
                K_func((x_star_vec_loop.squeeze(dim=3) - x_star_loop) / h_star_loop)
                / h_star_loop
            )  # batch_size*N*N
            return torch.sum(arg1 * arg2 / N, dim=2)

        def f_xstar(x_star_loop, x_star_vec_loop, h_star_loop):
            # equation 15 in the paper
            # x_star_loop: generalized x_star, small x*
            # x_star_vec_loop: generalized X*, capitalized X*; a vector of generalized x* for kenel density proximation
            # h_star_loop: h*; bandwith for generalized x*
            N = x_star_vec_loop.shape[2]
            arg = (
                K_func((x_star_vec_loop.squeeze(dim=3) - x_star_loop) / h_star_loop)
                / h_star_loop
            )  # batch_size*N*N
            return torch.sum(arg, dim=2) / N

        def f_xj_cond_xstar(
            x_j_loop, x_star_loop, x_j_vec_loop, x_star_vec_loop, h_j_loop, h_star_loop
        ):
            # equation 13 in the paper
            # x_j_loop: xj, small x
            # x_star_loop: generalized x_star, small x*
            # x_j_vec_loop: Xj, capitalized X; a vector of xj for kenel density proximation
            # x_star_vec_loop: generalized X*, capitalized X*; a vector of generalized x* for kenel density proximation
            # h_j_loop: hj; bandwith for xj
            # h_star_loop: h*; bandwith for generalized x*
            return f_xj_xstar(
                x_j_loop,
                x_star_loop,
                x_j_vec_loop,
                x_star_vec_loop,
                h_j_loop,
                h_star_loop,
            ) / f_xstar(x_star_loop, x_star_vec_loop, h_star_loop)

        def f_joint(x, x_star, h_loop, h_star_loop):
            # equation 16 in the paper
            # x: x1, x2, ... xk
            # x_star: generalized x_star
            # h_loop: h1, h2, ... hk; bandwith for xj
            # h_star_loop: h*; bandwith for generalized x*
            batch_size = x.shape[0]
            k = x.shape[2]
            N = x.shape[1]
            # Xj, capitalized X in eq 6
            x_vec_loop = x.unsqueeze(dim=1).expand(-1, N, -1, -1)
            x_loop = x.unsqueeze(dim=2)  # xj, small x in eq 6
            # capitalized generated X* in eq6
            x_star_vec_loop = x_star.unsqueeze(dim=1).expand(-1, N, -1, -1)
            x_star_loop = x_star  # small generalized x* in eq 6
            arg = (
                K_func((x_star_vec_loop.squeeze(dim=3) - x_star_loop) / h_star_loop)
                / h_star_loop
            )

            for j in range(k):
                arg = (
                    arg
                    * K_func(
                        (x_vec_loop[:, :, :, j] - x_loop[:, :, :, j])
                        / h_loop[:, :, :, j]
                    )
                    / h_loop[:, :, :, j]
                )
            return torch.mean(arg, dim=2)

        def loss(x, x_star_gen):
            # x: measurement of x_star
            # x_star_gen: generated x_star
            k = x.shape[2]
            N = x.shape[1]
            batch_size = x.shape[0]

            x_loop = x.unsqueeze(dim=1).expand(-1, N, -1, -1)
            x_obs_loop = x.unsqueeze(dim=2)
            x_star_loop = x_star_gen.unsqueeze(dim=1).expand(-1, N, -1, -1)

            h_vec_loop = torch.zeros(
                (batch_size, N, 1, k), device=self.com_device, dtype=torch.float64
            )
            for j in range(k):
                if type(self.window_size) is list:
                    h_vec_loop[:, :, :, [j]] = (
                        (
                            self.window_size[j]
                            * torch.std(x[:, :, [j]], dim=1, keepdim=True)
                            * N ** (-1 / 5)
                        )
                        .unsqueeze(dim=1)
                        .expand(-1, N, -1, -1)
                    )
                else:
                    h_vec_loop[:, :, :, [j]] = (
                        (
                            self.window_size
                            * torch.std(x[:, :, [j]], dim=1, keepdim=True)
                            * N ** (-1 / 5)
                        )
                        .unsqueeze(dim=1)
                        .expand(-1, N, -1, -1)
                    )
            h_star = (
                self.window_size_star * torch.std(x_star_gen, dim=1) * N ** (-1 / 5)
            )  # batch_size*1
            h_star_loop = h_star.unsqueeze(dim=1).expand(
                -1, N, -1
            )  # batch_size*N*1 batch,loop

            arg1_loop = f_joint(x, x_star_gen, h_vec_loop, h_star_loop)  # batch*loop
            arg1_loop_log = torch.log(arg1_loop)

            arg2_loop = torch.ones(
                batch_size, N, device=self.com_device, dtype=torch.float64
            )  # batch_size * N
            x_star_obs_loop = x_star_gen  # batch*N*1

            for j in range(k):
                x_j_obs_loop = x_obs_loop[:, :, :, j]  # batch_size*N*1
                x_j_vec_loop = x_loop[:, :, :, j]  # batch_size*N*N
                h_j_loop = h_vec_loop[:, :, :, j]  # batch_size*N*1
                arg2_loop *= f_xj_cond_xstar(
                    x_j_obs_loop,
                    x_star_obs_loop,
                    x_j_vec_loop,
                    x_star_loop,
                    h_j_loop,
                    h_star_loop,
                )
            arg2_loop *= f_xstar(x_star_obs_loop, x_star_loop, h_star_loop)
            arg2_loop_log = torch.log(arg2_loop)

            if self.loss_fun == 'dl_hellinger':
                hellinger_distance = torch.square(torch.sqrt(arg1_loop) - torch.sqrt(arg2_loop))
                divergence_loss = torch.mean(hellinger_distance, dim=1)
            elif self.loss_fun == 'kl_loss':
                # KL_divergence = torch.abs(arg1_loop*(arg1_loop_log-arg2_loop_log))   ## PyTorch and Tensorflow implementation
                KL_divergence = arg1_loop * (arg1_loop_log - arg2_loop_log)
                divergence_loss = torch.mean(KL_divergence, dim=1)
            else:
                arg1 = torch.mean(arg1_loop_log, dim=1)  ##take as sample average
                arg2 = torch.mean(arg2_loop_log, dim=1)
                divergence_loss = torch.abs(arg1 - arg2)
                # divergence_loss = arg1 - arg2
            
            return divergence_loss

        return loss(x, x_star_gen)

    def training_step(self, batch, batch_idx):
        if self.with_xstar:
            x, x_star = batch
            x_star = x_star.to(self.com_device)
        else:
            x, x_index = batch
        x = x.to(self.com_device)

        x_star_gen = self(x)
        if self.is_normal:
            divergence_loss, normalization = self.divergence_loss_normal(x, x_star_gen)
        else:
            divergence_loss = self.divergence_loss_wo_normal(x, x_star_gen)
            normalization = 0

        if self.with_xstar:
            train_mse = F.mse_loss(x_star_gen.squeeze(dim=2), x_star)
            train_corr = np.corrcoef(
                torch.flatten(x_star).detach().cpu().numpy(),
                torch.flatten(x_star_gen).detach().cpu().numpy(),
            )[0, 1]
            self.log("train_mse", train_mse)
            self.log("train_corr", train_corr)

        self.log(
            "train_loss", torch.mean(divergence_loss + normalization), prog_bar=True
        )
        return torch.mean(divergence_loss + normalization)

    def validation_step(self, batch, batch_idx):
        if self.with_xstar:
            x, x_star = batch
            x_star = x_star.to(self.com_device)
        else:
            x, x_index = batch
        x = x.to(self.com_device)

        x_star_gen = self(x)
        if self.with_xstar:
            # val_mse and val_corr only used to monitor, not used for validation/early stoping
            val_mse = F.mse_loss(x_star_gen.squeeze(dim=2), x_star)
            val_corr = np.corrcoef(
                torch.flatten(x_star).detach().cpu().numpy(),
                torch.flatten(x_star_gen).detach().cpu().numpy(),
            )[0, 1]
            self.log("val_mse", val_mse, prog_bar=True)
            self.log("val_corr", val_corr, prog_bar=True)

        if self.is_normal:
            val_divergence_loss, normalization = self.divergence_loss_normal(
                x, x_star_gen
            )
        else:
            val_divergence_loss = self.divergence_loss_wo_normal(x, x_star_gen)
            normalization = 0
        val_loss = torch.mean(val_divergence_loss + normalization)
        self.log("val_loss", val_loss, prog_bar=True)
        return val_loss

    def test_step(self, batch, batch_idx):
        if self.with_xstar:
            x, x_star = batch
            x_star = x_star.to(self.com_device)
        else:
            x, x_index = batch
        x = x.to(self.com_device)

        x_star_gen = self(x)
        if self.is_normal:
            test_divergence_loss, normalization = self.divergence_loss_normal(
                x, x_star_gen
            )
            normalization = torch.tensor(normalization, dtype=torch.float64)
            self.log("test_normalization", torch.mean(normalization), prog_bar=False)
        else:
            test_divergence_loss = self.divergence_loss_wo_normal(x, x_star_gen)
            normalization = 0
            self.log("test_normalization", 0, prog_bar=False)
        if self.with_xstar:
            test_mse = F.mse_loss(x_star_gen.squeeze(dim=2), x_star)
            test_corr = np.corrcoef(
                torch.flatten(x_star).detach().cpu().numpy(),
                torch.flatten(x_star_gen).detach().cpu().numpy(),
            )[0, 1]
            self.log("test_mse", test_mse, prog_bar=True)
            self.log("test_corr", test_corr, prog_bar=True)
        test_divergence_loss = torch.mean(test_divergence_loss)
        test_loss = torch.mean(test_divergence_loss + normalization)
        self.log("test_divergence_loss", test_divergence_loss, prog_bar=True)
        self.log(
            "test_loss", torch.mean(test_divergence_loss + normalization), prog_bar=True
        )
        return test_loss

    def predict_step(self, batch, batch_idex):
        x, _ = batch
        if x.dim() == 1:
            x = x.view(1, 1, self.input_size)
        elif x.dim() == 2:
            x = x.view(-1, 1, self.input_size)
        else:
            raise RuntimeError("tensor dimension must be 1d or 2d (batch prediction)")
        return self(x)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)
        return optimizer

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
    ) = get_data(config['N_sample'], config)

    genModel = LitGenDNN(config).double()
    genModel = genModel.to(config['device'])

    early_stop_callback = EarlyStopping(
        monitor="val_loss", patience=config["patience"], verbose=True
    )
    logger = TensorBoardLogger(save_dir=config['dir'])

    count = 0
    if config['device'] == "cuda":
        accelerator = "gpu"

    error_message = ""
    while count < 10:
        try:
            trainer = Trainer(
                accelerator=accelerator,
                max_epochs=config["epochs"],
                log_every_n_steps=10,
                gradient_clip_val=config["clip"],
                callbacks=[early_stop_callback],
                precision=64,
                logger=logger,
                enable_progress_bar=False,
                enable_model_summary=False,
            )
            trainer.fit(
                genModel,
                train_dataloaders=train_dataloader,
                val_dataloaders=val_dataloader,
            )
            break
        except Exception as e:
            error_message = e
            genModel = LitGenDNN(config).double()
            genModel = genModel.to(config['device'])
            count += 1

    if count >= 10:
        raise ValueError("Failed to train the model with error: %s" % error_message)
    else:
        print(f"succeed after {count} trials...")

    test_result = trainer.test(dataloaders=test_dataloader)

    for l in test_result:
        for k, v in l.items():
            print(f"{k}: {v}")

    x_gen_train = (
        genModel.predict_step([torch.tensor(x_train), None], 0)
        .flatten()
        .detach()
        .cpu()
        .numpy()
    )
    x_gen_val = (
        genModel.predict_step([torch.tensor(x_val), None], 0)
        .flatten()
        .detach()
        .cpu()
        .numpy()
    )
    x_gen_test = (
        genModel.predict_step([torch.tensor(x_test), None], 0)
        .flatten()
        .detach()
        .cpu()
        .numpy()
    )

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
