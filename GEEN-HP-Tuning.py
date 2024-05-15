# Databricks notebook source
!pip install lightning

# COMMAND ----------

import os
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

experiment_name = "/Users/cadminyliu10@imf.org/GEEN/GEEN-HP-Tuning"
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
max_evals = 256
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
               'input_size': 4,
               'sample_size': 500,
               'num_layer': 6,
               'hidden_size': 10,
               'learning_rate': 0.05,
               'number_simulation':5,
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
    #realization term
    #eps_4: binary
    #x_4 = norm.cdf*(-1)^(eps_5>0.5)
    eps_4 = np.random.binomial(1, 0.5, len(x))
    x_4 = norm.cdf(x/3)*(-1)**eps_4
    return x_4

def normalize(x):
    #x is a numpy array
    #normalize x with mean and std
    mean = x.mean()
    std = x.std()
    return (x-mean)/std

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

class GenDNN(nn.Module):
    def __init__(self, config):
        super().__init__()
        # Set our init args as class attributes
        self.batch_size = config['batch_size']
            
        self.input_size = input_size = config['input_size']
        self.sample_size = config['sample_size']
        self.num_layer = num_layer = config['num_layer']
        self.hidden_size = hidden_size = config['hidden_size']
        self.drop_out = drop_out = config['drop_out']
        self.activation = activation = config['activation']
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

# COMMAND ----------

def divergence_loss_normal(x, x_star_gen, config):
    # kernel function
    def K_func(x):
        return torch.exp(Normal(0, 1).log_prob(x))

    def f_x1_cond_xstar(x1_loop, x_star_loop, x1_vec_loop, x_star_vec_loop, h1_loop):
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
        arg1 = K_func((x_j_vec_loop - x_j_loop) / h_j_loop) / h_j_loop  # batch_size*N*N
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
            x_j_loop, x_star_loop, x_j_vec_loop, x_star_vec_loop, h_j_loop, h_star_loop
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
                    (x_vec_loop[:, :, :, j] - x_loop[:, :, :, j]) / h_loop[:, :, :, j]
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

        h_vec_loop = torch.zeros((batch_size, N, 1, k), device=config['device'], dtype=torch.float64)
        for j in range(k):
            if j == 0:
                if type(config['window_size']) is list:
                    h_vec_loop[:, :, :, [j]] = (
                        (
                            config['window_size'][j]
                            * torch.std(x[:, :, [j]] - x_star_gen, dim=1, keepdim=True)
                            * N ** (-1 / 5)
                        )
                        .unsqueeze(dim=1)
                        .expand(-1, N, -1, -1)
                    )
                else:
                    h_vec_loop[:, :, :, [j]] = (
                        (
                            config['window_size']
                            * torch.std(x[:, :, [j]] - x_star_gen, dim=1, keepdim=True)
                            * N ** (-1 / 5)
                        )
                        .unsqueeze(dim=1)
                        .expand(-1, N, -1, -1)
                    )
            else:
                if type(config['window_size']) is list:
                    h_vec_loop[:, :, :, [j]] = (
                        (
                            config['window_size'][j]
                            * torch.std(x[:, :, [j]], dim=1, keepdim=True)
                            * N ** (-1 / 5)
                        )
                        .unsqueeze(dim=1)
                        .expand(-1, N, -1, -1)
                    )
                else:
                    h_vec_loop[:, :, :, [j]] = (
                        (
                            config['window_size']
                            * torch.std(x[:, :, [j]], dim=1, keepdim=True)
                            * N ** (-1 / 5)
                        )
                        .unsqueeze(dim=1)
                        .expand(-1, N, -1, -1)
                    )
        h_star = config['window_size_star'] * torch.std(x_star_gen, dim=1) * N ** (-1 / 5)
        h_star_loop = h_star.unsqueeze(dim=1).expand(-1, N, -1)

        arg1_loop = f_joint(x, x_star_gen, h_vec_loop, h_star_loop)

        arg2_loop = torch.ones(batch_size, N, device=config['device'], dtype=torch.float64)
        x_star_obs_loop = x_star_gen

        for j in range(k):
            x_j_obs_loop = x_obs_loop[:, :, :, j]
            x_j_vec_loop = x_loop[:, :, :, j]
            h_j_loop = h_vec_loop[:, :, :, j]
            if j == 0:
                arg2_loop *= f_x1_cond_xstar(
                    x_j_obs_loop, x_star_obs_loop, x_j_vec_loop, x_star_loop, h_j_loop
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
       
        if config['loss_fun'] == 'dl_hellinger':
            hellinger_distance = torch.square(torch.sqrt(arg1_loop) - torch.sqrt(arg2_loop))
            divergence_loss = torch.mean(hellinger_distance, dim=1)
        elif config['loss_fun'] == 'kl_loss':
            # KL_divergence = torch.abs(arg1_loop*(arg1_loop_log-arg2_loop_log))   ## PyTorch and Tensorflow implementation
            arg1_loop_log = torch.log(arg1_loop)
            arg2_loop_log = torch.log(arg2_loop)
            KL_divergence = arg1_loop * (arg1_loop_log - arg2_loop_log)
            divergence_loss = torch.mean(KL_divergence, dim=1)
        else:
            arg1_loop_log = torch.log(arg1_loop)
            arg2_loop_log = torch.log(arg2_loop)
            arg1 = torch.mean(arg1_loop_log, dim=1)  ##take as sample average
            arg2 = torch.mean(arg2_loop_log, dim=1)
            divergence_loss = torch.abs(arg1 - arg2)
            # divergence_loss = arg1 - arg2

        if config['lambda']:
            normalization = config['lambda'] * torch.square(
                torch.mean(x_star_gen.squeeze(dim=2), dim=1)
                - torch.mean(x[:, :, 0], dim=1)
            )
        else:
            normalization = 0

        return divergence_loss, normalization

    return loss(x, x_star_gen)

# this is only used for non-normalization case
def divergence_loss_wo_normal(x, x_star_gen, config):
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
        arg1 = K_func((x_j_vec_loop - x_j_loop) / h_j_loop) / h_j_loop  # batch_size*N*N
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
            x_j_loop, x_star_loop, x_j_vec_loop, x_star_vec_loop, h_j_loop, h_star_loop
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
        x_vec_loop = x.unsqueeze(dim=1).expand(
            -1, N, -1, -1
        )  # Xj, capitalized X in eq 6
        x_loop = x.unsqueeze(dim=2)  # xj, small x in eq 6
        x_star_vec_loop = x_star.unsqueeze(dim=1).expand(
            -1, N, -1, -1
        )  # capitalized generated X* in eq6
        x_star_loop = x_star  # small generalized x* in eq 6
        arg = (
            K_func((x_star_vec_loop.squeeze(dim=3) - x_star_loop) / h_star_loop)
            / h_star_loop
        )

        for j in range(k):
            arg = (
                arg
                * K_func(
                    (x_vec_loop[:, :, :, j] - x_loop[:, :, :, j]) / h_loop[:, :, :, j]
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

        h_vec_loop = torch.zeros((batch_size, N, 1, k), device=config['device'], dtype=torch.float64)
        for j in range(k):
            if type(config['window_size']) is list:
                h_vec_loop[:, :, :, [j]] = (
                    (
                        config['window_size'][j]
                        * torch.std(x[:, :, [j]], dim=1, keepdim=True)
                        * N ** (-1 / 5)
                    )
                    .unsqueeze(dim=1)
                    .expand(-1, N, -1, -1)
                )
            else:
                h_vec_loop[:, :, :, [j]] = (
                    (
                        config['window_size']
                        * torch.std(x[:, :, [j]], dim=1, keepdim=True)
                        * N ** (-1 / 5)
                    )
                    .unsqueeze(dim=1)
                    .expand(-1, N, -1, -1)
                )
        h_star = (
            config['window_size_star'] * torch.std(x_star_gen, dim=1) * N ** (-1 / 5)
        )  # batch_size*1
        h_star_loop = h_star.unsqueeze(dim=1).expand(
            -1, N, -1
        )  # batch_size*N*1 batch,loop

        arg1_loop = f_joint(x, x_star_gen, h_vec_loop, h_star_loop)  # batch*loop

        arg2_loop = torch.ones(batch_size, N, device=config['device'], dtype=torch.float64)  # batch_size * N
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
        
        if config['loss_fun'] == 'dl_hellinger':
            hellinger_distance = torch.square(torch.sqrt(arg1_loop) - torch.sqrt(arg2_loop))
            divergence_loss = torch.mean(hellinger_distance, dim=1)
        elif config['loss_fun'] == 'kl_loss':
            # KL_divergence = torch.abs(arg1_loop*(arg1_loop_log-arg2_loop_log))   ## PyTorch and Tensorflow implementation
            arg1_loop_log = torch.log(arg1_loop)
            arg2_loop_log = torch.log(arg2_loop)
            KL_divergence = arg1_loop * (arg1_loop_log - arg2_loop_log)
            divergence_loss = torch.mean(KL_divergence, dim=1)
        else:
            arg1_loop_log = torch.log(arg1_loop)
            arg2_loop_log = torch.log(arg2_loop)
            arg1 = torch.mean(arg1_loop_log, dim=1)  ##take as sample average
            arg2 = torch.mean(arg2_loop_log, dim=1)
            divergence_loss = torch.abs(arg1 - arg2)
            # divergence_loss = arg1 - arg2

        return divergence_loss

    return loss(x, x_star_gen)

# COMMAND ----------

def train_GEEN(config, verbose=False):
    def train_one_epoch(epoch, model, train_dataloader, optimizer):
        model.train()
        train_loss = 0

        for batch_idx, batch in enumerate(train_dataloader):
            if config["with_xstar"]:
                x, x_star = batch
                x = x.to(config["device"]).double()
                x_star = x_star.to(config["device"]).double()
            else:
                x, x_index = batch
                x = x.to(config["device"]).double()

            optimizer.zero_grad()
            x_star_gen = model(x)

            if config["is_normal"]:
                divergence_loss, normalization = divergence_loss_normal(
                    x, x_star_gen, config
                )
            else:
                divergence_loss = divergence_loss_wo_normal(x, x_star_gen, config)
                normalization = 0

            loss = torch.mean(divergence_loss + normalization)
            loss.backward()
            train_loss += loss.item()
            
            if config["clip"] > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), config["clip"])
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
                if config["with_xstar"]:
                    x, x_star = batch
                    x = x.to(config["device"]).double()
                    x_star = x_star.to(config["device"]).double()
                else:
                    x, x_index = batch
                    x = x.to(config["device"]).double()
                x_star_gen = model(x)
                if config["is_normal"]:
                    divergence_loss, normalization = divergence_loss_normal(
                        x, x_star_gen, config
                    )
                else:
                    divergence_loss = divergence_loss_wo_normal(x, x_star_gen, config)
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
    ) = get_data(config["N_sample"], config)

    genModel = GenDNN(config).to(config["device"]).double()
    optimizer = optim.Adam(genModel.parameters(), lr=config["learning_rate"])

    patience = config["patience"]
    best_val_loss = np.inf
    epoch_no = 0
    for epoch in range(config["epochs"]):
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
                    torch.tensor(x_test, device=config["device"]).view(
                        -1, 1, config["input_size"]
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
            torch.tensor(x_train, device=config["device"], dtype=torch.float64).view(
                -1, 1, config["input_size"]
            )
        )
        .flatten()
        .detach()
        .cpu()
        .numpy()
        )
    x_gen_val = (
        genModel(
            torch.tensor(x_val, device=config["device"], dtype=torch.float64).view(
                -1, 1, config["input_size"]
            )
        )
        .flatten()
        .detach()
        .cpu()
        .numpy()
        )
    x_gen_test = (
        genModel(
            torch.tensor(x_test, device=config["device"], dtype=torch.float64).view(
                -1, 1, config["input_size"]
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
    config["lambda"] = 0.1
    config["window_size"] = float(1 * 1.06)
    config["window_size_star"] = float(1 * 1.06)
    train_GEEN(config, verbose=True)

# COMMAND ----------

def train_GEEN_whyperopt(params, init_config):
    config = copy.deepcopy(init_config)
    if config['is_normal']:
        lm = params["lambda"]
    else:
        lm = 0
    window_size = params["window_size"]
    config["lambda"] = float(lm)
    config["window_size"] = float(window_size * 1.06)
    config["window_size_star"] = float(window_size * 1.06)

    min_loss = np.inf
    min_mse = np.inf
    max_corr = -np.inf

    with mlflow.start_run(nested=True):
        for i in range(config['number_simulation']):
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

 if init_config['is_normal']:
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
