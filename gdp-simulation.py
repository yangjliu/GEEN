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
from src.gdpDataset import get_GDPdata
from src.train_validation import train_and_val_GEEN

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
               'patience':3,
               'is_normal':True,
               'with_xstar': False,
               'batch_size': 64,
               'num_measurement': 4,
               'sample_size': 200,
               'num_layer': 6,
               'hidden_size': 6,
               'learning_rate': 0.05,
               'number_simulation':25,
               'gdpFileName': './data/data_date_FE_removed_allsample.csv',
               'feature_columns': ["qGDP_growth_res", "dNL_res", "dhits_res"],
               'fixed_effect': True,
               'country_inTraining': True,
               'dir': '/dbfs/mnt/prod/GEEN'
               }
#init_config = SimpleNamespace(**init_config)

# COMMAND ----------
try:
    loss_fun = dbutils.widgets.get("loss_fun")
except:
    loss_fun = 'kl_loss_from_avg'
try:
    batch_norm = dbutils.widgets.get("batch_norm") == 'True'
except:
    batch_norm = False
try:
    window_size = float(dbutils.widgets.get("window_size"))
except:
    window_size = 1.1247613400258931
try:
    lm = float(dbutils.widgets.get("lambda"))
except:
    lm = 0.9491195387601212

# COMMAND ----------    
def gdpSimulation(config):
    gdp_data = pd.read_csv(config.gdpFileName).dropna(how='any')
    gdp_data['index'] = gdp_data.index
    
    if config.country_inTraining:
        gdp_train = gdp_data
        gdp_test = gdp_data
        
    
    