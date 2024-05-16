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

from src.GEEN import *

def train_one_epoch(config, epoch, model, train_dataloader, optimizer, verbose):
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
        
        if verbose:
            if batch_idx % 1000 == 0:
                print(
                    "Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6e}".format(
                        epoch,
                        batch_idx * x.shape[0],
                        len(train_dataloader.dataset),
                        100.0 * batch_idx / len(train_dataloader),
                        loss.item() / x.shape[0],
                    )
                )
    return train_loss / len(train_dataloader.dataset)

def test_one_epoch(config, model, test_dataloader, test_dataset=None):
    '''
    test_dataset: tuple of x_test, x_star_test; used to monitor mse and corr for simulation data
    '''
    
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
        
        if config.with_xstar and test_dataset is not None:
            x_gen_test = (
                model(
                    torch.tensor(test_dataset[0], device=config.device).view(
                        -1, 1, config.num_measurement
                    )
                )
                .flatten()
                .detach()
                .cpu()
                .numpy()
            )   
            mse = mean_squared_error(test_dataset[1], x_gen_test, squared=False)
            corr = np.corrcoef(test_dataset[1], x_gen_test)[0, 1]
        else:
            mse = np.nan
            corr = np.nan
        
    return test_loss / len(test_dataloader.dataset), mse, corr

def train_and_val_GEEN(config, model, train_dataloader, val_dataloader, test_dataloader, test_dataset=None, verbose=False):
    
    optimizer = optim.Adam(model.parameters(), lr=config.learning_rate)
    patience = config.patience
    best_val_loss = np.inf
    
    for epoch in range(config.epochs):
        train_loss = train_one_epoch(
            config, epoch, model, train_dataloader, optimizer, verbose
        )
        val_loss, _, _ = test_one_epoch(epoch, model, val_dataloader)
        test_loss, mse, corr = test_one_epoch(epoch, model, test_dataloader, test_dataset)
        if best_val_loss > val_loss:
            best_val_loss = val_loss
        else:
            patience -= 1

        if patience < 0:
            break
            
        if verbose and test_dataset is not None:
            print(
                "====> Epoch: {} Test corr: {:.4f} Test mse: {:.4E} Test loss: {:.4E}".format(
                    epoch, corr, mse, test_loss
                )
            )
        else:
            print(
                "====> Epoch: {} Test loss: {:.4E}".format(epoch, test_loss)
            )

    return model, test_loss