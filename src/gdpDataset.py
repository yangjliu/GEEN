import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset
from sklearn.model_selection import train_test_split

class GDPDataset(Dataset):
    def __init__(self, data_index, data):
        ## Fetch dataset
        self.data_index = data_index
        self.data = data
        self.len = self.data.shape[0]
    def __len__(self):
        return self.len
    def __getitem__(self, idx):
        return self.data[idx], self.data_index[idx]

def get_GDPdata(gdp_data, config, random_seed=0):
    np.random.seed(random_seed)
        
    train_data, test_data = train_test_split(gdp_data, test_size=0.2)
    
    n_train, n_test = train_data.shape[0], test_data.shape[0]
    sample_train = np.random.choice(n_train, size=(n_train, config.sample_size), replace=True, p=None)
    sample_test = np.random.choice(n_test, size=(n_test, config.sample_size), replace=True, p=None)
    x_train_resampled = np.zeros((n_train, config.sample_size, len(config.feature_columns),))
    x_test_resampled = np.zeros((n_test, config.sample_size, len(config.feature_columns),))

    index_train_resampled = np.zeros((n_train, config.sample_size))
    index_test_resampled = np.zeros((n_test, config.sample_size))
    
    x_train_resampled = train_data[config.feature_columns].values[sample_train, :]
    x_test_resampled = test_data[config.feature_columns].values[sample_test, :]
    
    index_train_resampled = train_data['index'].values[sample_train, :]
    index_test_resampled = test_data['index'].values[sample_test, :]
    
    train_dataset = GDPDataset(index_train_resampled, x_train_resampled)
    test_dataset = GDPDataset(index_test_resampled, x_test_resampled)
    
    train_dataloader = DataLoader(
            dataset=train_dataset,
            batch_size=config['batch_size'],
            shuffle=True)
    test_dataloader = DataLoader(
            dataset=test_dataset,
            batch_size=config['batch_size'],
            shuffle=False)

    return train_dataloader, test_dataloader,\
           train_data, test_data