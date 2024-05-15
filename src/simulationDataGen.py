from torch.utils.data import DataLoader, Dataset
import numpy as np
from scipy.stats import norm

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

def get_simulation_data(config, random_seed=0):
    '''
    eps_1: normal distribution
    eps_2: beta distribution - alpha and beta
    eps_3: laplace distribution - loc and scale
    eps_4: uniform - a and b
    '''
    # set seed
    np.random.seed(random_seed)
    N_sample = config.N_sample
    num_measurement = 4

    if config.discrete_xstar:
        x_star = np.random.multinomial(10, [0.5]*2, N_sample)[:,0]
    else:
        x_star = np.random.normal(config.mu_star, config.sigma_star, N_sample)     

    # linear error experiment in the paper
    if config.linear_error:
        mu_1, scale_1 = 0, 0.5
        alpha, beta = 2, 2
        loc, scale = 0, 0.5
        a, scale_b = 0, 0.5 

        scale_1, beta, scale, scale_b = (
            config.error_scale*scale_1, 
            config.error_scale*beta, 
            config.error_scale*scale, 
            config.error_scale*scale_b
            )

        eps_1 = np.random.normal(mu_1, scale_1*np.abs(x_star)) 
        eps_2 = np.random.beta(alpha, beta, N_sample)-1/(1+beta/alpha)
        eps_3 = np.random.laplace(loc, scale*np.abs(x_star), N_sample)-loc
        eps_4 = np.random.uniform(a, scale_b*np.abs(x_star))-(a+scale_b*np.abs(x_star))/2
        
    else:
        mu_1, sigma_1 = 0, 1
        alpha, beta = 2, 2
        loc, scale = 0, 1
        a,b = 0, 1

        sigma_1, beta, scale, b = (
            config.error_scale*sigma_1, 
            config.error_scale*beta, 
            config.error_scale*scale, 
            config.error_scale*b
            )
            
        eps_1 = np.random.normal(mu_1, sigma_1, N_sample)  
        eps_2 = np.random.beta(alpha, beta, N_sample)-1/(1+beta/alpha)
        eps_3 = np.random.laplace(loc, scale, N_sample)-loc
        eps_4 = np.random.uniform(a, b, N_sample)-(a+b)/2

    x_star_vec = x_star

    if config.is_normal:
        x1 = m1(x_star)+eps_1
    else:
        x1 = m1_wonormal(x_star)+eps_1
        
    x2 = m2(x_star)+eps_2
    x3 = m3(x_star)+eps_3
    
    if config.x4_distinct:
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
    sample_train = np.random.choice(
        n_train, 
        size=(n_train, config.sample_size), 
        replace=True, 
        p=None)
    sample_val = np.random.choice(
        n_val, 
        size=(n_val, config.sample_size), 
        replace=True, 
        p=None)
    sample_test = np.random.choice(
        n_test, 
        size=(n_test, config.sample_size), 
        replace=True, 
        p=None)

    x_train_resampled = np.zeros((n_train, config.sample_size, num_measurement,))
    x_val_resampled = np.zeros((n_val, config.sample_size, num_measurement,))
    x_test_resampled = np.zeros((n_test, config.sample_size, num_measurement,))

    x_star_train_resampled = np.zeros((n_train, config.sample_size))
    x_star_val_resampled = np.zeros((n_val, config.sample_size))
    x_star_test_resampled = np.zeros((n_test, config.sample_size))

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
            batch_size=config.batch_size,
            shuffle=True)
    val_dataloader = DataLoader(
            dataset=val_dataset,
            batch_size=config.batch_size,
            shuffle=False)
    test_dataloader = DataLoader(
            dataset=test_dataset,
            batch_size=config.batch_size,
            shuffle=False)

    return train_dataloader, val_dataloader, test_dataloader,\
           x_train, x_star_train, \
           x_val, x_star_val,\
           x_test, x_star_test