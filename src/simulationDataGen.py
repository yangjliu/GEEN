from torch.utils.data import DataLoader, Dataset

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