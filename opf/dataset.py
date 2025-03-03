import torch
class Dataset(torch.utils.data.Dataset):
    def __init__(self,x,y):
        self.x=torch.from_numpy(x).float()
        self.y=torch.from_numpy(y).float()
    def __len__(self):
        return len(self.x)
    def __getitem__(self, idx):
        if torch.is_tensor(idx):
          idx=idx.tolist()
        return self.x[idx],self.y[idx]