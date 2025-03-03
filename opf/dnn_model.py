import torch
import torch.nn as nn

class DNN(nn.Module):
    def __init__(self, shape):
        super(DNN, self).__init__()
        layers = []
        for idx in range(len(shape) - 2):
            layers.extend([
                nn.Linear(shape[idx], shape[idx+1]),
                nn.BatchNorm1d(num_features=shape[idx+1]),  # 確保 num_features 正確
                nn.ReLU(),
                nn.Dropout(0.5),
            ])
        layers.append(nn.Linear(shape[-2], shape[-1]))  # 最後一層
        self.features = nn.Sequential(*layers)

    def forward(self, x):
        print(f"input shape: {x.shape}")
        x = x.view(x.size(0), -1)  # 保證 batch_size 在第一維
        return self.features(x)
