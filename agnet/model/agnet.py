import torch
import torch.nn as nn


class AGNet(nn.Module):
    def __init__(self, base_model, output_dim):
        super().__init__()
        self.base_model = base_model.features
        feat_dim = base_model.feat_dim
        self.fc = nn.Sequential(*[
            nn.Linear(feat_dim[1] * feat_dim[2] * feat_dim[3], 4096),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(4096, 1204),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(1204, output_dim),
            nn.ReLU()
        ])
    
    def forward(self, x):
        b = x.shape[0]
        x = self.base_model(x)
        x = x.view(b,-1)
        x = self.fc(x)
        return x

