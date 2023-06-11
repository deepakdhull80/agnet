import torch
import torch.nn as nn


class AGNet(nn.Module):
    def __init__(self, base_model, output_dim, base_model_name='_vgg'):
        super().__init__()
        self.base_model_name = base_model_name
        if base_model_name == '_vgg':
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
        elif base_model_name == 'resnet34':
            
            ### freeze cnn layers
            freeze_layers = False
            if freeze_layers:
                for name, param in base_model.named_parameters():
                #     print(name)
                    if 'fc' not in name:
                        print(name)
                        param.requires_grad = False
            self.base_model = base_model
            self.base_model.fc = nn.Sequential(*[
                nn.Linear(512, output_dim)
            ])

    
    def forward(self, x):
        if self.base_model_name == '_vgg':
            b = x.shape[0]
            x = self.base_model(x)
            x = x.view(b,-1)
            x = self.fc(x)
            return x
        elif self.base_model_name == "resnet34":
            return self.base_model(x)

