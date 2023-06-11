import torch
import torch.nn as nn


resnet_output_mapping = {
    'resnet34':{
        'fc':'fc',
        'dim': 512,
        'rfc':[
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
            nn.ReLU()
        ],
        'cfc':[
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 100),
        ]
    },
    'resnet50':{
        'fc':'fc',
        'dim': 2048,
        'rfc': [
            nn.Linear(2048, 1024),
            nn.ReLU(),
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Linear(512, 128),
            nn.ReLU(),
            nn.Linear(128,32),
            nn.ReLU(),
            nn.Linear(32, 1),
            nn.ReLU()
        ],
        'cfc':[
            nn.Linear(2048, 1024),
            nn.ReLU(),
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Linear(512, 128),
            nn.ReLU(),
            nn.Linear(128,100)
        ]
    },
    'efficientnet_b5':{
        'fc':'classifier',
        'rfc':[
            nn.Linear(2048, 1024),
            nn.ReLU(),
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Linear(512, 128),
            nn.ReLU(),
            nn.Linear(128,32),
            nn.ReLU(),
            nn.Linear(32,1),
            nn.ReLU()
        ]
    }
}

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
        else:
            
            # freeze_layers = False
            # if freeze_layers:
            #     for name, param in base_model.named_parameters():
            #         if 'fc' not in name:
            #             print(name)
            #             param.requires_grad = False
            self.base_model = base_model
            if "resnet" in base_model_name:
                self.base_model.fc = nn.Sequential(*resnet_output_mapping[base_model_name]['rfc'])
            elif "efficient" in base_model_name:
                self.base_model.classifier = nn.Sequential(*resnet_output_mapping[base_model_name]['rfc'])
            else:
                raise ValueError()

    
    def forward(self, x):
        if self.base_model_name == '_vgg':
            b = x.shape[0]
            x = self.base_model(x)
            x = x.view(b,-1)
            x = self.fc(x)
            return x
        else:
            return self.base_model(x)

