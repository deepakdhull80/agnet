import torch
import torch.nn as nn


resnet_output_mapping = {
    'resnet34':{
        'fc':'fc',
        'dim': 512,
        'age':[
            nn.Linear(512, 1),
            nn.ReLU()
        ],
        'gender':[
            nn.Linear(512, 1)
        ]
    },
    'resnet50':{
        'fc':'fc',
        'dim': 2048,
        'age': [
             nn.Linear(2048, 1024),
            nn.ReLU(),
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Linear(512, 128),
            nn.ReLU(),
            nn.Linear(128,32),
            nn.ReLU(),
            nn.Linear(32, 10),
            nn.ReLU()
        ],
        'gender':[
            nn.Linear(2048, 512),
            nn.ReLU(),
            nn.Linear(512, 128),
            nn.ReLU(),
            nn.Linear(128,1)
        ]
    },
    'efficientnet_b5':{
        'fc':'classifier',
        'age':[
            nn.Linear(2048, 1024),
            nn.ReLU(),
            nn.Linear(1024, 10)
        ],
        'gender':[
            nn.Linear(2048, 1024),
            nn.ReLU(),
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Linear(512, 1)
        ]
    }
}

class AGNet(nn.Module):
    def __init__(self, base_model, **kwargs):
        super().__init__()
        self._base_model = kwargs.get("_base_model",'resnet34')
        self.estimator = kwargs.get("estimator",'gender')
        
        print(kwargs['mlp_layer_name'], kwargs['_base_model'])
        
        if kwargs['transfer_learning']:
            for name, param in base_model.named_parameters():
                if kwargs['mlp_layer_name'] not in name:
                    print(name)
                    param.requires_grad = False
        self.base_model = base_model

        if "efficient" in self._base_model:
            print("***EfficientNet MLP***")
            self.base_model.classifier = nn.Sequential(*resnet_output_mapping[self._base_model][self.estimator])
        elif "resnet" in self._base_model:
            self.base_model.fc = nn.Sequential(*resnet_output_mapping[self._base_model][self.estimator])
        else:
            raise ValueError()

    
    def forward(self, x):
        return self.base_model(x)

