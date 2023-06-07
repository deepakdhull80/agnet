import argparse
import yaml

import torch
import torch.nn as nn
import torchmetrics

from data import prep_dataloader
from model import Trainer, AGNet, VGG

class AGTrainer(Trainer):
    def __init__(self, model, device, fp='fp32', model_version="v1", **kwargs):
        self.model = model.to(device)
        self.lr = kwargs['lr']
        
        optim = self.get_optimier()
        loss_fn = self.get_loss_fn()
        metric = None
        super().__init__(
            model, 
            device, fp=fp, 
            model_version=model_version,
            optim=optim,
            loss_fn=loss_fn,
            metric=metric
        )
    
    def get_optimier(self):
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)
        return optimizer
    
    def get_loss_fn(self):
        return nn.MSELoss()


def argparser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config-file",'-c', dest="config_file", required=True)
    parser.add_argument("--model-version",'-vv', dest="model_version", default=1)
    parser.add_argument("--device",'-d',dest="device",default="cpu",choices=['cpu','cuda'])
    return parser.parse_args()

def run(args):
    
    config = yaml.load(open(args.config_file,'r'))
    print(config)
    
    train_dl, val_dl = prep_dataloader(config['data']['file_path'], config['data'])
    base_model = VGG("VGG19", output_dim=1, image_size=config['data']['image_size'])
    model = AGNet(base_model, output_dim=1)
    device = torch.device(args.device)

    trainer = AGTrainer(model, device, **config['model'])

    trainer.fit(train_dl, val_dl, epochs=config['model']['epochs'])
    
if __name__ == "__main__":
    args = argparser()
    run(args)