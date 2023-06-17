import argparse
import yaml
import warnings
warnings.filterwarnings('ignore')

import torch
import torch.nn as nn
import torchvision
import torchmetrics

from data import prep_dataloader
from model import Trainer, AGNet, VGG

class AGTrainer(Trainer):
    def __init__(self, model, device, fp='fp32', model_version="v1", **kwargs):
        self.model = model.to(device)
        self.lr = kwargs['lr']
        self.output_dim = kwargs['output_dim']
        optim = self.get_optimier()
        loss_fn = self.get_loss_fn()
        metric = self.get_metric(device)
        scheduler = torch.optim.lr_scheduler.StepLR(optim, step_size=kwargs.get('scheduler_step_size',9), gamma=kwargs.get('scheduler_gamma',0.6))
        super().__init__(
            model, 
            device, fp=fp, 
            model_version=model_version,
            optim=optim,
            loss_fn=loss_fn,
            metric=metric,
            scheduler=scheduler,
            **kwargs

        )
    
    def get_optimier(self):
        optimizer = torch.optim.Adam(filter(lambda x: x.requires_grad, self.model.parameters()), lr=self.lr)
        # print('')
        # optimizer = torch.optim.Adagrad(self.model.parameters(), lr=self.lr)
        return optimizer
    
    def get_loss_fn(self):
        if self.output_dim>1:
            return nn.CrossEntropyLoss()
        else:
            return nn.BCEWithLogitsLoss()

    def get_metric(self, device):
        # return None
        return torchmetrics.Accuracy(task="binary").to(device)

def argparser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config-file",'-c', dest="config_file", required=True)
    parser.add_argument("--model-version",'-vv', dest="model_version", default=1)
    parser.add_argument("--file-path",'-fp',dest='file_path', default=None)
    parser.add_argument("--image_base_path",'-ip', dest="image_base_path", default=None)
    parser.add_argument("--device",'-d',dest="device",default="cpu",choices=['cpu','cuda'])
    return parser.parse_args()

def run(args):
    
    config = yaml.load(open(args.config_file,'r'))
    print(config)
    if args.file_path:
        config['data']['file_path'] = args.file_path
    if args.image_base_path:
        config['data']['image_base_path'] = args.image_base_path
    
    train_dl, val_dl = prep_dataloader(config['data']['file_path'], config['data'])
    
    if config['model']['_base_model'] == '_vgg':
        base_model = VGG("VGG19", output_dim=1, image_size=config['data']['image_size'])
    elif config['model']['_base_model'] == 'vgg':
        raise NotImplementedError()
    else:
        print("base_model")
        base_model = getattr(torchvision.models, config['model']['_base_model'])(pretrained=True)

    model = AGNet(base_model,  **config['model'])
    print(model)
    print(model(torch.randn(1,3,config['data']['image_size'],config['data']['image_size'])).shape)
    device = torch.device(args.device)

    trainer = AGTrainer(model, device, **config['model'])
    trainer.fit(train_dl, val_dl, epochs=config['model']['epochs'])
    
if __name__ == "__main__":
    args = argparser()
    run(args)