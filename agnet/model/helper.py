import os
import torch
from tqdm import tqdm
from typing import *

from torch.cuda.amp import autocast, GradScaler

class Trainer:
    """
    ## Trainer

    abstract function need to be defined
        get_optimizer
        get_loss_fn
        get_metric [Optional]
        predict [optional] 
    """
    def __init__(self, model: torch.nn.Module, device: torch.device, fp='fp16', model_version="v1", **kwargs):
        self.model_version = model_version
        self.model = model.to(device)
        self.device = device
        if fp not in ['fp16', 'fp32']:
            raise ValueError(f"fp either fp16 or fp32 but found {fp}")
        self.fp = fp
        self.loss_fn = kwargs.get("loss_fn")
        self.optim = kwargs.get('optim')
        self.metric = kwargs.get('metric')
        self.tqdm_enable = kwargs.get("tqdm_enable",True)
        self.output_dim = kwargs.get("output_dim", 1)
        self.model_save_path = kwargs.get('model_save_path','./')
        self.load_last_checkpoint = kwargs.get('load_last_checkpoint', True)
        self.scheduler: Optional[torch.optim.lr_scheduler.StepLR] = kwargs.get('scheduler',None)

        _last_checkpoint_path = os.path.join(self.model_save_path, 'cp')
        self._epoch = 0
        print(_last_checkpoint_path)
        if os.path.exists(_last_checkpoint_path):
            _paths = sorted(os.listdir(_last_checkpoint_path), reverse=True)
            if self.load_last_checkpoint and len(_paths) > 0 and os.path.exists(os.path.join(_last_checkpoint_path,_paths[0])):
                # load last checkpoint of the model for continuing the training
                model_path = os.path.join(_last_checkpoint_path,_paths[0])
                _save_dict = torch.load(model_path)
                print(f"path: {model_path}, epochs= {_save_dict['epoch']}, loss={_save_dict['loss']}")
                print(self.model.load_state_dict(_save_dict['state_dict']))
                self._epoch = _save_dict['epoch']
    
    def train_step(self, dataloader, epoch):
        if self.tqdm_enable:
            tqdm_iter = tqdm(enumerate(dataloader), total=len(dataloader))
        else:
            tqdm_iter = enumerate(dataloader)
        total_loss = 0
        total_score = 0
        scaler = GradScaler()
        self.model = self.model.train(True)
        for indx, batch in tqdm_iter:
            self.optim.zero_grad()
            if self.fp == 'fp16':
                # raise NotImplementedError()
                with autocast():
                    x, y = batch[0].to(self.device), batch[1].to(self.device)
                    y_h = self.model(x)
            else:
                x, y = batch[0].float().to(self.device), batch[1].view(-1, self.output_dim).float().to(self.device)
                y_h = self.model(x)
            if y_h.shape[-1]>1:
                y_h = torch.softmax(y_h,dim=-1)
            
            loss = self.loss_fn(y_h,y)
            if self.fp == 'fp32':
                loss.backward()
                self.optim.step()
            else:
                scaler.scale(loss).backward()
                scaler.step(self.optim)
                scaler.update()

            loss = loss.detach().item()
            total_loss += loss
            desc = f"(Train) epoch={epoch} batch={indx} loss={loss:.3f}, avg_loss={total_loss/(1+indx): .3f}"
            
            if self.metric is not None:
                with torch.no_grad():
                    name, score = self.metric(y_h.detach(), y)
                assert isinstance(score, torch.Tensor), f"metric score should be tensor but found {type(score)}"
                total_score += score.item()
                desc += f", {name}={score.item():.3f}, avg_{name}:{total_score/(1+indx):.3f}"
            if self.tqdm_enable:
                tqdm_iter.set_description(desc)
            
        print(desc)
        return total_loss/len(dataloader), total_score/len(dataloader)
    
    @torch.no_grad()
    def val_step(self, dataloader, epoch):
        if self.tqdm_enable:
            tqdm_iter = tqdm(enumerate(dataloader), total=len(dataloader))
        else:
            tqdm_iter = enumerate(dataloader)
        total_loss = 0
        total_score = 0
        self.model = self.model.eval()
        for indx, batch in tqdm_iter:
            
            x, y = batch[0].to(self.device), batch[1].view(-1, self.output_dim).float().to(self.device)
            y_h = self.model(x)
            if y_h.shape[-1]>1:
                y_h = torch.softmax(y_h,dim=-1)
            loss = self.loss_fn(y_h,y)
            
            loss = loss.item()
            total_loss += loss
            desc = f"(Eval) epoch={epoch} batch={indx} loss={loss:.3f}, avg_loss={total_loss/(1+indx): .3f}"
            
            if self.metric is not None:
                name, score = self.metric(y_h, y)
                assert isinstance(score, torch.Tensor), f"metric score should be tensor but found {type(score)}"
                total_score += score.item()
                desc += f", {name}={score.item():.3f}, avg_{name}:{total_score/(1+indx):.3f}"
            if self.tqdm_enable:
                tqdm_iter.set_description(desc)
            elif indx % 200 == 0:
                print(desc)
        print(desc)
        
        return total_loss/len(dataloader), total_score/len(dataloader)
    
    def fit(self, dataloader, val_dataloader=None, epochs=120):

        self.global_loss = 1e3
        for epoch in range(self._epoch, epochs):
            print("*"*20)
            print(f"EPOCH: {epoch} started")
            self.train_step(dataloader, epoch)
            if val_dataloader is not None:
                v_loss, v_score = self.val_step(val_dataloader, epoch)
                if self.global_loss > v_loss:
                    self.global_loss = v_loss
                    self.save_weights(epoch, v_loss, v_score)
                    print(">>>model weight saved<<<")
                self.scheduler.step()
    
    def save_weights(self, epoch, loss, score):
        sv_path = os.path.join(self.model_save_path, "cp")
        os.makedirs(sv_path, exist_ok=True)
        torch.save(
            {
                "state_dict": self.model.state_dict(),
                "epoch":epoch,
                "loss":loss,
                "score":score
            },
            f"{sv_path}/model_{self.model_version}_{epoch}.pt"
        )

    def load_model_weights(self, model_path):
        model_dict = torch.load(model_path)
        print(self.model.load_state_dict(model_dict['state_dict']))
        
        print(f"""
                Model loss: {model_dict['loss']}
                Model score: {model_dict['score']}
                Epoch: {model_dict['epoch']}
              """)

    @torch.no_grad()
    def predict(self, x, y):
        raise NotImplementedError()

    def get_optimizer(self):
        raise NotImplementedError()
    
    def get_loss_fn(self):
        raise NotImplementedError()
    
    def get_metric(self):
        """
        metric
            input: y_h, y
            return metric should return [metric_name, score]
        """
        raise NotImplementedError()

class Predictor:
    pass