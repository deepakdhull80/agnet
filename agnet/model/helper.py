import os
import torch
from tqdm import tqdm

class Trainer:
    """
    ## Trainer

    abstract function need to be defined
        get_optimizer
        get_loss_fn
        get_metric [Optional]
        predict [optional] 
    """
    def __init__(self, model, device, fp='fp16', model_version="v1", **kwargs):
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
    
    def train_step(self, dataloader, epoch):
        if self.tqdm_enable:
            tqdm_iter = tqdm(enumerate(dataloader), total=len(dataloader))
        else:
            tqdm_iter = enumerate(dataloader)
        total_loss = 0
        total_score = 0
        self.model = self.model.train(True)
        for indx, batch in tqdm_iter:
            self.optim.zero_grad()
            if self.fp == 'fp16':
                raise NotImplementedError()
                x, y = batch[0].to(self.device).half(), batch[1].to(self.device)
            else:
                x, y = batch[0].float().to(self.device), batch[1].unsqueeze(1).float().to(self.device)
            y_h = self.model(x)
            loss = self.loss_fn(y_h,y)
            loss.backward()
            self.optim.step()

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
            else:
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
            if self.fp == 'fp16':
                x, y = batch[0].to(self.device).half(), batch[1].to(self.device)
            else:
                x, y = batch[0].to(self.device), batch[1].to(self.device)
            y_h = self.model(x)
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
            else:
                print(desc)
        
        return total_loss/len(dataloader), total_score/len(dataloader)
    
    def fit(self, dataloader, val_dataloader=None, epochs=120):

        global_loss = 1e3
        for epoch in range(epochs):
            print("*"*20)
            print(f"EPOCH: {epoch} started")
            self.train_step(dataloader, epoch)
            if val_dataloader is not None:
                v_loss, v_score = self.val_step(val_dataloader, epoch)
                if global_loss > v_loss:
                    self.save_weights(epoch, v_loss, v_score)
                    print(">>>model weight saved<<<")
    
    def save_weights(self, epoch, loss, score):
        os.makedirs("cp", exist_ok=True)
        torch.save(
            {
                "state_dict": self.model.state_dict(),
                "epoch":epoch,
                "loss":loss,
                "score":score
            },
            f"cp/model_{self.model_version}_{epoch}.pt"
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