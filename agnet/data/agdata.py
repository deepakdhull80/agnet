from PIL import Image
from typing import Any
import torch
import torchvision
from torch.utils.data import Dataset

class AGDataset(Dataset):
    def __init__(self, df, base_path, target_field=['age','gender'], **kwargs):
        self.df = df
        self._min = 1
        self._max = 100
        
        self.base_path = base_path
        self.target_field = target_field
        self.image_size = kwargs.get("image_size", 256)
        self.scale = kwargs.get("scale_factor", 10)

        self.transforms1 = torchvision.transforms.Compose([
            torchvision.transforms.PILToTensor(),
            torchvision.transforms.ConvertImageDtype(torch.float)
        ])
        self.transforms2 = torchvision.transforms.Compose([
            torchvision.transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

    def age_scale(self,x):
        return self.scale * (x-self._min)/(self._max - self._min)
    
    def __getitem__(self, index: Any) -> Any:
        row = self.df.iloc[index]
        image = Image.open(f"{self.base_path}/{row['file_path']}")
        image = image.resize((self.image_size,self.image_size))
        
        image = self.transforms1(image)
        if image.shape[0] == 1:
            # grey scale
            image = torch.concat([image, image, image], axis=0)
        image = image/255.
        # image = self.transforms2(image)
        assert isinstance(image, torch.Tensor), f"image dtype should be torch.Tensor but found {type(image)}"
        target = self.age_scale(row['age'])
        return image, target
    
    def __len__(self):
        return self.df.shape[0]
