from data import prep_dataloader

from torchvision import transforms
import yaml
import os
image_size = 256
config = yaml.load(open(os.path.join(os.path.dirname(__file__),"config.yaml"),"r"))
print(config)

def test_data():
    transform = transforms.Compose([
        transforms.PILToTensor(),
        transforms.Resize((image_size, image_size)),
        transforms.Lambda(lambda x: x/255.0)
    ])
    gen_dl = prep_dataloader(
        "", transform, config
    )
    print(gen_dl)

if __name__ == '__main__':
    test_data()