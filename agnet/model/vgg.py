import torch
import torch.nn as nn

# VGGNet Configuration
cfg = {
    'VGG11': [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'VGG13': [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'VGG16': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'],
    'VGG19': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M'],
}

class VGG(nn.Module):
    def __init__(self, vgg_name, output_dim=2,image_size=299):
        super(VGG, self).__init__()
        self.image_size=image_size
        self.features = self._make_layers(cfg[vgg_name])
        self.feat_dim = self._get_feature_dim()
        self.classifier = nn.Sequential(
            nn.Linear(self.feat_dim[1] * self.feat_dim[2] * self.feat_dim[3], 4096),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(4096, output_dim)
        )

    def _get_feature_dim(self):
        x = torch.randn(1,3,self.image_size,self.image_size)
        return self.features(x).shape

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x

    def _make_layers(self, cfg):
        layers = []
        in_channels = 3
        for x in cfg:
            if x == 'M':
                layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
            else:
                layers += [nn.Conv2d(in_channels, x, kernel_size=3, padding=1),
                           nn.ReLU(inplace=True)]
                in_channels = x
        layers += [nn.AvgPool2d(kernel_size=1, stride=1)]
        return nn.Sequential(*layers)

if __name__ == '__main__':
    # Creating an instance of VGGNet
    image_size = 256
    model = VGG('VGG16', image_size=image_size)

    # Printing the model architecture
    # print(model)
    r = torch.randn(1,3, image_size, image_size)
    print(model.features(r).shape)
    print(model(r).shape)
