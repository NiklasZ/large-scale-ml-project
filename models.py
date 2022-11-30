from torch import nn
from torchvision.models import resnet18


# These are standard models that we should not alter after adding them.
# Otherwise we have to re-run all experiments using them.
# Model source: https://pytorch.org/vision/master/models.html

class ResNet18MNIST(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = resnet18(num_classes=10)  # Use ResNet18
        self.model.conv1 = nn.Conv2d(1, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3),
                                     bias=False)  # Fit first layer to shape of MNIST data

    def forward(self, x):
        return self.model(x)


class ResNet18CIFAR10(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = resnet18(num_classes=10)  # Use ResNet18

    def forward(self, x):
        return self.model(x)


# Dict of known models
MODELS = {
    ResNet18MNIST.__name__: ResNet18MNIST,
    ResNet18CIFAR10.__name__: ResNet18CIFAR10
}
