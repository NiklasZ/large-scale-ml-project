from typing import Tuple, Any
from torchvision.datasets import MNIST, CIFAR10
from torchvision.transforms import ToTensor


# Contains all datasets to train on

class IndexedMNIST(MNIST):
    def __getitem__(self, index: int) -> Tuple[Any, Any, int]:
        img, target = super().__getitem__(index)
        return img, target, index


class IndexedCIFAR10(CIFAR10):
    def __getitem__(self, index: int) -> Tuple[Any, Any, int]:
        img, target = super().__getitem__(index)
        return img, target, index


MNIST_config = ['data/mnist', {'train': True, 'download': True, 'transform': ToTensor()}]
CIFAR10_config = ['data/cifar10', {'train': True, 'download': True,
                                   'transform': ToTensor()}]  # TODO consider normalisation or augmentation for transform (https://pytorch.org/tutorials/beginner/blitz/cifar10_tutorial.html)

# First dataset is for training, the second for evaluation, third the constructor args for both datasets
DATASETS = {
    MNIST.__name__: (IndexedMNIST, MNIST, MNIST_config),
    CIFAR10.__name__: (IndexedCIFAR10, CIFAR10, CIFAR10_config)
}
