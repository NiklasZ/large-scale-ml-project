from typing import Tuple, Any
from torchvision.datasets import MNIST, CIFAR10
from torchvision.transforms import ToTensor, transforms


# Contains all datasets to train on

class IndexedMNIST(MNIST):
    def __getitem__(self, index: int) -> Tuple[Any, Any, int]:
        img, target = super().__getitem__(index)
        return img, target, index


class IndexedCIFAR10(CIFAR10):
    def __getitem__(self, index: int) -> Tuple[Any, Any, int]:
        img, target = super().__getitem__(index)
        return img, target, index


transform_train = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

transform_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

MNIST_config = ['data/mnist', {'train': True, 'download': True, 'transform': ToTensor()}]
CIFAR10_train_config = ['data/cifar10', {'train': True, 'download': True, 'transform': ToTensor()}]
CIFAR10_test_config = ['data/cifar10', {'train': False, 'download': True, 'transform': ToTensor()}]
augmented_CIFAR10_train_config = ['data/cifar10', {'train': True, 'download': True, 'transform': transform_train}]
augmented_CIFAR10_test_config = ['data/cifar10', {'train': False, 'download': True, 'transform': transform_test}]

# First dataset is for training, the second for evaluation, third the constructor args for both datasets
DATASETS = {
    MNIST.__name__: (IndexedMNIST, MNIST, MNIST_config),
    CIFAR10.__name__: (IndexedCIFAR10, CIFAR10, CIFAR10_train_config, CIFAR10_test_config),
    f'Augmented{CIFAR10.__name__}': (IndexedCIFAR10, CIFAR10, augmented_CIFAR10_train_config, augmented_CIFAR10_test_config)
}
