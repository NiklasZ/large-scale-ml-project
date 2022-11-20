from typing import Tuple, Any
from torchvision.datasets import MNIST
from torchvision.transforms import ToTensor


# Contains all datasets to train on

class IndexedMNIST(MNIST):
    def __getitem__(self, index: int) -> Tuple[Any, Any, int]:
        img, target = super().__getitem__(index)
        return img, target, index


MNIST_config = ["mnist", {'train': True, 'download': True, 'transform': ToTensor()}]

# First dataset is for training, the second for evaluation, third the constructor args for both datasets
DATASETS = {
    MNIST.__name__: (IndexedMNIST, MNIST, MNIST_config)
}
