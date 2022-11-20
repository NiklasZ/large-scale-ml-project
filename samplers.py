from typing import Sized, Iterator, TypeVar
import torch
from torch.utils.data import RandomSampler, Sampler

# These are where we define sampler classes that perform data selection. If you modify an existing sampler you should
# re-run experiments that used the sampler.

T_co = TypeVar('T_co', covariant=True)


# Base abstract class
class UpdatableSampler(Sampler):
    def __iter__(self) -> Iterator[T_co]:
        raise NotImplementedError

    def update_scores(self, losses: torch.Tensor, indices: torch.Tensor):
        pass


# Baseline random sampler
class RandomSamplerBase(RandomSampler, UpdatableSampler):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)


# Based off of torch's WeightedRandomSampler
class LastLossWeightedSampler(UpdatableSampler):
    def __init__(self, data_source: Sized,
                 num_samples: int,
                 device: torch.device,
                 replacement: bool = True) -> None:
        if not isinstance(num_samples, int) or isinstance(num_samples, bool) or \
                num_samples <= 0:
            raise ValueError("num_samples should be a positive integer "
                             "value, but got num_samples={}".format(num_samples))
        if not isinstance(replacement, bool):
            raise ValueError("replacement should be a boolean value, but got "
                             "replacement={}".format(replacement))
        self.scores = torch.zeros(len(data_source), dtype=torch.float32, device=device)
        self.num_samples = num_samples
        self.replacement = replacement
        super().__init__(data_source)

    def __iter__(self) -> Iterator[int]:
        # Creates a softmax distribution
        rand_tensor = torch.multinomial(self.scores, self.num_samples, self.replacement)
        yield from iter(rand_tensor.tolist())

    def __len__(self) -> int:
        return self.num_samples

    def update_scores(self, losses: torch.Tensor, indices: torch.Tensor):
        self.scores[indices] = losses


SAMPLERS = {
    # Basic sampler. Works by shuffling the dataset and then yielding the elements in order.
    RandomSamplerBase.__name__: RandomSamplerBase,
    LastLossWeightedSampler.__name__: LastLossWeightedSampler
}
