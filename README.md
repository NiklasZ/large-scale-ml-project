# Importance Sampling based on Loss
This repository contains a university project where we implement a data selection method known as "Importance sampling based on loss".
Effectively, we reweight the data sampling distribution for training batches on gradient descent
in favour of datapoints that yield the highest training loss. We experimented on the CIFAR10 dataset and found that the
while the convergence speed improves slightly, it does not enhance test accuracy and ultimately there are more valuable
methods to obtaining better convergence and accuracy. For contrast, we also show the effect data augmentation has on the
accuracy (see below plot and [report](docs/report.pdf) for more details).

![test_data_accuracy](docs/test_data_accuracy.png)

## Installation
To run the code, you will need to install all packages in the `requirements.txt`, which can be done by:

```bash
 pip install -r requirements.txt
```

Note that `torch==1.12.1+cu113` version inside assumes CUDA 11.3 is installed. If using a different CUDA version, then replace
these dependencies with the corresponding ones.
```
torch==1.12.1+cu113
torchaudio==0.12.1+cu113
torchvision==0.13.1+cu113
```

## Project Structure
The main file that trains models is `train.py`. The additional files have the following purposes:
- `arg_parser.py` - parses commandline arguments (see next section)
- `datasets.py` - contains datasets to train on
- `models.py` - contains model architectures to train with
- `samplers.py` - contains sampling methods to experiment with
- `tutorials/*` - a dumping folder for tutorial code from other sources


## Running code
The code is designed to be run a commandline where the full extent of commands can be seen from `python train.py new -h`.
Below are two example commands:

This runs the baseline standard mini-batch sampling
```bash
python train.py new --model-name ResNetMNIST --sampler-name RandomSamplerBase --training-runs 3 --wandb
```

This runs a sampling using the most recent loss
```bash
python train.py new --model-name ResNetMNIST --sampler-name LastLossWeightedSampler --sampler-config "{'num_samples':10000, 'replacement':true}" --training-runs 3 --wandb
```

## Experiment Tracking and Analysis
The project uses `wandb` to centrally track training metrics with results visible [here](https://wandb.ai/ece-239-as/cs-260D).
Using it we can aggregate results and output them in whatever format we like.