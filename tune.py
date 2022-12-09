from typing import Union, List, OrderedDict, Set

import numpy as np

from train import train


def pick_randomly(candidates: Union[float, int, List]):
    if isinstance(candidates, list):
        idx = np.random.choice(len(candidates))
        return candidates[idx]
    else:
        return candidates


def pick_random_hyper_parameters(hyper_parameters: dict, already_chosen_hypers: Set[str]) -> dict:
    chosen_parameters = {}
    for key, value in hyper_parameters.items():
        if key == 'sampler_name':
            sampler = pick_randomly(value)
            chosen_parameters['sampler_name'] = sampler
            chosen_parameters['sampler_config'] = {k: pick_randomly(v) for k, v in
                                                   hyper_parameters['sampler_config'][sampler].items()}
        elif key == 'sampler_config':
            continue
        elif key == 'optimiser_config':
            chosen_parameters['optimiser_config'] = {k: pick_randomly(v) for k, v in
                                                     hyper_parameters['optimiser_config'].items()}
        elif key == 'lr_scheduler':
            scheduler = pick_randomly(value)
            chosen_parameters['lr_scheduler'] = scheduler
            chosen_parameters['lr_scheduler_config'] = {k: pick_randomly(v) for k, v in
                                                        hyper_parameters['lr_scheduler_config'][scheduler].items()}
        elif key == 'lr_scheduler_config':
            continue
        else:
            chosen_parameters[key] = pick_randomly(value)

    combined = str(chosen_parameters)

    if combined in already_chosen_hypers:
        return pick_random_hyper_parameters(hyper_parameters, already_chosen_hypers)

    already_chosen_hypers.add(combined)
    return chosen_parameters


if __name__ == "__main__":
    search_args = {
        # Properties that usually stay the same
        'command': 'new',
        'model_name': 'CustomResNet18CIFAR10',
        'dataset': 'AugmentedCIFAR10',
        'training_runs': 2,
        'stop_after': 20,
        'wandb': True,
        'cpu_only': False,
        'run_name': 'tune',
        'log_minibatch_every': 100,
        # Properties to search
        'sampler_name': ['RunningLossWeightedSampler', 'LastLossWeightedSampler'],
        'sampler_config': {
            'LastLossWeightedSampler': {'num_samples': [256, 1024, 2048, 4096, 8192, 16384, 30000],
                                        'replacement': [False, True],
                                        'additive_smoothing': [0, 1, 2, 4]},
            'RunningLossWeightedSampler': {'num_samples': [256, 1024, 2048, 4096, 8192, 16384, 30000],
                                           'replacement': [False, True],
                                           'additive_smoothing': [0, 1, 2, 4],
                                           'running_loss_pct': [0.2, 0.4, 0.6, 0.8]}
        },
        'warm_up_epochs': [0, 1, 2, 3, 4, 5],
        # 'optimiser_config': {'lr': [0.01, 0.001, 0.0001, 0.00001], 'momentum': [0.5, 0.7, 0.9, 0.95]},
        'optimiser': 'SGD',
        'optimiser_config': {'lr': 0.01, 'momentum': 0.9},
        # 'scale_grad': [False, True],
        'scale_grad': False,
        'lr_scheduler': ['CosineAnnealingLR', 'MultiStepLR'],
        'lr_scheduler_config': {'CosineAnnealingLR': {'T_max': 200},
                                'MultiStepLR': {'milestones': [[15, 30]], 'gamma': 0.1}},
        'sgd_minibatch_size': [64, 128, 256, 512]
    }

    already_chosen = set()
    runs = 0
    while True:
        out = pick_random_hyper_parameters(search_args, already_chosen)
        print(out)
        try:
            if runs == 0:
                train(out)
            else:
                train(out, patch_tb=False)
            runs += 1

        except Exception as e:
            print(vars(e))
            # Sometimes a particular configuration will yield to a run with exploding scores, gradients or activations
            # which is often indicated by NaN, 0 or inf errors. While not desirable, this should not stop future
            # tuning attempts on other configurations.
            if hasattr(e, 'args') and any([x in e.args[0] for x in ['nan', 'NaN', '0', 'inf']]):
                print('Exception occurred:')
                print(e)
            # Other cases should still be thrown though
            else:
                raise e
