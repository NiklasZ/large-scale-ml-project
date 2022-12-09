import argparse
from datasets import DATASETS
from models import MODELS
from optimisers import OPTIMISERS, SCHEDULERS
from samplers import SAMPLERS
import json


def get_train_args():
    parser = argparse.ArgumentParser(description='Data Selection Args')
    subparsers = parser.add_subparsers(help='sub-command help', dest='command')
    subparsers.required = True

    # Arguments to resume a run
    # TODO: support resuming runs (if runs take a lot of time).
    # parser_resume = subparsers.add_parser('resume', help='when you want to resume training an existing agent.')
    # parser_resume.add_argument('--run-id', type=str, required=False, default=None,
    #                            help='supply the wandb ID of the run to resume.')
    # parser_resume.add_argument('--entity-name', type=str, required=False, default='ece-239-as',
    #                            help="name of the entity whose run should be resumed")

    # Arguments for a new run
    parser_new = subparsers.add_parser('new', help='when you want to train a new agent.')
    parser_new.add_argument('--run-name', type=str, required=False, default='run',
                            help='the name of this training run')
    # parser_new.add_argument('--group-name', type=str, required=False, default=None,
    #                         help='name of the group of runs this one belongs to')
    parser_new.add_argument('--sgd-minibatch-size', type=int, required=False, default=128,
                            help='how many sampled environment steps to use per gradient descent training batch.')
    parser_new.add_argument('--cpu-only', type=bool, action=argparse.BooleanOptionalAction, required=False,
                            default=False,
                            help='toggle if you only want to use the CPU even though a GPU is available.')
    parser_new.add_argument('--model-name', type=str, required=True,
                            help='which model to use for training. Available models are: '
                                 f"{', '.join(MODELS.keys())}")
    parser_new.add_argument('--sampler-name', type=str, required=True,
                            help='which data selection sampler to use for training. Available samplers are: '
                                 f"{', '.join(SAMPLERS.keys())}")
    parser_new.add_argument('--sampler-config', type=str, required=False,
                            help='what constructor arguments to pass to the sampler (should be a JSON). '
                                 "E.g: \"{'num_samples':10000, 'replacement':true}\"")
    parser_new.add_argument('--wandb', type=bool, action=argparse.BooleanOptionalAction, required=False, default=False,
                            help='toggles wandb for training logging and model saving')
    parser_new.add_argument('--stop-after', type=int, required=False, default=2,
                            help='after how many evaluations with no improvement to stop training.')
    parser_new.add_argument('--training-runs', type=int, required=False, default=1,
                            help='how often to run a particular training configuration')
    parser_new.add_argument('--warm-up-epochs', type=int, required=False, default=1,
                            help='how many warm-up epochs of regular mini-batch to use before custom sampling')
    parser_new.add_argument('--dataset', type=str, required=True,
                            help='which dataset to use for training Available datasets are are: '
                                 f"{', '.join(DATASETS.keys())}")
    parser_new.add_argument('--log-minibatch-every', type=int, required=False, default=100,
                            help='after how many mini-batches to log the progress of the training')
    parser_new.add_argument('--scale-grad', type=bool, action=argparse.BooleanOptionalAction, required=False,
                            default=False,
                            help='scales gradients by scalar of 1/(num samples * probability of sample). Used to account for biasing of gradient')
    parser_new.add_argument('--optimiser', type=str, required=False, default='SGD',
                            help=f"What optimiser to use. Available optimisers: {', '.join(OPTIMISERS.keys())}")
    parser_new.add_argument('--optimiser-config', type=str, required=False,
                            default="{'lr':0.1, 'momentum':0.9, 'weight_decay':0.0005}",
                            help='what arguments to pass to the optimiser sampler (should be a JSON). '
                                 "E.g: \"{'lr':0.1, 'momentum':0.9, 'weight_decay':0.0005}\"")
    parser_new.add_argument('--lr-scheduler', type=str, required=False,
                            help=f"what learning rate scheduler to use. Available schedulers: {', '.join(SCHEDULERS.keys())}")
    parser_new.add_argument('--lr-scheduler-config', type=str, required=False, default="{'step_size':10, 'gamma':0.2}",
                            help='what arguments to pass to the scheduler sampler (should be a JSON). '
                                 "E.g: \"{'step_size':10, 'gamma':0.2}\"")

    args = vars(parser.parse_args())

    if args['optimiser'] not in OPTIMISERS:
        raise Exception(f"Unknown optimiser {args['optimiser']}. Available optimisers: {', '.join(OPTIMISERS.keys())}")

    if args['lr_scheduler'] and args['lr_scheduler'] not in SCHEDULERS:
        raise Exception(
            f"Unknown scheduler {args['lr_scheduler']}. Available schedulers: {', '.join(SCHEDULERS.keys())}")

    if args['model_name'] not in MODELS:
        raise Exception(f"Unknown model {args['model_name']}. Available models: {', '.join(MODELS.keys())}")

    if args['sampler_name'] not in SAMPLERS:
        raise Exception(f"Unknown sampler {args['sampler_name']}. Available samplers: {', '.join(SAMPLERS.keys())}")

    if args['dataset'] not in DATASETS:
        raise Exception(f"Unknown dataset {args['dataset']}. Available datasets: {', '.join(DATASETS.keys())}")

    if args['sampler_config']:
        args['sampler_config'] = json.loads(args['sampler_config'].replace("\'", "\""))

    if args['optimiser_config']:
        args['optimiser_config'] = json.loads(args['optimiser_config'].replace("\'", "\""))

    if args['lr_scheduler_config']:
        args['lr_scheduler_config'] = json.loads(args['lr_scheduler_config'].replace("\'", "\""))

    return args
