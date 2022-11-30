from typing import Optional
import uuid
import torch
import tqdm
from sklearn.metrics import accuracy_score
from torch.optim import Optimizer
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from arg_parser import get_train_args
from datasets import DATASETS
from models import MODELS
from samplers import SAMPLERS, UpdatableSampler, RandomSamplerBase
from time import time


def train(args_dict: dict, patch_tb=True):
    device = torch.device("cuda:0" if torch.cuda.is_available() and not args_dict['cpu_only'] else "cpu")
    TrainingDataset, EvauationDataset, ctor_args = DATASETS[args_dict['dataset']]
    train_ds = TrainingDataset(*ctor_args[:-1], **ctor_args[-1])
    test_ds = EvauationDataset(*ctor_args[:-1], **ctor_args[-1])
    use_wandb = args_dict['wandb']
    scale_gradients = args_dict['scale_grad']
    run_group_id = uuid.uuid1()
    args_dict['run_group_id'] = run_group_id

    if use_wandb:
        import wandb
        base_run_id = wandb.util.generate_id()
        # Should only run once
        if patch_tb:
            wandb.tensorboard.patch(save=False)

    results = []
    for r in range(args_dict['training_runs']):

        writer = None
        # If using wandb, set up the run so it is logged later.
        if use_wandb:
            run_id = f'{base_run_id}_{r}'
            experiment_name = f"{args_dict['run_name']}_{args_dict['model_name']}_{args_dict['sampler_name']}_{run_id}"
            run = wandb.init(id=run_id,
                             project='cs-260D', entity='ece-239-as',
                             config=args_dict, name=experiment_name)
            writer = SummaryWriter(f"/tmp/{experiment_name}")
            print(f'\nTraining run {r}, wandb: {run_id}')
        else:
            print(f'\nTraining run {r}')

        # Start timer
        start = time()
        model = MODELS[args_dict['model_name']]().to(device)
        if args_dict['sampler_config']:
            sampler = SAMPLERS[args_dict['sampler_name']](train_ds, **args_dict['sampler_config'], device=device)
        else:
            sampler = SAMPLERS[args_dict['sampler_name']](train_ds)

        train_dl = DataLoader(dataset=train_ds, batch_size=args_dict['sgd_minibatch_size'], sampler=sampler)
        warmup_dl = DataLoader(dataset=train_ds, batch_size=args_dict['sgd_minibatch_size'],
                               sampler=RandomSamplerBase(train_ds))
        test_dl = DataLoader(test_ds, batch_size=args_dict['sgd_minibatch_size'])
        optimizer = torch.optim.SGD(model.parameters(),
                                    lr=args_dict['optimiser_config']['lr'],
                                    momentum=args_dict['optimiser_config']['momentum'])
        final_accuracy = train_once(model=model,
                                    device=device,
                                    train_dl=train_dl,
                                    test_dl=test_dl,
                                    warmup_dl=warmup_dl,
                                    warm_up_epochs=args_dict['warm_up_epochs'],
                                    optimizer=optimizer,
                                    stop_after=args_dict['stop_after'],
                                    sampler=sampler, use_wandb=use_wandb,
                                    writer=writer,
                                    log_minibatch_every=args_dict['log_minibatch_every'],
                                    scale_gradients=scale_gradients)

        results.append({'time': time() - start,
                        'test_accuracy': final_accuracy})

        if use_wandb:
            run.finish()
            writer.close()
    print(results)


def train_once(model: torch.nn.Module,
               device: torch.device,
               train_dl: DataLoader,
               warmup_dl: DataLoader,
               test_dl: DataLoader,
               optimizer: Optimizer,
               stop_after: int,
               warm_up_epochs: int,
               sampler: UpdatableSampler,
               use_wandb: bool,
               writer: Optional[SummaryWriter],
               log_minibatch_every: int,
               scale_gradients: bool) -> float:
    criterion = torch.nn.CrossEntropyLoss(reduction='none')  # return loss per datapoint rather than the mean.

    epoch = 0
    data_point_counter = 0
    batch_counter = 0
    last_evaluation = 0
    last_test_accuracy = 0
    no_eval_improvements = 0
    avg_loss = 0
    training_time = 0
    evaluation_time = 0

    # Get accuracy once before training
    model.eval()  # set model to eval mode
    test_accuracy = evaluate(test_dl, device, model)
    print(f'Initial test accuracy: {test_accuracy:.3f}')
    model.train()  # set back to train mode
    if use_wandb:
        writer.add_scalar(f"test_accuracy", test_accuracy, data_point_counter)

    while True:  # loop over the dataset until stopping condition is met
        running_loss = 0.0
        sampling_dl = warmup_dl if epoch < warm_up_epochs else train_dl

        # Iterate through batches of an epoch.
        print(f'==Epoch {epoch}==')
        for i, data in enumerate(sampling_dl, 0):

            train_start = time()

            # get the inputs; data is a list of [inputs, labels]
            inputs, labels, indices = data

            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            outputs = model(inputs.to(device))
            data_point_loss = criterion(outputs, labels.to(device))

            # Scale gradients to make the estimator unbiased.
            if scale_gradients and sampling_dl == train_dl:
                loss = ((data_point_loss * 1 / sampler.probabilities[indices]) / len(sampler.probabilities)).mean()
            else:
                loss = data_point_loss.mean()
            # Calculate gradients
            loss.backward()
            # Update weights
            optimizer.step()

            batch_counter += 1

            # track statistics
            running_loss += loss.item()
            avg_loss += 1 / batch_counter * (loss.item() - avg_loss)  # update average as we go
            data_point_counter += len(labels)

            # Assume training ends here
            training_time += time() - train_start

            with torch.no_grad():
                sampler.update_scores(data_point_loss, indices)

            if use_wandb:
                writer.add_scalar(f"training_loss", avg_loss, data_point_counter)

            # Print loss
            if batch_counter > 0 and batch_counter % log_minibatch_every == 0:
                print(f'[{epoch}, {i:5d}] loss: {running_loss / 100:.3f}')
                running_loss = 0.0

            # Do evaluation run roughly once per epoch's data.
            if data_point_counter - last_evaluation >= len(sampling_dl.dataset):
                evaluation_start = time()
                last_evaluation = data_point_counter
                model.eval()  # set model to eval mode
                test_accuracy = evaluate(test_dl, device, model)
                print(f'[{epoch}, {i:5d}] test accuracy: {test_accuracy:.3f}')
                model.train()  # set back to train mode
                evaluation_time += time() - evaluation_start

                if use_wandb:
                    writer.add_scalar(f"test_accuracy", test_accuracy, data_point_counter)

                # We round as more minor improvements are not worth tracking.
                if round(test_accuracy, 3) <= round(last_test_accuracy, 3):
                    no_eval_improvements += 1
                    print(f'Accuracy no better than previous of {last_test_accuracy:.3f}. '
                          f'Will stop after {stop_after - no_eval_improvements} more occurrence(s)')

                    # If evaluation does not improve
                    if no_eval_improvements >= stop_after:
                        print('Finished training run')
                        writer.add_scalar(f"training_time", training_time)
                        writer.add_scalar(f"sampling_time", sampler.sampling_time)
                        writer.add_scalar(f"evaluation_time", evaluation_time)
                        return test_accuracy

                last_test_accuracy = test_accuracy

        epoch += 1


def get_prediction(x, model: torch.nn.Module):
    model.eval()  # prepares model for predicting
    probabilities = torch.softmax(model(x), dim=1)
    predicted_class = torch.argmax(probabilities, dim=1)
    return predicted_class, probabilities


def evaluate(test_dl: DataLoader, device: torch.device, model: torch.nn.Module) -> float:
    true_y, pred_y = [], []
    for batch in tqdm.tqdm(iter(test_dl), total=len(test_dl)):
        x, y = batch
        true_y.extend(y)
        preds, probs = get_prediction(x.to(device), model)
        pred_y.extend(preds.cpu())
    return accuracy_score(true_y, pred_y)


if __name__ == "__main__":
    args_dict = get_train_args()
    train(args_dict)
