from torch.optim import SGD
from torch.optim.lr_scheduler import StepLR, LinearLR, MultiStepLR, CosineAnnealingLR

OPTIMISERS = {
    SGD.__name__: SGD
}
SCHEDULERS = {
    StepLR.__name__: StepLR,  # "{'step_size':5, 'gamma':0.2}"
    LinearLR.__name__: LinearLR,
    MultiStepLR.__name__: MultiStepLR,  # milestones=[15, 30], gamma=0.1
    CosineAnnealingLR.__name__: CosineAnnealingLR  # "{'T_max':200}"
}
