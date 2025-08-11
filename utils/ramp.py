from torch.optim.lr_scheduler import LRScheduler

def polynomialRamp(current, rampup_length, power):
    
    return 1 - (1 - current / rampup_length) ** power