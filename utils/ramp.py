from torch.optim.lr_scheduler import LRScheduler
import numpy as np

def polynomialRamp(current, rampup_length, power):
    
    return 1 - (1 - current / rampup_length) ** power

def sigmoid_ramp_up(current, rampup_length):
    """Exponential rampup from https://arxiv.org/abs/1610.02242"""
    if rampup_length == 0:
        return 1.0
    else:
        current = np.clip(current, 0.0, rampup_length)
        phase = 1.0 - current / rampup_length
        return float(np.exp(-5.0 * phase * phase))
    
def cosine_ramp_down(current, ramp_down_length, out_multiplier = 1.0, in_multiplier = 1.0):
    # assert 0 <= current <= rampdown_length
    return out_multiplier * (np.cos(np.pi * current / ramp_down_length + in_multiplier) + 1)