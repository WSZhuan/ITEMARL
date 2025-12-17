# utils/scheduler.py
from torch.optim.lr_scheduler import LambdaLR

def get_linear_scheduler(optimizer, start_factor, end_factor, total_iters):
    def lr_lambda(step):
        if step>=total_iters: return end_factor
        return start_factor + (end_factor-start_factor)*(step/total_iters)
    return LambdaLR(optimizer, lr_lambda)