
from contextlib import nullcontext

import torch
import torch.backends

from modules import cmd_opts

cpu = torch.device('cpu')
cuda = torch.device('cuda') if torch.cuda.is_available() else cpu


def get_optimal_device():
    if torch.cuda.is_available():
        torch.backends.cudnn.enabled = True
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.allow_tf32 = True
        torch.backends.cuda.matmul.allow_tf32 = True

        return torch.device('cuda')

    return cpu


device = get_optimal_device()


def torch_gc():
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()


dtype = torch.float32 if cmd_opts.no_half else torch.float16


def autocast():
    if dtype == torch.float32 or cmd_opts.precision == 'full':
        return nullcontext()

    return torch.autocast('cuda')
