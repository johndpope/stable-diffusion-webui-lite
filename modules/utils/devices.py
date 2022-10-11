import contextlib

import torch
import torch.backends


cpu = torch.device("cpu")

def torch_backends_info():
    print(torch.backends.cuda)
    print(torch.backends.cudnn)
    print(torch.backends.mkl)
    print(torch.backends.mkldnn)
    print(torch.backends.mps)
    print(torch.backends.openmp)
    print(torch.backends.quantized)


def get_optimal_device():
    if torch.cuda.is_available():
        torch.backends.cudnn.enabled = True
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.allow_tf32 = True
        torch.backends.cuda.matmul.allow_tf32 = True

        return torch.device("cuda")

    # for Mac: has_mps is only available in nightly pytorch (for now), `getattr` for compatibility
    if hasattr(torch, 'has_mps'):
        return torch.device("mps")

    return cpu


torch_backends_info()
device = get_optimal_device()
device_gfpgan     = device
device_bsrgan     = device
device_esrgan     = device
device_scunet     = device
device_codeformer = device
dtype = torch.float16


def torch_gc():
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()


def randn(seed, shape):
    # Pytorch currently doesn't handle setting randomness correctly when the metal backend is used.
    if device.type == 'mps':
        generator = torch.Generator(device=cpu)
        generator.manual_seed(seed)
        noise = torch.randn(shape, generator=generator, device=cpu).to(devices)
        return noise

    torch.manual_seed(seed)
    return torch.randn(shape, device=device)


def randn_without_seed(shape):
    # Pytorch currently doesn't handle setting randomness correctly when the metal backend is used.
    if device.type == 'mps':
        generator = torch.Generator(device=cpu)
        noise = torch.randn(shape, generator=generator, device=cpu).to(devices)
        return noise

    return torch.randn(shape, device=device)


def autocast():
    from modules import shared

    if dtype == torch.float32 or shared.cmd_opts.precision == "full":
        return contextlib.nullcontext()

    return torch.autocast("cuda")

