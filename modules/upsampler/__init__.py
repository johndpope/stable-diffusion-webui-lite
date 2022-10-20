from upscaler import UpscalerLanczos

from bsrgan_model import UpscalerBSRGAN
from esrgan_model import UpscalerESRGAN
from realesrgan_model import UpscalerRealESRGAN
from ldsr_model import UpscalerLDSR
from scunet_model import UpscalerScuNET
from swinir_model import UpscalerSwinIR


import os
from urllib.parse import urlparse


def friendly_name(file: str):
    if "http" in file:
        file = urlparse(file).path

    file = os.path.basename(file)
    name, ext = os.path.splitext(file)
    return name
