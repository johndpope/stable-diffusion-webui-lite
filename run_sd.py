#!/usr/bin/env python3
# Author: Armit
# Create Time: 2022/10/20 

# the minimal script to run sd model

import os
import sys

import torch
from omegaconf import OmegaConf

from modules.paths import MODEL_PATH, REPO_PATHS
#from modules.diffuser.sd_hijack import model_hijack

sys.path.append(REPO_PATHS['stable-diffusion'])
from ldm.util import instantiate_from_config
sys.path.append(REPO_PATHS['k-diffusion'])
from k_diffusion import sampling as S


device = 'cuda'

model_name = 'stable-diffusion'
model_path = os.path.abspath(os.path.join(MODEL_PATH, model_name))

ckpt_fp = os.path.join(model_path, 'model.ckpt')
#config_fp = os.path.join(model_path, 'config.yaml')
config_fp = os.path.join(REPO_PATHS['stable-diffusion'], 'configs/stable-diffusion/v1-inference.yaml')


def load_model():
    print(f"Loading config from: {config_fp}")
    sd_config = OmegaConf.load(config_fp)
    sd_model = instantiate_from_config(sd_config.model)

    print(f"Loading weights from {ckpt_fp}")
    pl_sd = torch.load(ckpt_fp, map_location="cpu")
    if "global_step" in pl_sd: print(f"Global Step: {pl_sd['global_step']}")
    sd = pl_sd["state_dict"] if "state_dict" in pl_sd else pl_sd 

    sd_model.load_state_dict(sd, strict=False)
    sd_model.half()

    sd_model.to(device)
    #model_hijack.hijack(sd_model)
    sd_model.eval()

    print("Model weights loaded.")
    return sd_model


model = load_model()
breakpoint()
