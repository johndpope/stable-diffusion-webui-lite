# command line opts to start 'webui.py'

import os
import sys
import json
from argparse import ArgumentParser
from typing import Dict, Tuple

import gradio as gr

from modules import runtime
from modules.runtime import change_current_model
from modules.diffuser import sd_model, sd_samplers
from modules.diffuser.hypernetwork import list_hypernetworks
from modules.upsampler.realesrgan_model import get_realesrgan_models
from modules.face_restorer.codeformer_model import setup_model

def list_sd_ckpts(): return sd_model.list_models()
def list_samplers(): return [x.name for x in sd_samplers.samplers]
def list_upsamplers(): return [x.name for x in get_realesrgan_models(None)]
def list_face_restorers(): return [x.name for x in runtime.face_restorers]
def list_interrogaters(): return ['CLIP', 'danbooru']


parser = ArgumentParser()

# gradio server
parser.add_argument("--port", type=int, help="launch gradio with given server port, you need root/admin rights for ports < 1024, defaults to 7860 if available", default=7860)
parser.add_argument("--share", action='store_true', help="use share=True for gradio and make the UI accessible through their site (doesn't work for me but you might have better luck)")
parser.add_argument("--debug",  action='store_true', help="launch gradio in debug mode")

# general model runtime detailed setting
parser.add_argument("--precision", type=str, help="evaluate at this precision", choices=["full", "autocast"], default="autocast")
parser.add_argument("--disable-safe-unpickle", action='store_true', help="disable checking pytorch models for malicious code", default=False)
parser.add_argument("--xformers", action='store_true', help="enable xformers for cross attention layers")
parser.add_argument("--force-enable-xformers", action='store_true', help="enable xformers for cross attention layers regardless of whether the checking code thinks you can run it; do not make bug reports if this fails to work")
parser.add_argument("--opt-split-attention", action='store_true', help="force-enables cross-attention layer optimization. By default, it's on for torch.cuda and off for other torch devices.")
parser.add_argument("--disable-opt-split-attention", action='store_true', help="force-disables cross-attention layer optimization")
parser.add_argument("--opt-split-attention-v1", action='store_true', help="enable older version of split attention optimization that does not consume all the VRAM it can find")

cmd_opts = parser.parse_args()


class OptionInfo:

    def __init__(self, default=None, label="", component=None, component_args=None, onchange=None):
        self.section = None
        self.default = default
        self.label = label
        self.component = component
        self.component_args = component_args
        self.onchange = onchange


def make_section(name: Tuple[str, str], options: Dict[str, OptionInfo]):
    for v in options.values():
        v.section = name            # ['inner name', 'display name']
    return options


options_templates = { }

options_templates.update(make_section(('save', "Saving images/grids"), {
    "samples_save": OptionInfo(True, "Always save all generated images"),
    "samples_format": OptionInfo('jpg', 'File format for images to save'),
    "samples_filename_pattern": OptionInfo("", "Images filename pattern"),

    "grid_save": OptionInfo(True, "Always save all generated image grids"),
    "grid_format": OptionInfo('jpg', 'File format for grids to save'),
    "grid_extended_filename": OptionInfo(False, "Add extended info (seed, prompt) to filename when saving grid"),

    "show_format": OptionInfo('png', 'File format for images to show in browser'),
    "download_format": OptionInfo('png', 'File format for images to download'),
    "enable_pnginfo": OptionInfo(True, "Save text information about generation parameters as chunks to png files"),
    "jpeg_quality": OptionInfo(80, "Quality for saved jpeg images", gr.Slider, {"minimum": 1, "maximum": 100, "step": 1}),
}))

options_templates.update(make_section(('diffuser', "Stable Diffusion"), {
    "sd_checkpoints": OptionInfo(None, "Stable Diffusion checkpoint", gr.Dropdown, lambda: {"choices": list_sd_ckpts()}),
    "sd_hypernetwork": OptionInfo('None', "Stable Diffusion finetune hypernetwork", gr.Dropdown, lambda: {"choices": ['None'] + list_hypernetworks()}),
    "img2img_color_correction": OptionInfo(False, "Apply color correction to img2img results to match original colors."),
    "img2img_fix_steps": OptionInfo(False, "With img2img, do exactly the amount of steps the slider specifies (normally you'd do less with less denoising)."),
    "enable_emphasis": OptionInfo(True, "Emphasis: use (text) to make model pay more attention to text and [text] to make it pay less attention"),
    "enable_quantization": OptionInfo(False, "Enable quantization in K samplers for sharper and cleaner results. This may change existing seeds. Requires restart to apply."),
    "enable_batch_seeds": OptionInfo(True, "Make K-diffusion samplers produce same images in a batch as when making a single image"),
    "filter_nsfw": OptionInfo(False, "Filter NSFW content"),
}))

options_templates.update(make_section(('sampler', "Sampler Parameters"), {
    "hide_samplers": OptionInfo([], "Hide samplers in user interface (requires restart)", gr.CheckboxGroup, lambda: {"choices": list_samplers()}),
    "eta_ancestral": OptionInfo(1.0, "eta (noise multiplier) for ancestral samplers", gr.Slider, {"minimum": 0.0, "maximum": 1.0, "step": 0.01}),
    "eta_ddim": OptionInfo(0.0, "eta (noise multiplier) for DDIM", gr.Slider, {"minimum": 0.0, "maximum": 1.0, "step": 0.01}),
    "ddim_discretize": OptionInfo('uniform', "img2img DDIM discretize", gr.Radio, {"choices": ['uniform', 'quad']}),
    's_churn': OptionInfo(0.0, "sigma churn", gr.Slider, {"minimum": 0.0, "maximum": 1.0, "step": 0.01}),
    's_tmin':  OptionInfo(0.0, "sigma tmin",  gr.Slider, {"minimum": 0.0, "maximum": 1.0, "step": 0.01}),
    's_noise': OptionInfo(1.0, "sigma noise", gr.Slider, {"minimum": 0.0, "maximum": 1.0, "step": 0.01}),
}))

options_templates.update(make_section(('face-restorer', "Face Restoration"), {
    "face_restorer": OptionInfo(None, "Face restoration model", gr.Radio, lambda: {"choices": list_face_restorers()}, onchange=lambda: change_current_model('face_restorer')),
    "code_former_weight": OptionInfo(0.5, "CodeFormer weight parameter; 0 = maximum effect; 1 = minimum effect", gr.Slider, {"minimum": 0, "maximum": 1, "step": 0.01}),
    "face_restoration_unload": OptionInfo(False, "Move face restoration model from VRAM into RAM after processing"),
}))

options_templates.update(make_section(('upsampler', "Super Resolution"), {
    "upsampler": OptionInfo(None, "Upsampler", gr.Radio, lambda: {"choices": list_upsamplers()}, onchange=lambda: change_current_model('upsampler')),
    "ESRGAN_tile": OptionInfo(192, "Tile size for ESRGAN. 0 = no tiling.", gr.Slider, {"minimum": 0, "maximum": 512, "step": 16}),
    "ESRGAN_tile_overlap": OptionInfo(8, "Tile overlap, in pixels for ESRGAN. Low values = visible seam.", gr.Slider, {"minimum": 0, "maximum": 48, "step": 1}),
    "realesrgan_enabled_models": OptionInfo(["R-ESRGAN x4+", "R-ESRGAN x4+ Anime6B"], "Select which Real-ESRGAN models to show in the web UI. (Requires restart)", gr.CheckboxGroup, lambda: {"choices": [x.name for x in get_realesrgan_models(None)]}),
    "SWIN_tile": OptionInfo(192, "Tile size for all SwinIR.", gr.Slider, {"minimum": 16, "maximum": 512, "step": 16}),
    "SWIN_tile_overlap": OptionInfo(8, "Tile overlap, in pixels for SwinIR. Low values = visible seam.", gr.Slider, {"minimum": 0, "maximum": 48, "step": 1}),
    "ldsr_steps": OptionInfo(100, "LDSR processing steps. Lower = faster", gr.Slider, {"minimum": 1, "maximum": 200, "step": 1}),
}))

options_templates.update(make_section(('interrogater', "Interrogate"), {
    "interrogater": OptionInfo(None, "Interrogater", gr.Radio, lambda: {"choices": list_interrogaters()}, onchange=lambda: change_current_model('interrogater')),
    "interrogate_use_builtin_artists": OptionInfo(True, "Interrogate: use artists from artists.csv"),
    "interrogate_clip_num_beams": OptionInfo(4, "Interrogate: num_beams for BLIP", gr.Slider, {"minimum": 1, "maximum": 16, "step": 1}),
    "interrogate_clip_min_length": OptionInfo(24, "Interrogate: minimum description length (excluding artists, etc..)", gr.Slider, {"minimum": 1, "maximum": 128, "step": 1}),
    "interrogate_clip_max_length": OptionInfo(48, "Interrogate: maximum description length", gr.Slider, {"minimum": 1, "maximum": 256, "step": 1}),
    "interrogate_clip_dict_limit": OptionInfo(1500, "Interrogate: maximum number of lines in text file (0 = No limit)"),
    'CLIP_stop_at_last_layers': OptionInfo(2, "Stop At last layers of CLIP model", gr.Slider, {"minimum": 1, "maximum": 12, "step": 1}),
}))

options_templates.update(make_section(('ui', "User interface"), {
    "show_progress_every_n_steps": OptionInfo(0, "Show image creation progress every N sampling steps. Set 0 to disable.", gr.Slider, {"minimum": 0, "maximum": 10, "step": 1}),
    "return_grid": OptionInfo(True, "Show grid in results for web"),
    "add_model_hash_to_info": OptionInfo(True, "Add model hash to generation information"),
    "add_model_name_to_info": OptionInfo(False, "Add model name to generation information"),
    "js_modal_lightbox": OptionInfo(True, "Enable full page image viewer"),
}))

options_templates.update(make_section(('system', "System"), {
    "models_loaded": OptionInfo([], "track all loaded models in runtine"),
    "memmon_poll_rate": OptionInfo(8, "VRAM usage polls per second during generation. Set to 0 to disable.", gr.Slider, {"minimum": 0, "maximum": 40, "step": 1}),
}))


class Options:

    def __init__(self, options):
        self.options = options
        self.data = {k: v.default for k, v in self.options.items()}

    def __setattr__(self, key, value):
        if self.data is not None:
            if key in self.data:
                self.data[key] = value

        return super(Options, self).__setattr__(key, value)

    def __getattr__(self, item):
        if self.data is not None:
            if item in self.data:
                return self.data[item]

        if item in self.options:
            return self.options[item].default

        return super(Options, self).__getattribute__(item)

    def save(self, fn):
        with open(fn, 'w') as file:
            json.dump(self.data, file)

    def load(self, fn):
        def is_same_type(x, y):
            if None in [x, y]: return True

            # numerical type are the same
            typemap = { int: float }
            type_x = typemap.get(type(x), type(x))
            type_y = typemap.get(type(y), type(y))

            return type_x == type_y

        with open(fn) as file:
            self.data = json.load(file)

        bad_settings = 0
        for k, v in self.data.items():
            info = self.options.get(k)
            if info is not None and not is_same_type(info.default, v):
                print(f"Warning: bad setting value: {k}: {v} typed ({type(v).__name__}; expected {type(info.default).__name__})", file=sys.stderr)
                bad_settings += 1

        if bad_settings > 0:
            print(f'The program is likely to not work with bad settings.\n' + 
                  f'Settings file: {fn}\n' + 
                  f'Either fix the file, or delete it and restart.', file=sys.stderr)

    def onchange(self, key, func):
        item = self.options.get(key)
        item.onchange = func

    def dumpjson(self):
        d = {k: self.data.get(k, self.options[k].default) for k in self.options.keys()}
        return json.dumps(d)


opts = Options(options_templates)
opts_fp = cmd_opts.ui_settings_file
if os.path.exists(opts_fp):
    opts.load(opts_fp)
