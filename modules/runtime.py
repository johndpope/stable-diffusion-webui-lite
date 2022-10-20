# global objects at runtime should be all put here

import os
import sys
import glob
import importlib
from threading import Lock
from datetime import datetime

import torch
import tqdm
from basicsr.utils.download_util import load_file_from_url

from modules.upsampler.upscaler import Upscaler
from modules.paths import MODEL_PATH, MODULE_PATH, RESOURCE_PATH
from modules import devices
from modules.cmd_opts import cmd_opts, opts
from modules.diffuser import hypernetwork, sd_model
from modules.prompt_helper.artists import ArtistsDatabase
from modules.prompt_helper.styles import StyleDatabase
from modules.interrogator.interrogate import InterrogateModels
from modules.perfcount import HardwareMonitor
from modules.diffuser import sd_samplers, hypernetwork
from modules import face_restorer
from modules.scripts import ScriptRunner 


'''
These model pools work as follows:
  { 'model_name': (model_object, model_device, model_getter_func) }
when model not loaded yet or unloaded:
  { 'animefull': (None, None, lambda: get_sd_model(ckpt_fp)) }
when model loaded:
  { 'animefull': (StableDiffusion(ckpt_fp), torch.device("cuda"), lambda: get_sd_model(ckpt_fp)) }
'''
diffusers      = { }
hypernetworks  = { }
embeddings     = { }
upsamplers     = { }
face_restorers = { }
interrogators  = { }
model_pools = {
    'diffusers':      diffusers,
    'hypernetworks':  hypernetworks,
    'embeddings':     embeddings,
    'upsamplers':     upsamplers,
    'face_restorers': face_restorers,
    'interrogators':  interrogators,
}
model_defaults = {
    'diffusers':      None,     # 'model_name'
    'hypernetworks':  None,
    'embeddings':     None,
    'upsamplers':     None,
    'face_restorers': None,
    'interrogators':  None,
}

artists_db = ArtistsDatabase(os.path.join(RESOURCE_PATH, 'artists.csv'))
styles_db  = StyleDatabase  (os.path.join(RESOURCE_PATH, 'styles.csv'))

script_runner = ScriptRunner()
monitor = HardwareMonitor(opts.memmon_poll_rate)

batch_cond_uncond = cmd_opts.always_batch_cond_uncond or not (cmd_opts.lowvram or cmd_opts.medvram)
parallel_processing_allowed = not cmd_opts.lowvram and not cmd_opts.medvram


lock = Lock()


class State:

    skipped = False
    interrupted = False
    job = ""
    job_no = 0
    job_count = 0
    job_timestamp = '0'
    sampling_step = 0
    sampling_steps = 0
    current_latent = None
    current_image = None
    current_image_sampling_step = 0
    textinfo = None

    def skip(self):
        self.skipped = True

    def interrupt(self):
        self.interrupted = True

    def nextjob(self):
        self.job_no += 1
        self.sampling_step = 0
        self.current_image_sampling_step = 0
        
    def get_job_timestamp(self):
        return datetime.now().strftime("%Y%m%d%H%M%S")  # shouldn't this return job_timestamp?


state = State()


def init():
    # stable-diffusion
    sd_model.setup_model()
    diffusers['default'] = sd_model.load_model()
    hypernetwork.load_hypernetwork(opts.sd_hypernetwork)
    hypernetworks['default'] = None

    # face_restorer
    face_restorer.codeformer.setup_model(cmd_opts.codeformer_models_path)
    face_restorer.gfpgan.setup_model(cmd_opts.gfpgan_models_path)
    face_restorers.append(face_restorer.FaceRestore())

    # upsampler
    _load_upscalers()

    monitor.start()
    script_runner.load_scripts()


def reload():
    state.interrupt()

    unload_model()

    print('Reloading Custom Scripts')
    script_runner.reload_scripts()
    

def _load_upscalers():
    # We can only do this 'magic' method to dynamically load upscalers if they are referenced,
    # so we'll try to import any _model.py files before looking in __subclasses__
    for file in os.listdir(MODULE_PATH):
        if "_model.py" in file:
            model_name = file.replace("_model.py", "")
            full_model = f"modules.{model_name}_model"
            try:
                importlib.import_module(full_model)
            except:
                pass
    datas = []
    c_o = vars(shared.cmd_opts)
    for cls in Upscaler.__subclasses__():
        name = cls.__name__
        module_name = cls.__module__
        module = importlib.import_module(module_name)
        class_ = getattr(module, name)
        cmd_name = f"{name.lower().replace('upscaler', '')}_models_path"
        opt_string = None
        try:
            if cmd_name in c_o:
                opt_string = c_o[cmd_name]
        except:
            pass
        scaler = class_(opt_string)
        for child in scaler.scalers:
            datas.append(child)

    sd_upscalers = datas


def init_model_pools():
    def load_local_ckpts(model_pool, model_dp, load_func):
        ckpt_fns = [fn for fn in os.listdir(model_dp) if os.path.splitext(fn)[-1] in ['.pt', '.pth']]
        names = [os.path.splitext(os.path.basename(name))[0] for name in ckpt_fns]
        model_pool.update({
            name: (None, None, lambda: load_func(os.path.join(model_dp, ckpt_fn)))
                for name, ckpt_fn in zip(names, ckpt_fns)
        })

    load_local_ckpts(diffusers,     os.path.join(MODEL_PATH, 'stable-diffusion'), get_sd_model),
    load_local_ckpts(hypernetworks, os.path.join(MODEL_PATH, 'hypernetworks'),    get_hypernet_model),
    load_local_ckpts(embeddings,    os.path.join(MODEL_PATH, 'embeddings'),       get_embedding_model),

    face_restorers.update({
        'FPGAN':      (None, None, lambda: get_fpgan()),
        'CodeFormer': (None, None, lambda: get_codeformer()),
    })

    # real-esrgan
    upsamplers.update({

    })
    # swirl
    upsamplers.update({

    })
    # swirl
    upsamplers.update({

    })

    interrogators.update({
        'CLIP':         (None, None, lambda: InterrogateModels("interrogate")),
        'deepdanbooru': (None, None, lambda: get_deepdanbooru()),
    })
    

def load_model(kind:str, name:str, force_reload=False) -> object:
    if name is None: return

    model_kind = model_pools[kind]
    for _name, (model, _, getter) in model_kind.items():
        if _name == name:
            if model is None or force_reload:
                model = getter().to(devices.cpu)
                model_kind[name] = (model, devices.cpu, getter)
                print(f'[load_model] model {_name} loaded')
            else:
                print(f'[load_model] model {_name} already loaded')
            return model

    print(f'[load_model] model {name} not found')


def unload_model(kind:str, name:str):
    if name is None: return

    model_kind = model_pools[kind]
    for _name, (model, device, getter) in model_kind.items():
        if _name == name:
            if model is not None:
                del model_kind[name]
                model_kind[name] = (None, None, getter)
                if device == devices.cuda:
                    devices.torch_gc()
                print(f'[unload_model] model {_name} unloaded')
            else:
                print(f'[unload_model] model {_name} not loaded')
            return

    print(f'[unload_model] model {name} not found')


def send_model(kind:str, name:str, to_device:torch.device):
    if name is None: return

    model_kind = model_pools[kind]
    for _name, (model, device, getter) in model_kind.items():
        if _name == name:
            if model is not None:
                del model_kind[name]
                try: model = model.to(to_device)
                except: pass
                model_kind[name] = (model, model.device, getter)
                if device == devices.cuda and to_device == devices.cpu:
                    devices.torch_gc()
                print(f'[send_model] model {_name} from {device} to {to_device}, now at {model.device}')
            else:
                print(f'[send_model] model {_name} not loaded')
            return

    print(f'[send_model] model {name} not found')


def set_default_model():
    change_current_model('diffusers', 'animefull')
    change_current_model('hypernetworks', None)
    change_current_model('embeddings', None)
    change_current_model('upsamplers', 'real-esrgan anime')
    change_current_model('face_restorers', 'CodeConformer')
    change_current_model('interrogators', 'animefull')


def change_current_model(kind:str, name:str):
    cur_name = model_defaults[kind]

    if name is None:
        unload_model(kind, cur_name)
    else:
        send_model(kind, cur_name, devices.cpu)
        load_model(kind, name)
    
    model_defaults[kind] = name


def load_models(model_path: str, model_url: str = None, command_path: str = None, ext_filter=None, download_name=None) -> list:
    """
    A one-and done loader to try finding the desired models in specified directories.

    @param download_name: Specify to download from model_url immediately.
    @param model_url: If no other models are found, this will be downloaded on upscale.
    @param model_path: The location to store/find models in.
    @param command_path: A command-line argument to search for models in first.
    @param ext_filter: An optional list of filename extensions to filter by
    @return: A list of paths containing the desired model(s)
    """
    output = []

    if ext_filter is None:
        ext_filter = []

    try:
        places = []

        if command_path is not None and command_path != model_path:
            pretrained_path = os.path.join(command_path, 'experiments/pretrained_models')
            if os.path.exists(pretrained_path):
                print(f"Appending path: {pretrained_path}")
                places.append(pretrained_path)
            elif os.path.exists(command_path):
                places.append(command_path)

        places.append(model_path)

        for place in places:
            if os.path.exists(place):
                for file in glob.iglob(place + '**/**', recursive=True):
                    full_path = file
                    if os.path.isdir(full_path):
                        continue
                    if len(ext_filter) != 0:
                        model_name, extension = os.path.splitext(file)
                        if extension not in ext_filter:
                            continue
                    if file not in output:
                        output.append(full_path)

        if model_url is not None and len(output) == 0:
            if download_name is not None:
                dl = load_file_from_url(model_url, model_path, True, download_name)
                output.append(dl)
            else:
                output.append(model_url)

    except Exception:
        pass

    return output
