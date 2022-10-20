''' service: complicated functions called from ui.py are placed here '''

import os
import math
import json
import random
import logging
from PIL import Image, ImageFilter, ImageOps, ImageChops

import piexif
import piexif.helper
import torch
import numpy as np
import cv2
import torch
import numpy as np
from skimage import exposure

from modules.prompt_helper import prompt_parser
from modules.diffuser import sd_samplers
from modules.diffuser.sd_hijack import model_hijack
from modules.cmd_opts import cmd_opts, opts
from modules.runtime import state
from modules import devices
import modules.face_restorer
import modules.images as images
import modules.prompt_helper.styles
from modules import runtime, images, devices
from modules.ui import plaintext_to_html
import modules.face_restorer.gfpgan_model
import modules.face_restorer.codeformer_model
from modules.cmd_opts import opts, state, cmd_opts
import modules.scripts
from modules.ui import plaintext_to_html
from modules.runtime import state

# some of those options should not be changed at all because they would break the model, so I removed them from options.
opt_C = 4
opt_f = 8


def setup_color_correction(image):
    logging.info("Calibrating color correction.")
    return cv2.cvtColor(np.asarray(image.copy()), cv2.COLOR_RGB2LAB)

def apply_color_correction(correction, image):
    logging.info("Applying color correction.")
    image = Image.fromarray(cv2.cvtColor(exposure.match_histograms(
        cv2.cvtColor(
            np.asarray(image),
            cv2.COLOR_RGB2LAB
        ),
        correction,
        channel_axis=2
    ), cv2.COLOR_LAB2RGB).astype("uint8"))

    return image

def slerp(val, low, high):
    # from https://discuss.pytorch.org/t/help-regarding-slerp-function-for-generative-model-sampling/32475/3
    
    low_norm = low/torch.norm(low, dim=1, keepdim=True)
    high_norm = high/torch.norm(high, dim=1, keepdim=True)
    dot = (low_norm*high_norm).sum(1)

    if dot.mean() > 0.9995:
        return low * val + high * (1 - val)

    omega = torch.acos(dot)
    so = torch.sin(omega)
    res = (torch.sin((1.0-val)*omega)/so).unsqueeze(1)*low + (torch.sin(val*omega)/so).unsqueeze(1) * high
    return res

def create_random_tensors(shape, seeds, subseeds=None, subseed_strength=0.0, seed_resize_from_h=0, seed_resize_from_w=0, p=None):
    xs = []

    # if we have multiple seeds, this means we are working with batch size>1; this then
    # enables the generation of additional tensors with noise that the sampler will use during its processing.
    # Using those pre-generated tensors instead of simple torch.randn allows a batch with seeds [100, 101] to
    # produce the same images as with two batches [100], [101].
    if p is not None and p.sampler is not None and len(seeds) > 1 and cmd_opts.enable_batch_seeds:
        sampler_noises = [[] for _ in range(p.sampler.number_of_needed_noises(p))]
    else:
        sampler_noises = None

    for i, seed in enumerate(seeds):
        noise_shape = shape if seed_resize_from_h <= 0 or seed_resize_from_w <= 0 else (shape[0], seed_resize_from_h//8, seed_resize_from_w//8)

        subnoise = None
        if subseeds is not None:
            subseed = 0 if i >= len(subseeds) else subseeds[i]

            subnoise = devices.randn(subseed, noise_shape)

        # randn results depend on device; gpu and cpu get different results for same seed;
        # the way I see it, it's better to do this on CPU, so that everyone gets same result;
        # but the original script had it like this, so I do not dare change it for now because
        # it will break everyone's seeds.
        noise = devices.randn(seed, noise_shape)

        if subnoise is not None:
            noise = slerp(subseed_strength, noise, subnoise)

        if noise_shape != shape:
            x = devices.randn(seed, shape)
            dx = (shape[2] - noise_shape[2]) // 2
            dy = (shape[1] - noise_shape[1]) // 2
            w = noise_shape[2] if dx >= 0 else noise_shape[2] + 2 * dx
            h = noise_shape[1] if dy >= 0 else noise_shape[1] + 2 * dy
            tx = 0 if dx < 0 else dx
            ty = 0 if dy < 0 else dy
            dx = max(-dx, 0)
            dy = max(-dy, 0)

            x[:, ty:ty+h, tx:tx+w] = noise[:, dy:dy+h, dx:dx+w]
            noise = x

        if sampler_noises is not None:
            cnt = p.sampler.number_of_needed_noises(p)

            for j in range(cnt):
                sampler_noises[j].append(devices.randn_without_seed(tuple(noise_shape)))

        xs.append(noise)

    if sampler_noises is not None:
        p.sampler.sampler_noises = [torch.stack(n).to(cmd_opts.device) for n in sampler_noises]

    x = torch.stack(xs).to(cmd_opts.device)
    return x

def get_fixed_seed(seed):
    if seed in [None, '', -1]:
        return int(random.randrange(4294967294))

    return seed

def fix_seed(p):
    p.seed = get_fixed_seed(p.seed)
    p.subseed = get_fixed_seed(p.subseed)

def create_infotext(p, all_prompts, all_seeds, all_subseeds, comments, iteration=0, position_in_batch=0):
    index = position_in_batch + iteration * p.batch_size

    clip_skip = getattr(p, 'clip_skip', cmd_opts.CLIP_stop_at_last_layers)

    generation_params = {
        "Steps": p.steps,
        "Sampler": sd_samplers.samplers(p)[p.sampler_index].name,
        "CFG scale": p.cfg_scale,
        "Seed": all_seeds[index],
        "Face restoration": (cmd_opts.face_restoration_model if p.restore_faces else None),
        "Size": f"{p.width}x{p.height}",
        "Model hash": getattr(p, 'sd_model_hash', None if not cmd_opts.add_model_hash_to_info or not cmd_opts.sd_model.sd_model_hash else cmd_opts.sd_model.sd_model_hash),
        "Model": (None if not cmd_opts.add_model_name_to_info or not cmd_opts.sd_model.sd_checkpoint_info.model_name else cmd_opts.sd_model.sd_checkpoint_info.model_name.replace(',', '').replace(':', '')),
        "Hypernet": (None if cmd_opts.loaded_hypernetwork is None else cmd_opts.loaded_hypernetwork.name.replace(',', '').replace(':', '')),
        "Batch size": (None if p.batch_size < 2 else p.batch_size),
        "Batch pos": (None if p.batch_size < 2 else position_in_batch),
        "Variation seed": (None if p.subseed_strength == 0 else all_subseeds[index]),
        "Variation seed strength": (None if p.subseed_strength == 0 else p.subseed_strength),
        "Seed resize from": (None if p.seed_resize_from_w == 0 or p.seed_resize_from_h == 0 else f"{p.seed_resize_from_w}x{p.seed_resize_from_h}"),
        "Denoising strength": getattr(p, 'denoising_strength', None),
        "Eta": (None if p.sampler is None or p.sampler.eta == p.sampler.default_eta else p.sampler.eta),
        "Clip skip": None if clip_skip <= 1 else clip_skip,
    }

    generation_params.update(p.extra_generation_params)

    generation_params_text = ", ".join([k if k == v else f'{k}: {v}' for k, v in generation_params.items() if v is not None])

    negative_prompt_text = "\nNegative prompt: " + p.negative_prompt if p.negative_prompt else ""

    return f"{all_prompts[index]}{negative_prompt_text}\n{generation_params_text}".strip()

def get_crop_region(mask, pad=0):
    """finds a rectangular region that contains all masked ares in an image. Returns (x1, y1, x2, y2) coordinates of the rectangle.
    For example, if a user has painted the top-right part of a 512x512 image", the result may be (256, 0, 512, 256)"""
    
    h, w = mask.shape

    crop_left = 0
    for i in range(w):
        if not (mask[:, i] == 0).all():
            break
        crop_left += 1

    crop_right = 0
    for i in reversed(range(w)):
        if not (mask[:, i] == 0).all():
            break
        crop_right += 1

    crop_top = 0
    for i in range(h):
        if not (mask[i] == 0).all():
            break
        crop_top += 1

    crop_bottom = 0
    for i in reversed(range(h)):
        if not (mask[i] == 0).all():
            break
        crop_bottom += 1

    return (
        int(max(crop_left-pad, 0)),
        int(max(crop_top-pad, 0)),
        int(min(w - crop_right + pad, w)),
        int(min(h - crop_bottom + pad, h))
    )

def expand_crop_region(crop_region, processing_width, processing_height, image_width, image_height):
    """expands crop region get_crop_region() to match the ratio of the image the region will processed in; returns expanded region
    for example, if user drew mask in a 128x32 region, and the dimensions for processing are 512x512, the region will be expanded to 128x128."""

    x1, y1, x2, y2 = crop_region

    ratio_crop_region = (x2 - x1) / (y2 - y1)
    ratio_processing = processing_width / processing_height

    if ratio_crop_region > ratio_processing:
        desired_height = (x2 - x1) * ratio_processing
        desired_height_diff = int(desired_height - (y2-y1))
        y1 -= desired_height_diff//2
        y2 += desired_height_diff - desired_height_diff//2
        if y2 >= image_height:
            diff = y2 - image_height
            y2 -= diff
            y1 -= diff
        if y1 < 0:
            y2 -= y1
            y1 -= y1
        if y2 >= image_height:
            y2 = image_height
    else:
        desired_width = (y2 - y1) * ratio_processing
        desired_width_diff = int(desired_width - (x2-x1))
        x1 -= desired_width_diff//2
        x2 += desired_width_diff - desired_width_diff//2
        if x2 >= image_width:
            diff = x2 - image_width
            x2 -= diff
            x1 -= diff
        if x1 < 0:
            x2 -= x1
            x1 -= x1
        if x2 >= image_width:
            x2 = image_width

    return x1, y1, x2, y2

def fill(image, mask):
    """fills masked regions with colors from image using blur. Not extremely effective."""

    image_mod = Image.new('RGBA', (image.width, image.height))

    image_masked = Image.new('RGBa', (image.width, image.height))
    image_masked.paste(image.convert("RGBA").convert("RGBa"), mask=ImageOps.invert(mask.convert('L')))

    image_masked = image_masked.convert('RGBa')

    for radius, repeats in [(256, 1), (64, 1), (16, 2), (4, 4), (2, 2), (0, 1)]:
        blurred = image_masked.filter(ImageFilter.GaussianBlur(radius)).convert('RGBA')
        for _ in range(repeats):
            image_mod.alpha_composite(blurred)

    return image_mod.convert("RGB")


class StableDiffusionProcessing:
    def __init__(self, sd_model=None, outpath_samples=None, outpath_grids=None, prompt="", styles=None, 
                 seed=-1, subseed=-1, subseed_strength=0, seed_resize_from_h=-1, seed_resize_from_w=-1, seed_enable_extras=True, 
                 sampler_index=0, batch_size=1, n_iter=1, steps=50, cfg_scale=7.0, width=512, height=512, restore_faces=False, 
                 tiling=False, do_not_save_samples=False, do_not_save_grid=False, extra_generation_params=None, 
                 overlay_images=None, negative_prompt=None, eta=None):
        self.sd_model = sd_model
        self.outpath_samples: str = outpath_samples
        self.outpath_grids: str = outpath_grids
        self.prompt: str = prompt
        self.prompt_for_display: str = None
        self.negative_prompt: str = (negative_prompt or "")
        self.styles: list = styles or []
        self.seed: int = seed
        self.subseed: int = subseed
        self.subseed_strength: float = subseed_strength
        self.seed_resize_from_h: int = seed_resize_from_h
        self.seed_resize_from_w: int = seed_resize_from_w
        self.sampler_index: int = sampler_index
        self.batch_size: int = batch_size
        self.n_iter: int = n_iter
        self.steps: int = steps
        self.cfg_scale: float = cfg_scale
        self.width: int = width
        self.height: int = height
        self.restore_faces: bool = restore_faces
        self.tiling: bool = tiling
        self.do_not_save_samples: bool = do_not_save_samples
        self.do_not_save_grid: bool = do_not_save_grid
        self.extra_generation_params: dict = extra_generation_params or {}
        self.overlay_images = overlay_images
        self.eta = eta
        self.paste_to = None
        self.color_corrections = None
        self.denoising_strength: float = 0
        self.sampler_noise_scheduler_override = None
        self.ddim_discretize = cmd_opts.ddim_discretize
        self.s_churn = cmd_opts.s_churn
        self.s_tmin = cmd_opts.s_tmin
        self.s_tmax = float('inf')  # not representable as a standard ui option
        self.s_noise = cmd_opts.s_noise

        if not seed_enable_extras:
            self.subseed = -1
            self.subseed_strength = 0
            self.seed_resize_from_h = 0
            self.seed_resize_from_w = 0

    def init(self, all_prompts, all_seeds, all_subseeds):
        raise NotImplementedError

    def sample(self, conditioning, unconditional_conditioning, seeds, subseeds, subseed_strength):
        raise NotImplementedError

class StableDiffusionProcessingTxt2Img(StableDiffusionProcessing):
    sampler = None
    firstphase_width = 0
    firstphase_height = 0
    firstphase_width_truncated = 0
    firstphase_height_truncated = 0

    def __init__(self, enable_hr=False, scale_latent=True, denoising_strength=0.75, **kwargs):
        super().__init__(**kwargs)

        self.enable_hr = enable_hr
        self.scale_latent = scale_latent
        self.denoising_strength = denoising_strength

    def init(self, all_prompts, all_seeds, all_subseeds):
        if self.enable_hr:
            if state.job_count == -1:
                state.job_count = self.n_iter * 2
            else:
                state.job_count = state.job_count * 2

            desired_pixel_count = 512 * 512
            actual_pixel_count = self.width * self.height
            scale = math.sqrt(desired_pixel_count / actual_pixel_count)

            self.firstphase_width = math.ceil(scale * self.width / 64) * 64
            self.firstphase_height = math.ceil(scale * self.height / 64) * 64
            self.firstphase_width_truncated = int(scale * self.width)
            self.firstphase_height_truncated = int(scale * self.height)

    def sample(self, conditioning, unconditional_conditioning, seeds, subseeds, subseed_strength):
        self.sampler = sd_samplers.create_sampler_with_index(sd_samplers.samplers, self.sampler_index, self.sd_model)

        if not self.enable_hr:
            x = create_random_tensors([opt_C, self.height // opt_f, self.width // opt_f], seeds=seeds, subseeds=subseeds, subseed_strength=self.subseed_strength, seed_resize_from_h=self.seed_resize_from_h, seed_resize_from_w=self.seed_resize_from_w, p=self)
            samples = self.sampler.sample(self, x, conditioning, unconditional_conditioning)
            return samples

        # use highres fix to fix deformality
        x = create_random_tensors([opt_C, self.firstphase_height // opt_f, self.firstphase_width // opt_f], seeds=seeds, subseeds=subseeds, subseed_strength=self.subseed_strength, seed_resize_from_h=self.seed_resize_from_h, seed_resize_from_w=self.seed_resize_from_w, p=self)
        samples = self.sampler.sample(self, x, conditioning, unconditional_conditioning)

        truncate_x = (self.firstphase_width - self.firstphase_width_truncated) // opt_f
        truncate_y = (self.firstphase_height - self.firstphase_height_truncated) // opt_f

        samples = samples[:, :, truncate_y//2:samples.shape[2]-truncate_y//2, truncate_x//2:samples.shape[3]-truncate_x//2]

        if self.scale_latent:
            samples = torch.nn.functional.interpolate(samples, size=(self.height // opt_f, self.width // opt_f), mode="bilinear")
        else:
            decoded_samples = self.sd_model.decode_first_stage(samples)

            if opts.upscaler_for_img2img in [None, "None"]:
                decoded_samples = torch.nn.functional.interpolate(decoded_samples, size=(self.height, self.width), mode="bilinear")
            else:
                lowres_samples = torch.clamp((decoded_samples + 1.0) / 2.0, min=0.0, max=1.0)

                batch_images = []
                for i, x_sample in enumerate(lowres_samples):
                    x_sample = 255. * np.moveaxis(x_sample.cpu().numpy(), 0, 2)
                    x_sample = x_sample.astype(np.uint8)
                    image = Image.fromarray(x_sample)
                    image = images.resize_image(0, image, self.width, self.height)
                    image = np.array(image).astype(np.float32) / 255.0
                    image = np.moveaxis(image, 2, 0)
                    batch_images.append(image)

                decoded_samples = torch.from_numpy(np.array(batch_images))
                decoded_samples = decoded_samples.to(cmd_opts.device)
                decoded_samples = 2. * decoded_samples - 1.

            samples = self.sd_model.get_first_stage_encoding(self.sd_model.encode_first_stage(decoded_samples))

        state.nextjob()

        #self.sampler = sd_samplers.create_sampler_with_index(sd_samplers.samplers, self.sampler_index, self.sd_model)

        noise = create_random_tensors(samples.shape[1:], seeds=seeds, subseeds=subseeds, subseed_strength=subseed_strength, seed_resize_from_h=self.seed_resize_from_h, seed_resize_from_w=self.seed_resize_from_w, p=self)

        # GC now before running the next img2img to prevent running out of memory
        del x ; devices.torch_gc()

        samples = self.sampler.sample_img2img(self, samples, noise, conditioning, unconditional_conditioning, steps=self.steps)
        return samples

class StableDiffusionProcessingImg2Img(StableDiffusionProcessing):
    sampler = None

    def __init__(self, init_images=None, resize_mode=0, denoising_strength=0.75, mask=None, mask_blur=4, inpainting_fill=0, inpaint_full_res=True, inpaint_full_res_padding=0, inpainting_mask_invert=0, **kwargs):
        super().__init__(**kwargs)

        self.init_images = init_images
        self.resize_mode: int = resize_mode
        self.denoising_strength: float = denoising_strength
        self.init_latent = None
        self.image_mask = mask
        #self.image_unblurred_mask = None
        self.latent_mask = None
        self.mask_for_overlay = None
        self.mask_blur = mask_blur
        self.inpainting_fill = inpainting_fill
        self.inpaint_full_res = inpaint_full_res
        self.inpaint_full_res_padding = inpaint_full_res_padding
        self.inpainting_mask_invert = inpainting_mask_invert
        self.mask = None
        self.nmask = None

    def init(self, all_prompts, all_seeds, all_subseeds):
        self.sampler = sd_samplers.create_sampler_with_index(sd_samplers.samplers_for_img2img, self.sampler_index, self.sd_model)
        crop_region = None

        if self.image_mask is not None:
            self.image_mask = self.image_mask.convert('L')

            if self.inpainting_mask_invert:
                self.image_mask = ImageOps.invert(self.image_mask)

            #self.image_unblurred_mask = self.image_mask

            if self.mask_blur > 0:
                self.image_mask = self.image_mask.filter(ImageFilter.GaussianBlur(self.mask_blur))

            if self.inpaint_full_res:
                self.mask_for_overlay = self.image_mask
                mask = self.image_mask.convert('L')
                crop_region = masking.get_crop_region(np.array(mask), self.inpaint_full_res_padding)
                crop_region = masking.expand_crop_region(crop_region, self.width, self.height, mask.width, mask.height)
                x1, y1, x2, y2 = crop_region

                mask = mask.crop(crop_region)
                self.image_mask = images.resize_image(2, mask, self.width, self.height)
                self.paste_to = (x1, y1, x2-x1, y2-y1)
            else:
                self.image_mask = images.resize_image(self.resize_mode, self.image_mask, self.width, self.height)
                np_mask = np.array(self.image_mask)
                np_mask = np.clip((np_mask.astype(np.float32)) * 2, 0, 255).astype(np.uint8)
                self.mask_for_overlay = Image.fromarray(np_mask)

            self.overlay_images = []

        latent_mask = self.latent_mask if self.latent_mask is not None else self.image_mask

        add_color_corrections = cmd_opts.img2img_color_correction and self.color_corrections is None
        if add_color_corrections:
            self.color_corrections = []
        imgs = []
        for img in self.init_images:
            image = img.convert("RGB")

            if crop_region is None:
                image = images.resize_image(self.resize_mode, image, self.width, self.height)

            if self.image_mask is not None:
                image_masked = Image.new('RGBa', (image.width, image.height))
                image_masked.paste(image.convert("RGBA").convert("RGBa"), mask=ImageOps.invert(self.mask_for_overlay.convert('L')))

                self.overlay_images.append(image_masked.convert('RGBA'))

            if crop_region is not None:
                image = image.crop(crop_region)
                image = images.resize_image(2, image, self.width, self.height)

            if self.image_mask is not None:
                if self.inpainting_fill != 1:
                    image = fill(image, latent_mask)

            if add_color_corrections:
                self.color_corrections.append(setup_color_correction(image))

            image = np.array(image).astype(np.float32) / 255.0
            image = np.moveaxis(image, 2, 0)

            imgs.append(image)

        if len(imgs) == 1:
            batch_images = np.expand_dims(imgs[0], axis=0).repeat(self.batch_size, axis=0)
            if self.overlay_images is not None:
                self.overlay_images = self.overlay_images * self.batch_size
        elif len(imgs) <= self.batch_size:
            self.batch_size = len(imgs)
            batch_images = np.array(imgs)
        else:
            raise RuntimeError(f"bad number of images passed: {len(imgs)}; expecting {self.batch_size} or less")

        image = torch.from_numpy(batch_images)
        image = 2. * image - 1.
        image = image.to(cmd_opts.device)

        self.init_latent = self.sd_model.get_first_stage_encoding(self.sd_model.encode_first_stage(image))

        if self.image_mask is not None:
            init_mask = latent_mask
            latmask = init_mask.convert('RGB').resize((self.init_latent.shape[3], self.init_latent.shape[2]))
            latmask = np.moveaxis(np.array(latmask, dtype=np.float32), 2, 0) / 255
            latmask = latmask[0]
            latmask = np.around(latmask)
            latmask = np.tile(latmask[None], (4, 1, 1))

            self.mask = torch.asarray(1.0 - latmask).to(cmd_opts.device).type(self.sd_model.dtype)
            self.nmask = torch.asarray(latmask).to(cmd_opts.device).type(self.sd_model.dtype)

            # this needs to be fixed to be done in sample() using actual seeds for batches
            if self.inpainting_fill == 2:
                self.init_latent = self.init_latent * self.mask + create_random_tensors(self.init_latent.shape[1:], all_seeds[0:self.init_latent.shape[0]]) * self.nmask
            elif self.inpainting_fill == 3:
                self.init_latent = self.init_latent * self.mask

    def sample(self, conditioning, unconditional_conditioning, seeds, subseeds, subseed_strength):
        x = create_random_tensors([opt_C, self.height // opt_f, self.width // opt_f], seeds=seeds, subseeds=subseeds, subseed_strength=self.subseed_strength, seed_resize_from_h=self.seed_resize_from_h, seed_resize_from_w=self.seed_resize_from_w, p=self)

        samples = self.sampler.sample_img2img(self, self.init_latent, x, conditioning, unconditional_conditioning)

        if self.mask is not None:
            samples = samples * self.nmask + self.init_latent * self.mask

        del x ; devices.torch_gc()

        return samples

class Processed:
    def __init__(self, p: StableDiffusionProcessing, images_list, seed=-1, info="", 
                 subseed=None, all_prompts=None, all_seeds=None, all_subseeds=None, 
                 index_of_first_image=0, infotexts=None):
        self.images = images_list
        self.prompt = p.prompt
        self.negative_prompt = p.negative_prompt
        self.seed = seed
        self.subseed = subseed
        self.subseed_strength = p.subseed_strength
        self.info = info
        self.width = p.width
        self.height = p.height
        self.sampler_index = p.sampler_index
        self.sampler = sd_samplers.samplers[p.sampler_index].name
        self.cfg_scale = p.cfg_scale
        self.steps = p.steps
        self.batch_size = p.batch_size
        self.restore_faces = p.restore_faces
        self.face_restoration_model = cmd_opts.face_restoration_model if p.restore_faces else None
        self.sd_model_hash = cmd_opts.sd_model.sd_model_hash
        self.seed_resize_from_w = p.seed_resize_from_w
        self.seed_resize_from_h = p.seed_resize_from_h
        self.denoising_strength = getattr(p, 'denoising_strength', None)
        self.extra_generation_params = p.extra_generation_params
        self.index_of_first_image = index_of_first_image
        self.styles = p.styles
        self.job_timestamp = state.job_timestamp
        self.clip_skip = cmd_opts.CLIP_stop_at_last_layers

        self.eta = p.eta
        self.ddim_discretize = p.ddim_discretize
        self.s_churn = p.s_churn
        self.s_tmin = p.s_tmin
        self.s_tmax = p.s_tmax
        self.s_noise = p.s_noise
        self.sampler_noise_scheduler_override = p.sampler_noise_scheduler_override
        self.prompt = self.prompt if type(self.prompt) != list else self.prompt[0]
        self.negative_prompt = self.negative_prompt if type(self.negative_prompt) != list else self.negative_prompt[0]
        self.seed = int(self.seed if type(self.seed) != list else self.seed[0])
        self.subseed = int(self.subseed if type(self.subseed) != list else self.subseed[0]) if self.subseed is not None else -1

        self.all_prompts = all_prompts or [self.prompt]
        self.all_seeds = all_seeds or [self.seed]
        self.all_subseeds = all_subseeds or [self.subseed]
        self.infotexts = infotexts or [info]

    def js(self):
        obj = {
            "prompt": self.prompt,
            "all_prompts": self.all_prompts,
            "negative_prompt": self.negative_prompt,
            "seed": self.seed,
            "all_seeds": self.all_seeds,
            "subseed": self.subseed,
            "all_subseeds": self.all_subseeds,
            "subseed_strength": self.subseed_strength,
            "width": self.width,
            "height": self.height,
            "sampler_index": self.sampler_index,
            "sampler": self.sampler,
            "cfg_scale": self.cfg_scale,
            "steps": self.steps,
            "batch_size": self.batch_size,
            "restore_faces": self.restore_faces,
            "face_restoration_model": self.face_restoration_model,
            "sd_model_hash": self.sd_model_hash,
            "seed_resize_from_w": self.seed_resize_from_w,
            "seed_resize_from_h": self.seed_resize_from_h,
            "denoising_strength": self.denoising_strength,
            "extra_generation_params": self.extra_generation_params,
            "index_of_first_image": self.index_of_first_image,
            "infotexts": self.infotexts,
            "styles": self.styles,
            "job_timestamp": self.job_timestamp,
            "clip_skip": self.clip_skip,
        }

        return json.dumps(obj)

    def infotext(self,  p: StableDiffusionProcessing, index):
        return create_infotext(p, self.all_prompts, self.all_seeds, self.all_subseeds, comments=[], 
                               position_in_batch=index % self.batch_size, iteration=index // self.batch_size)

def process_images(p: StableDiffusionProcessing) -> Processed:
    """this is the main loop that both txt2img and img2img use; it calls func_init once inside all the scopes and func_sample once per batch"""

    if type(p.prompt) == list:
        assert(len(p.prompt) > 0)
    else:
        assert p.prompt is not None

    devices.torch_gc()

    seed = get_fixed_seed(p.seed)
    subseed = get_fixed_seed(p.subseed)

    if p.outpath_samples is not None:
        os.makedirs(p.outpath_samples, exist_ok=True)

    if p.outpath_grids is not None:
        os.makedirs(p.outpath_grids, exist_ok=True)

    model_hijack.apply_circular(p.tiling)
    model_hijack.clear_comments()

    comments = {}

    cmd_opts.prompt_styles.apply_styles(p)

    if type(p.prompt) == list:
        all_prompts = p.prompt
    else:
        all_prompts = p.batch_size * p.n_iter * [p.prompt]

    if type(seed) == list:
        all_seeds = seed
    else:
        all_seeds = [int(seed) + (x if p.subseed_strength == 0 else 0) for x in range(len(all_prompts))]

    if type(subseed) == list:
        all_subseeds = subseed
    else:
        all_subseeds = [int(subseed) + x for x in range(len(all_prompts))]

    def infotext(iteration=0, position_in_batch=0):
        return create_infotext(p, all_prompts, all_seeds, all_subseeds, comments, iteration, position_in_batch)

    if os.path.exists(cmd_opts.embeddings_dir):
        model_hijack.embedding_db.load_textual_inversion_embeddings()

    infotexts = []
    output_images = []

    with torch.no_grad(), p.sd_model.ema_scope():
        with devices.autocast():
            p.init(all_prompts, all_seeds, all_subseeds)

        if state.job_count == -1:
            state.job_count = p.n_iter

        for n in range(p.n_iter):
            if state.skipped:
                state.skipped = False
            
            if state.interrupted:
                break

            prompts  = all_prompts [n * p.batch_size:(n + 1) * p.batch_size]
            seeds    = all_seeds   [n * p.batch_size:(n + 1) * p.batch_size]
            subseeds = all_subseeds[n * p.batch_size:(n + 1) * p.batch_size]

            if (len(prompts) == 0):
                break

            #uc = p.sd_model.get_learned_conditioning(len(prompts) * [p.negative_prompt])
            #c = p.sd_model.get_learned_conditioning(prompts)
            with devices.autocast():
                uc = prompt_parser.get_learned_conditioning(cmd_opts.sd_model, len(prompts) * [p.negative_prompt], p.steps)
                c = prompt_parser.get_multicond_learned_conditioning(cmd_opts.sd_model, prompts, p.steps)

            if len(model_hijack.comments) > 0:
                for comment in model_hijack.comments:
                    comments[comment] = 1

            if p.n_iter > 1:
                cmd_opts.state.job = f"Batch {n+1} out of {p.n_iter}"

            with devices.autocast():
                samples_ddim = p.sample(conditioning=c, unconditional_conditioning=uc, seeds=seeds, subseeds=subseeds, subseed_strength=p.subseed_strength)

            if state.interrupted or state.skipped:

                # if we are interrupted, sample returns just noise
                # use the image collected previously in sampler loop
                samples_ddim = cmd_opts.state.current_latent

            samples_ddim = samples_ddim.to(devices.dtype)

            x_samples_ddim = p.sd_model.decode_first_stage(samples_ddim)
            x_samples_ddim = torch.clamp((x_samples_ddim + 1.0) / 2.0, min=0.0, max=1.0)

            del samples_ddim

            devices.torch_gc()

            if cmd_opts.filter_nsfw:
                import modules.safety as safety
                x_samples_ddim = modules.safety.censor_batch(x_samples_ddim)

            for i, x_sample in enumerate(x_samples_ddim):
                x_sample = 255. * np.moveaxis(x_sample.cpu().numpy(), 0, 2)
                x_sample = x_sample.astype(np.uint8)

                if p.restore_faces:
                    if cmd_opts.save and not p.do_not_save_samples and cmd_opts.save_images_before_face_restoration:
                        images.save_image(Image.fromarray(x_sample), p.outpath_samples, "", seeds[i], prompts[i], cmd_opts.samples_format, info=infotext(n, i), p=p, suffix="-before-face-restoration")

                    devices.torch_gc()

                    x_sample = modules.face_restorer.restore_faces(x_sample)
                    devices.torch_gc()

                image = Image.fromarray(x_sample)

                if p.color_corrections is not None and i < len(p.color_corrections):
                    image = apply_color_correction(p.color_corrections[i], image)

                if p.overlay_images is not None and i < len(p.overlay_images):
                    overlay = p.overlay_images[i]

                    if p.paste_to is not None:
                        x, y, w, h = p.paste_to
                        base_image = Image.new('RGBA', (overlay.width, overlay.height))
                        image = images.resize_image(1, image, w, h)
                        base_image.paste(image, (x, y))
                        image = base_image

                    image = image.convert('RGBA')
                    image.alpha_composite(overlay)
                    image = image.convert('RGB')

                images.save_image(image, p.outpath_samples, "", seeds[i], prompts[i], cmd_opts.samples_format, info=infotext(n, i), p=p)

                text = infotext(n, i)
                infotexts.append(text)
                image.info["parameters"] = text
                output_images.append(image)

            del x_samples_ddim
            devices.torch_gc()

            state.nextjob()

        p.color_corrections = None

        index_of_first_image = 0
        
        if len(output_images) >= 2:
            grid = images.image_grid(output_images, p.batch_size)

            text = infotext()
            infotexts.insert(0, text)
            grid.info["parameters"] = text
            output_images.insert(0, grid)
            index_of_first_image = 1

            images.save_image(grid, p.outpath_grids, "grid", all_seeds[0], all_prompts[0], cmd_opts.grid_format, info=infotext(), short_filename=True, p=p, grid=True)

    devices.torch_gc()
    return Processed(p, output_images, all_seeds[0], infotext() + "".join(["\n\n" + x for x in comments]), subseed=all_subseeds[0], 
                     all_prompts=all_prompts, all_seeds=all_seeds, all_subseeds=all_subseeds, index_of_first_image=index_of_first_image, infotexts=infotexts)


def img2img(mode: int, prompt: str, negative_prompt: str, prompt_style: str, prompt_style2: str, 
            init_img, init_img_with_mask, init_img_inpaint, init_mask_inpaint, mask_mode, steps: int, 
            sampler_index: int, mask_blur: int, inpainting_fill: int, restore_faces: bool, tiling: bool, 
            n_iter: int, batch_size: int, cfg_scale: float, denoising_strength: float, 
            seed: int, subseed: int, subseed_strength: float, seed_resize_from_h: int, seed_resize_from_w: int, 
            seed_enable_extras: bool, height: int, width: int, resize_mode: int, inpaint_full_res: bool, 
            inpaint_full_res_padding: int, inpainting_mask_invert: int, img2img_batch_input_dir: str, 
            img2img_batch_output_dir: str, *args):
    
    is_inpaint = mode == 1
    is_batch = mode == 2

    if is_inpaint:
        if mask_mode == 0:
            image = init_img_with_mask['image']
            mask = init_img_with_mask['mask']
            alpha_mask = ImageOps.invert(image.split()[-1]).convert('L').point(lambda x: 255 if x > 0 else 0, mode='1')
            mask = ImageChops.lighter(alpha_mask, mask.convert('L')).convert('L')
            image = image.convert('RGB')
        else:
            image = init_img_inpaint
            mask = init_mask_inpaint
    else:
        image = init_img
        mask = None

    assert 0. <= denoising_strength <= 1., 'can only work with strength in [0.0, 1.0]'

    p = StableDiffusionProcessingImg2Img(
        sd_model=cmd_opts.sd_model,
        outpath_samples=cmd_opts.outdir_samples or cmd_opts.outdir_img2img_samples,
        outpath_grids=cmd_opts.outdir_grids or cmd_opts.outdir_img2img_grids,
        prompt=prompt,
        negative_prompt=negative_prompt,
        styles=[prompt_style, prompt_style2],
        seed=seed,
        subseed=subseed,
        subseed_strength=subseed_strength,
        seed_resize_from_h=seed_resize_from_h,
        seed_resize_from_w=seed_resize_from_w,
        seed_enable_extras=seed_enable_extras,
        sampler_index=sampler_index,
        batch_size=batch_size,
        n_iter=n_iter,
        steps=steps,
        cfg_scale=cfg_scale,
        width=width,
        height=height,
        restore_faces=restore_faces,
        tiling=tiling,
        init_images=[image],
        mask=mask,
        mask_blur=mask_blur,
        inpainting_fill=inpainting_fill,
        resize_mode=resize_mode,
        denoising_strength=denoising_strength,
        inpaint_full_res=inpaint_full_res,
        inpaint_full_res_padding=inpaint_full_res_padding,
        inpainting_mask_invert=inpainting_mask_invert,
    )

    print(f"\nimg2img: {prompt}", file=cmd_opts.progress_print_out)

    p.extra_generation_params["Mask blur"] = mask_blur

    if is_batch:
        assert not cmd_opts.cmd_opts.hide_ui_dir_config, "Launched with --hide-ui-dir-config, batch img2img disabled"

        process_batch(p, img2img_batch_input_dir, img2img_batch_output_dir, args)

        processed = Processed(p, [], p.seed, "")
    else:
        processed = modules.scripts.scripts_img2img.run(p, *args)
        if processed is None:
            processed = process_images(p)

    cmd_opts.total_tqdm.clear()

    generation_info_js = processed.js()
    if cmd_opts.samples_log_stdout:
        print(generation_info_js)

    if cmd_opts.do_not_show_images:
        processed.images = []

    return processed.images, generation_info_js, plaintext_to_html(processed.info)

def txt2img(prompt: str, negative_prompt: str, prompt_style: str, prompt_style2: str, steps: int, 
            sampler_index: int, restore_faces: bool, tiling: bool, n_iter: int, batch_size: int, cfg_scale: float, 
            seed: int, subseed: int, subseed_strength: float, seed_resize_from_h: int, seed_resize_from_w: int, 
            seed_enable_extras: bool, height: int, width: int, enable_hr: bool, scale_latent: bool, 
            denoising_strength: float, *args):
    
    p = StableDiffusionProcessingTxt2Img(
        sd_model=cmd_opts.sd_model,
        outpath_samples=cmd_opts.outdir_samples or cmd_opts.outdir_txt2img_samples,
        outpath_grids=cmd_opts.outdir_grids or cmd_opts.outdir_txt2img_grids,
        prompt=prompt,
        styles=[prompt_style, prompt_style2],
        negative_prompt=negative_prompt,
        seed=seed,
        subseed=subseed,
        subseed_strength=subseed_strength,
        seed_resize_from_h=seed_resize_from_h,
        seed_resize_from_w=seed_resize_from_w,
        seed_enable_extras=seed_enable_extras,
        sampler_index=sampler_index,
        batch_size=batch_size,
        n_iter=n_iter,
        steps=steps,
        cfg_scale=cfg_scale,
        width=width,
        height=height,
        restore_faces=restore_faces,
        tiling=tiling,
        enable_hr=enable_hr,
        scale_latent=scale_latent if enable_hr else None,
        denoising_strength=denoising_strength if enable_hr else None,
    )

    print(f"\ntxt2img: {prompt}", file=cmd_opts.progress_print_out)

    processed = modules.scripts.script_runner.run(p, *args)

    if processed is None:
        processed = process_images(p)

    cmd_opts.total_tqdm.clear()

    generation_info_js = processed.js()
    if cmd_opts.samples_log_stdout:
        print(generation_info_js)

    if cmd_opts.do_not_show_images:
        processed.images = []

    return processed.images, generation_info_js, plaintext_to_html(processed.info)

def run_extras(extras_mode, image, image_folder, gfpgan_visibility, codeformer_visibility, codeformer_weight, 
               upscaling_resize, extras_upscaler_1, extras_upscaler_2, extras_upscaler_2_visibility):
    devices.torch_gc()
    cached_images = {}

    imageArr = []
    # Also keep track of original file names
    imageNameArr = []

    if extras_mode == 1:
        #convert file to pillow image
        for img in image_folder:
            image = Image.open(img)
            imageArr.append(image)
            imageNameArr.append(os.path.splitext(img.orig_name)[0])
    else:
        imageArr.append(image)
        imageNameArr.append(None)

    outpath = opts.outdir_samples or opts.outdir_extras_samples

    outputs = []
    for image, image_name in zip(imageArr, imageNameArr):
        if image is None:
            return outputs, "Please select an input image.", ''
        existing_pnginfo = image.info or {}

        image = image.convert("RGB")
        info = ""

        if gfpgan_visibility > 0:
            restored_img = modules.face_restorer.gfpgan_model.gfpgan_fix_faces(np.array(image, dtype=np.uint8))
            res = Image.fromarray(restored_img)

            if gfpgan_visibility < 1.0:
                res = Image.blend(image, res, gfpgan_visibility)

            info += f"GFPGAN visibility:{round(gfpgan_visibility, 2)}\n"
            image = res

        if codeformer_visibility > 0:
            restored_img = modules.face_restorer.codeformer.restore(np.array(image, dtype=np.uint8), w=codeformer_weight)
            res = Image.fromarray(restored_img)

            if codeformer_visibility < 1.0:
                res = Image.blend(image, res, codeformer_visibility)

            info += f"CodeFormer w: {round(codeformer_weight, 2)}, CodeFormer visibility:{round(codeformer_visibility, 2)}\n"
            image = res

        if upscaling_resize != 1.0:
            def upscale(image, scaler_index, resize):
                small = image.crop((image.width // 2, image.height // 2, image.width // 2 + 10, image.height // 2 + 10))
                pixels = tuple(np.array(small).flatten().tolist())
                key = (resize, scaler_index, image.width, image.height, gfpgan_visibility, codeformer_visibility, codeformer_weight) + pixels

                c = cached_images.get(key)
                if c is None:
                    upscaler = runtime.sd_upscalers[scaler_index]
                    c = upscaler.scaler.upscale(image, resize, upscaler.data_path)
                    cached_images[key] = c

                return c

            info += f"Upscale: {round(upscaling_resize, 3)}, model:{runtime.sd_upscalers[extras_upscaler_1].name}\n"
            res = upscale(image, extras_upscaler_1, upscaling_resize)

            if extras_upscaler_2 != 0 and extras_upscaler_2_visibility > 0:
                res2 = upscale(image, extras_upscaler_2, upscaling_resize)
                info += f"Upscale: {round(upscaling_resize, 3)}, visibility: {round(extras_upscaler_2_visibility, 3)}, model:{runtime.sd_upscalers[extras_upscaler_2].name}\n"
                res = Image.blend(res, res2, extras_upscaler_2_visibility)

            image = res

        while len(cached_images) > 2:
            del cached_images[next(iter(cached_images.keys()))]

        images.save_image(image, path=outpath, basename="", seed=None, prompt=None, extension=opts.samples_format, info=info, short_filename=True,
                          no_prompt=True, grid=False, pnginfo_section_name="extras", existing_info=existing_pnginfo,
                          forced_filename=image_name if opts.use_original_name_batch else None)

        if opts.enable_pnginfo:
            image.info = existing_pnginfo
            image.info["extras"] = info

        outputs.append(image)

    devices.torch_gc()

    return outputs, plaintext_to_html(info), ''

def run_pnginfo(image):
    if image is None:
        return '', '', ''

    items = image.info
    geninfo = ''

    if "exif" in image.info:
        exif = piexif.load(image.info["exif"])
        exif_comment = (exif or {}).get("Exif", {}).get(piexif.ExifIFD.UserComment, b'')
        try:
            exif_comment = piexif.helper.UserComment.load(exif_comment)
        except ValueError:
            exif_comment = exif_comment.decode('utf8', errors="ignore")

        items['exif comment'] = exif_comment
        geninfo = exif_comment

        for field in ['jfif', 'jfif_version', 'jfif_unit', 'jfif_density', 'dpi', 'exif', 'loop', 'background', 'timestamp', 'duration']:
            items.pop(field, None)

    geninfo = items.get('parameters', geninfo)

    info = ''
    for key, text in items.items():
        info += f"""
<div>
<p><b>{plaintext_to_html(str(key))}</b></p>
<p>{plaintext_to_html(str(text))}</p>
</div>
""".strip()+"\n"

    if len(info) == 0:
        message = "Nothing found in the image."
        info = f"<div><p>{message}<p></div>"

    return geninfo, info
