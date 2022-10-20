''' view: UI & UX '''

import os
import re
import sys
import io
import html
import json
import mimetypes
import base64
import random
import time
import traceback
from functools import reduce
from PIL import Image

import gradio as gr
import gradio.utils
import gradio.routes

from modules.paths import STATIC_PATH
from modules.interrogator.deepbooru import get_deepbooru_tags
from modules.diffuser.sd_samplers import samplers
from modules.diffuser.sd_hijack import model_hijack
import modules.upsampler.ldsr_model
import modules.face_restorer.gfpgan_model
import modules.face_restorer.codeformer_model
import modules.prompt_helper.styles
import modules.processing
from modules.prompt_helper import prompt_parser
from modules.images import save_image
import modules.textual_inverter.ui

from modules import ui
from modules import devices
from modules import runtime
from modules.cmd_opts import cmd_opts, opts
from modules.diffuser import sd_model, hypernetwork
from modules.runtime import lock, state


def gr_show(visible=True):
    return { "visible": visible, "__type__": "update" }


def get_javascript() -> str:
    # main script
    with open(os.path.join(STATIC_PATH, 'script.js')) as fh:
        javascript = f'<script>{fh.read()}</script>'
    
    # user defined script (optional)
    try:
        with open(os.path.join(STATIC_PATH, 'script-user.js')) as fh:
            javascript += f"\n<script>{fh.read()}</script>"
    except: pass

    # other script
    loaded = { 'script.js', 'script-user.js' }
    for fn in sorted(STATIC_PATH):
        if not fn.endswith('.js'): continue
        if fn in loaded: continue
        with open(os.path.join(STATIC_PATH, fn)) as fh:
            javascript += f"\n<script>{fh.read()}</script>"
        loaded.add(fn)

    return javascript


def get_css() -> str:
    # main css
    with open(os.path.join(STATIC_PATH, 'style.css')) as fh:
        css = fh.read()

    # user defined script (optional)
    try:
        with open(os.path.join(STATIC_PATH, 'style-user.css')) as fh:
            css += fh.read()
    except: pass

    # other css
    loaded = { 'style.css', 'style-user.css' }
    for fn in sorted(STATIC_PATH):
        if not fn.endswith('.css'): continue
        if fn in loaded: continue
        with open(os.path.join(STATIC_PATH, fn)) as fh:
            css += fh.read()
        loaded.add(fn)

    return css


re_param_code = r"\s*([\w ]+):\s*([^,]+)(?:,|$)"
re_param = re.compile(re_param_code)
re_params = re.compile(r"^(?:" + re_param_code + "){3,}$")
re_imagesize = re.compile(r"^(\d+)x(\d+)$")
type_of_gr_update = type(gr.update())


def parse_generation_parameters(x: str):
    """parses generation parameters string, the one you see in text field under the picture in UI:
```
girl with an artist's beret, determined, blue eyes, desert scene, computer monitors, heavy makeup, by Alphonse Mucha and Charlie Bowater, ((eyeshadow)), (coquettish), detailed, intricate
Negative prompt: ugly, fat, obese, chubby, (((deformed))), [blurry], bad anatomy, disfigured, poorly drawn face, mutation, mutated, (extra_limb), (ugly), (poorly drawn hands), messy drawing
Steps: 20, Sampler: Euler a, CFG scale: 7, Seed: 965400086, Size: 512x512, Model hash: 45dee52b
```

    returns a dict with field values
    """

    res = {}

    prompt = ""
    negative_prompt = ""

    done_with_prompt = False

    *lines, lastline = x.strip().split("\n")
    if not re_params.match(lastline):
        lines.append(lastline)
        lastline = ''

    for i, line in enumerate(lines):
        line = line.strip()
        if line.startswith("Negative prompt:"):
            done_with_prompt = True
            line = line[16:].strip()

        if done_with_prompt:
            negative_prompt += ("" if negative_prompt == "" else "\n") + line
        else:
            prompt += ("" if prompt == "" else "\n") + line

    if len(prompt) > 0:
        res["Prompt"] = prompt

    if len(negative_prompt) > 0:
        res["Negative prompt"] = negative_prompt

    for k, v in re_param.findall(lastline):
        m = re_imagesize.match(v)
        if m is not None:
            res[k+"-1"] = m.group(1)
            res[k+"-2"] = m.group(2)
        else:
            res[k] = v

    return res


def connect_paste(button, paste_fields, input_comp, js=None):
    def paste_func(prompt):
        params = parse_generation_parameters(prompt)
        res = []

        for output, key in paste_fields:
            if callable(key):
                v = key(params)
            else:
                v = params.get(key, None)

            if v is None:
                res.append(gr.update())
            elif isinstance(v, type_of_gr_update):
                res.append(v)
            else:
                try:
                    valtype = type(output.value)
                    val = valtype(v)
                    res.append(gr.update(value=val))
                except Exception:
                    res.append(gr.update())

        return res

    button.click(
        fn=paste_func,
        _js=js,
        inputs=[input_comp],
        outputs=[x[0] for x in paste_fields],
    )


# Using constants for these since the variation selector isn't visible.
# Important that they exactly match script.js for tooltip to work.
random_symbol = '\U0001f3b2\ufe0f'  # 🎲️
reuse_symbol  = '\u267b\ufe0f'      # ♻️
art_symbol    = '\U0001f3a8'        # 🎨
paste_symbol  = '\u2199\ufe0f'      # ↙
folder_symbol = '\U0001f4c2'        # 📂


def plaintext_to_html(text):
    text = "<p>" + "<br>\n".join([f"{html.escape(x)}" for x in text.split('\n')]) + "</p>"
    return text


def image_from_url_text(filedata):
    if type(filedata) == list:
        if len(filedata) == 0: return

        filedata = filedata[0]

    if filedata.startswith("data:image/png;base64,"):
        filedata = filedata[len("data:image/png;base64,"):]

    filedata = base64.decodebytes(filedata.encode('utf-8'))
    image = Image.open(io.BytesIO(filedata))
    return image


def save_files(js_data, images, index):
    import csv    
    filenames = []
    fullfns = []

    #quick dictionary to class object conversion. Its necessary due apply_filename_pattern requiring it
    class MyObject:
        def __init__(self, d=None):
            if d is not None:
                for key, value in d.items():
                    setattr(self, key, value)

    data = json.loads(js_data)

    p = MyObject(data)
    path = cmd_opts.outdir_save
    save_to_dirs = cmd_opts.use_save_to_dirs_for_ui
    extension: str = cmd_opts.samples_format
    start_index = 0

    if index > -1 and (index >= data["index_of_first_image"]):  # ensures we are looking at a specific non-grid picture, and we have save_selected_only
        images = [images[index]]
        start_index = index

    with open(os.path.join(cmd_opts.outdir_save, "log.csv"), "a", newline='') as file:
        at_start = file.tell() == 0
        writer = csv.writer(file)
        if at_start:
            writer.writerow(["prompt", "seed", "width", "height", "sampler", "cfgs", "steps", "filename", "negative_prompt"])

        for image_index, filedata in enumerate(images, start_index):
            if filedata.startswith("data:image/png;base64,"):
                filedata = filedata[len("data:image/png;base64,"):]

            image = Image.open(io.BytesIO(base64.decodebytes(filedata.encode('utf-8'))))

            is_grid = image_index < p.index_of_first_image
            i = 0 if is_grid else (image_index - p.index_of_first_image)

            fullfn, txt_fullfn = save_image(image, path, "", seed=p.all_seeds[i], prompt=p.all_prompts[i], extension=extension, info=p.infotexts[image_index], grid=is_grid, p=p, save_to_dirs=save_to_dirs)

            filename = os.path.relpath(fullfn, path)
            filenames.append(filename)
            fullfns.append(fullfn)
            if txt_fullfn:
                filenames.append(os.path.basename(txt_fullfn))
                fullfns.append(txt_fullfn)

        writer.writerow([data["prompt"], data["seed"], data["width"], data["height"], data["sampler"], data["cfg_scale"], data["steps"], filenames[0], data["negative_prompt"]])

    return gr.File.update(value=fullfns, visible=True), '', '', plaintext_to_html(f"Saved: {filenames[0]}")


def wrap_gradio_call(func, extra_outputs=None):
    def f(*args, extra_outputs_array=extra_outputs, **kwargs):
        run_memmon = cmd_opts.memmon_poll_rate > 0 and not cmd_opts.mem_mon.disabled
        if run_memmon:
            cmd_opts.mem_mon.monitor()
        t = time.perf_counter()

        try:
            res = list(func(*args, **kwargs))
        except Exception as e:
            print("Error completing request", file=sys.stderr)
            print("Arguments:", args, kwargs, file=sys.stderr)
            print(traceback.format_exc(), file=sys.stderr)

            state.job = ""
            state.job_count = 0

            if extra_outputs_array is None:
                extra_outputs_array = [None, '']

            res = extra_outputs_array + [f"<div class='error'>{plaintext_to_html(type(e).__name__+': '+str(e))}</div>"]

        elapsed = time.perf_counter() - t
        elapsed_m = int(elapsed // 60)
        elapsed_s = elapsed % 60
        elapsed_text = f"{elapsed_s:.2f}s"
        if (elapsed_m > 0):
            elapsed_text = f"{elapsed_m}m "+elapsed_text

        if run_memmon:
            mem_stats = {k: -(v//-(1024*1024)) for k, v in cmd_opts.mem_mon.stop().items()}
            active_peak = mem_stats['active_peak']
            reserved_peak = mem_stats['reserved_peak']
            sys_peak = mem_stats['system_peak']
            sys_total = mem_stats['total']
            sys_pct = round(sys_peak/max(sys_total, 1) * 100, 2)

            vram_html = f"<p class='vram'>Torch active/reserved: {active_peak}/{reserved_peak} MiB, <wbr>Sys VRAM: {sys_peak}/{sys_total} MiB ({sys_pct}%)</p>"
        else:
            vram_html = ''

        # last item is always HTML
        res[-1] += f"<div class='performance'><p class='time'>Time taken: <wbr>{elapsed_text}</p>{vram_html}</div>"

        state.skipped = False
        state.interrupted = False
        state.job_count = 0

        return tuple(res)

    return f


def wrap_queued_call(func):
    def f(*args, **kwargs):
        with lock:
            r = func(*args, **kwargs)    
        return r
    return f


def wrap_gradio_gpu_call(func, extra_outputs=None):
    def f(*args, **kwargs):
        devices.torch_gc()

        state.job_no = 0
        state.job_count = -1
        state.job_timestamp = state.get_job_timestamp()
        state.sampling_step = 0
        state.current_latent = None
        state.current_image = None
        state.current_image_sampling_step = 0
        state.skipped = False
        state.interrupted = False
        state.textinfo = None

        with lock:
            r = func(*args, **kwargs)

        state.job = ''
        state.job_count = 0

        devices.torch_gc()

        return r

    return ui.wrap_gradio_call(f, extra_outputs=extra_outputs)


opts.onchange("sd_model_checkpoint", wrap_queued_call(lambda: sd_model.reload_model_weights(runtime.sd_model)))
opts.onchange("sd_hypernetwork", wrap_queued_call(lambda: hypernetwork.load_hypernetwork(opts.sd_hypernetwork)))


def _do_check_progress():
    state.job_count = -1
    state.current_latent = None
    state.current_image = None
    state.textinfo = None

    if state.job_count == 0:
        return "", gr_show(False), gr_show(False), gr_show(False)

    progress = 0

    if state.job_count > 0:
        progress += state.job_no / state.job_count
    if state.sampling_steps > 0:
        progress += 1 / state.job_count * state.sampling_step / state.sampling_steps

    progress = min(progress, 1)

    progressbar = ""
    if cmd_opts.show_progressbar:
        progressbar = f"""<div class='progressDiv'><div class='progress' style="width:{progress * 100}%">{str(int(progress*100))+"%" if progress > 0.01 else ""}</div></div>"""

    image = gr_show(False)
    preview_visibility = gr_show(False)

    if cmd_opts.show_progress_every_n_steps > 0:
        if cmd_opts.parallel_processing_allowed:

            if state.sampling_step - state.current_image_sampling_step >= cmd_opts.show_progress_every_n_steps and state.current_latent is not None:
                state.current_image = runtime.sd_samplers['default'].sample_to_image(state.current_latent)
                state.current_image_sampling_step = state.sampling_step

        image = state.current_image

        if image is None:
            image = gr.update(value=None)
        else:
            preview_visibility = gr_show(True)

    if state.textinfo is not None:
        textinfo_result = gr.HTML.update(value=state.textinfo, visible=True)
    else:
        textinfo_result = gr_show(False)

    return f"<span id='progress_span' style='display: none'>{time.time()}</span><p>{progressbar}</p>", preview_visibility, image, textinfo_result


def _do_roll(prompt):
    allowed_cats = set([x for x in cmd_opts.artist_db.categories() if len(cmd_opts.random_artist_categories)==0 or x in cmd_opts.random_artist_categories])
    artist = random.choice([x for x in cmd_opts.artist_db.artists if x.category in allowed_cats])

    return prompt + ", " + artist.name if prompt != '' else artist.name


def _do_save_style(name: str, prompt: str, prompt_neg: str):
    if name is None:
        return [gr_show(), gr_show()]

    style = modules.prompt_helper.styles.PromptStyle(name, prompt, prompt_neg)
    cmd_opts.prompt_styles.styles[style.name] = style
    # Save all loaded prompt styles: this allows us to update the storage format in the future more easily, because we
    # reserialize all styles every time we save them
    cmd_opts.prompt_styles.save_styles(cmd_opts.styles_filename)

    return [gr.Dropdown.update(visible=True, choices=list(cmd_opts.prompt_styles.styles)) for _ in range(2)]


def _do_load_style(prompt, prompt_neg, style1_name, style2_name):
    prompt = cmd_opts.prompt_styles.apply_styles_to_prompt(prompt, [style1_name, style2_name])
    prompt_neg = cmd_opts.prompt_styles.apply_negative_styles_to_prompt(prompt_neg, [style1_name, style2_name])

    return [gr.Textbox.update(value=prompt), gr.Textbox.update(value=prompt_neg), gr.Dropdown.update(value="None"), gr.Dropdown.update(value="None")]


def interrogate(image):
    prompt = cmd_opts.interrogator.interrogate(image)

    return gr_show(True) if prompt is None else prompt


def interrogate_deepbooru(image):
    prompt = get_deepbooru_tags(image)
    return gr_show(True) if prompt is None else prompt


def connect_reuse_seed(seed: gr.Number, reuse_seed: gr.Button, generation_info: gr.Textbox, dummy_component, is_subseed):
    """ Connects a 'reuse (sub)seed' button's click event so that it copies last used
        (sub)seed value from generation info the to the seed field. If copying subseed and subseed strength
        was 0, i.e. no variation seed was used, it copies the normal seed value instead."""
    def copy_seed(gen_info_string: str, index):
        res = -1

        try:
            gen_info = json.loads(gen_info_string)
            index -= gen_info.get('index_of_first_image', 0)

            if is_subseed and gen_info.get('subseed_strength', 0) > 0:
                all_subseeds = gen_info.get('all_subseeds', [-1])
                res = all_subseeds[index if 0 <= index < len(all_subseeds) else 0]
            else:
                all_seeds = gen_info.get('all_seeds', [-1])
                res = all_seeds[index if 0 <= index < len(all_seeds) else 0]

        except json.decoder.JSONDecodeError as e:
            if gen_info_string != '':
                print("Error parsing JSON generation info:", file=sys.stderr)
                print(gen_info_string, file=sys.stderr)

        return [res, gr_show(False)]

    reuse_seed.click(
        fn=copy_seed,
        _js="(x, y) => [x, selected_gallery_index()]",
        show_progress=False,
        inputs=[generation_info, dummy_component],
        outputs=[seed, dummy_component]
    )


def update_token_counter(text, steps):
    try:
        _, prompt_flat_list, _ = prompt_parser.get_multicond_prompt_list([text])
        prompt_schedules = prompt_parser.get_learned_conditioning_prompt_schedules(prompt_flat_list, steps)

    except Exception:
        # a parsing error can happen here during typing, and we don't want to bother the user with
        # messages related to it in console
        prompt_schedules = [[[steps, text]]]

    flat_prompts = reduce(lambda list1, list2: list1+list2, prompt_schedules)
    prompts = [prompt_text for step, prompt_text in flat_prompts]
    tokens, token_count, max_length = max([model_hijack.tokenize(prompt) for prompt in prompts], key=lambda args: args[1])
    style_class = ' class="red"' if (token_count > max_length) else ""
    return f"<span {style_class}>{token_count}/{max_length}</span>"


def create_ui():
    '''
        Prefix for named ui widges:
           tx_    Textbox
           btn_   Button
           s_     Slider
           cb_    Dropdown (combo box)
           c_     Checkbox
           r_     Radio
           lbl_   Label
           p_     HTML
           img_   Image
           pv_    Gallery (picture viewer)
    '''

    with gr.Blocks(analytics_enabled=False) as txt2img_interface:
        with gr.Row(elem_id="toprow"):
            with gr.Column(scale=4):
                with gr.Row():
                    with gr.Column(scale=80):
                        with gr.Row():
                            tx_prompt_pos = gr.Textbox(label="Postive prompt", elem_id="prompt_pos", show_label=False, placeholder="Postive Prompt", lines=2)

                    with gr.Column(scale=1, elem_id="roll_col"):
                        btn_roll = gr.Button(value=art_symbol, elem_id="roll", visible=len(cmd_opts.artist_db.artists) > 0)
                        btn_roll.click(
                            fn=_do_roll,
                            _js="update_txt2img_tokens",
                            inputs=[
                                tx_prompt_pos,
                            ],
                            outputs=[
                                tx_prompt_pos,
                            ]
                        )

                        btn_paste = gr.Button(value=paste_symbol, elem_id="paste")
                        p_token_counter = gr.HTML(value="<span></span>", elem_id="token_counter")

                    with gr.Column(scale=10, elem_id="style_pos_col"):
                        cb_style_pos = gr.Dropdown(label="Style Pos", elem_id="style_pos_index", 
                                                   choices=[k for k, v in cmd_opts.prompt_styles.styles.items()], 
                                                   value=next(iter(cmd_opts.prompt_styles.styles.keys())))

                with gr.Row():
                    with gr.Column(scale=80):
                        tx_prompt_neg = gr.Textbox(label="Negative prompt", elem_id="prompt_neg", show_label=False, placeholder="Negative prompt", lines=2)

                    with gr.Column(scale=1, elem_id="style_neg_col"):
                        cb_style_neg = gr.Dropdown(label="Style Neg", elem_id="style_neg_index", 
                                                   choices=[k for k, v in cmd_opts.prompt_styles.styles.items()], 
                                                   value=next(iter(cmd_opts.prompt_styles.styles.keys())))

            with gr.Column(scale=1):
                with gr.Row():
                    skip = gr.Button('Skip', elem_id="skip")
                    interrupt = gr.Button('Interrupt', elem_id="interrupt")
                    submit = gr.Button('Generate', elem_id="generate", variant='primary')

                    skip.click(
                        fn=state.skip,
                        inputs=[],
                        outputs=[],
                    )

                    interrupt.click(
                        fn=state.interrupt,
                        inputs=[],
                        outputs=[],
                    )

                with gr.Row(scale=1):
                    interrogate = gr.Button('Interrogate\nCLIP', elem_id="interrogate")
                    deepbooru = gr.Button('Interrogate\nDeepBooru', elem_id="deepbooru")
                    btn_style_load = gr.Button('Load style', elem_id="style_load")
                    btn_style_save = gr.Button('Save style', elem_id="style_save")

        lbl_empty = gr.Label(visible=False)   # null object for placeholder 

        with gr.Row(elem_id='progress_row'):
            with gr.Column(scale=1):
                p_progressbar = gr.HTML(elem_id="progressbar")
                img_preview = gr.Image(elem_id='preview', visible=False)
                p_textinfo = gr.HTML(visible=False)

                btn_check_progress = gr.Button('Check progress', elem_id="check_progress", visible=True)
                btn_check_progress.click(
                    fn=_do_check_progress,
                    show_progress=False,
                    inputs=[],
                    outputs=[p_progressbar, img_preview, img_preview, p_textinfo],
                )

        with gr.Row(elem_id="controls").style(equal_height=False):
            with gr.Column(variant='panel'):
                with gr.Group():
                    c_mode_img2img = gr.Checkbox(label='Img2Img Mode', value=False)
                    c_mode_img2img.change(
                        fn=lambda x: gr_show(x),
                        inputs=[c_mode_img2img],
                        outputs=[gp_img2img],
                    )

                    r_mode_control = gr.Radio(label="Resize mode", choices=["Normal", "Control Variable"], type="index", value="Normal")
                    r_mode_control.change(
                        fn=lambda x: gr_show(x),
                        inputs=[c_mode_img2img1],
                        outputs=[gp_img2img],
                    )

                with gr.Group() as gp_img2img:
                    with gr.Tabs(elem_id="mode_img2img", visible=False):
                        with gr.TabItem('Img2Img', id='img2img'):
                            init_img = gr.Image(label="Image for img2img", elem_id="img2img_image", show_label=False, source="upload", interactive=True, type="pil", tool=cmd_opts.gradio_img2img_tool)

                        with gr.TabItem('Inpaint', id='inpaint'):
                            init_img_with_mask = gr.Image(label="Image for inpainting with mask",  show_label=False, elem_id="img2maskimg", source="upload", interactive=True, type="pil", tool="sketch", image_mode="RGBA")
                            init_img_inpaint = gr.Image(label="Image for img2img", show_label=False, source="upload", interactive=True, type="pil", visible=False, elem_id="img_inpaint_base")
                            init_mask_inpaint = gr.Image(label="Mask", source="upload", interactive=True, type="pil", visible=False, elem_id="img_inpaint_mask")

                            mask_blur = gr.Slider(label='Mask blur', minimum=0, maximum=64, step=1, value=4)

                            with gr.Row():
                                mask_mode = gr.Radio(label="Mask mode", show_label=False, choices=["Draw mask", "Upload mask"], type="index", value="Draw mask", elem_id="mask_mode")
                                inpainting_mask_invert = gr.Radio(label='Masking mode', show_label=False, choices=['Inpaint masked', 'Inpaint not masked'], value='Inpaint masked', type="index")

                            inpainting_fill = gr.Radio(label='Masked content', choices=['fill', 'original', 'latent noise', 'latent nothing'], value='original', type="index")

                            with gr.Row():
                                inpaint_full_res = gr.Checkbox(label='Inpaint at full resolution', value=False)
                                inpaint_full_res_padding = gr.Slider(label='Inpaint at full resolution padding, pixels', minimum=0, maximum=256, step=4, value=32)

                    with gr.Row():
                        resize_mode = gr.Radio(label="Resize mode", elem_id="resize_mode", show_label=False, choices=["Just resize", "Crop and resize", "Resize and fill"], type="index", value="Just resize")


                with gr.Column(variant='panel') as p_control_normal:
                    pass


                s_steps = gr.Slider(minimum=1, maximum=75, step=1, label="Sampling Steps", value=20)
                r_sampler = gr.Radio(label='Sampling method', elem_id="sampler", choices=[x.name for x in samplers], value=samplers[0].name, type="index")

                with gr.Group():
                    s_width = gr.Slider(minimum=384, maximum=1600, step=64, label="Width", value=512)
                    s_height = gr.Slider(minimum=384, maximum=1600, step=64, label="Height", value=512)

                s_count = gr.Slider(minimum=1, maximum=16, step=1, label='Number of images to generate', value=4)
                s_cfg_scale = gr.Slider(minimum=1.0, maximum=15.0, step=0.5, label='CFG Scale', value=7.0)

                with gr.Group():
                    with gr.Row():
                        with gr.Box():
                            with gr.Row(elem_id='seed_row'):
                                num_seed = gr.Number(label='Seed', value=-1).style(container=False)
                                btn_seed_random = gr.Button(random_symbol, elem_id='random_seed')
                                btn_seed_reuse = gr.Button(reuse_symbol, elem_id='reuse_seed')

                        with gr.Box(elem_id='subseed_show_box'):
                            c_seed_extra = gr.Checkbox(label='Extra', elem_id='subseed_show', value=False)

                    # Components to show/hide based on the 'Extra' checkbox
                    seed_extras = []

                    with gr.Row(visible=False) as seed_extra_row_1:
                        seed_extras.append(seed_extra_row_1)
                        with gr.Box():
                            with gr.Row(elem_id='subseed_row'):
                                subseed = gr.Number(label='Variation seed', value=-1).style(container=False)
                                btn_random_subseed = gr.Button(random_symbol, elem_id='random_subseed')
                                btn_reuse_subseed = gr.Button(reuse_symbol, elem_id='reuse_subseed')
                        subseed_strength = gr.Slider(label='Variation strength', value=0.0, minimum=0, maximum=1, step=0.01)

                    with gr.Row(visible=False) as seed_extra_row_2:
                        seed_extras.append(seed_extra_row_2)
                        seed_resize_from_w = gr.Slider(minimum=0, maximum=1600, step=64, label="Resize seed from width", value=0)
                        seed_resize_from_h = gr.Slider(minimum=0, maximum=1600, step=64, label="Resize seed from height", value=0)

                    btn_seed_random.click(fn=lambda: -1, show_progress=False, inputs=[], outputs=[num_seed])
                    btn_random_subseed.click(fn=lambda: -1, show_progress=False, inputs=[], outputs=[subseed])

                    c_seed_extra.change(
                        fn=lambda show: {comp: gr_show(show) for comp in seed_extras}, 
                        show_progress=False, 
                        inputs=[c_seed_extra], 
                        outputs=seed_extras
                    )

                with gr.Group():
                    custom_inputs = modules.scripts.scripts_txt2img.setup_ui(is_img2img=False)

                # 我猜测这些因该被移动到fx
                with gr.Group():
                    with gr.Row():
                        c_tiling = gr.Checkbox(label='Tiling', value=False)
                        c_restore_faces = gr.Checkbox(label='Restore faces', value=False)
                        c_highres_fix = gr.Checkbox(label='Highres. fix', value=False)
                        c_highres_fix.change(
                            fn=lambda x: gr_show(x),
                            inputs=[c_highres_fix],
                            outputs=[hr_options],
                        )
                    with gr.Row(visible=False) as hr_options:
                        c_scale_latent = gr.Checkbox(label='Scale latent', value=False)
                        s_denoising_strength = gr.Slider(minimum=0.0, maximum=1.0, step=0.01, label='Denoising strength', value=0.7)

            with gr.Column(variant='panel'):
                with gr.Group():
                    img_preview = gr.Image(elem_id='preview', visible=False)
                    pv_gallery = gr.Gallery(label='Output', show_label=False, elem_id='gallery').style(grid=4)

                with gr.Group():
                    with gr.Row():
                        btn_save = gr.Button('Save')
                        btn_save.click(
                            fn=wrap_gradio_call(save_files),
                            _js="(x, y, z, w) => [x, y, z, selected_gallery_index()]",
                            inputs=[
                                tx_generation_info,
                                pv_gallery,
                                html_info,
                            ],
                            outputs=[
                                download_files,
                                html_info,
                                html_info,
                                html_info,
                            ]
                        )

                        btn_send_to_img2img = gr.Button('Send to img2img')
                        btn_send_to_inpaint = gr.Button('Send to inpaint')
                        btn_send_to_skecth = gr.Button('Send to sketch')
                    
                    with gr.Row():
                        download_files = gr.File(None, file_count="multiple", interactive=False, show_label=False, visible=False)

                with gr.Group():
                    html_info = gr.HTML()
                    tx_generation_info = gr.Textbox(visible=False)

            connect_reuse_seed(num_seed, btn_seed_reuse, tx_generation_info, lbl_empty, is_subseed=False)
            connect_reuse_seed(subseed, btn_reuse_subseed, tx_generation_info, lbl_empty, is_subseed=True)

        txt2img_args = dict(
            fn=wrap_gradio_gpu_call(apps.txt2img),
            _js="submit",
            inputs=[
                tx_prompt_pos,
                tx_prompt_neg,
                cb_style_pos,
                cb_style_neg,
                s_steps,
                r_sampler,
                c_restore_faces,
                c_tiling,
                s_count,
                s_cfg_scale,
                num_seed,
                subseed, subseed_strength, seed_resize_from_h, seed_resize_from_w, c_seed_extra,
                s_height,
                s_width,
                c_mode_img2img,
                c_scale_latent,
                s_denoising_strength,
            ] + custom_inputs,
            outputs=[
                pv_gallery,
                tx_generation_info,
                html_info
            ],
            show_progress=False,
        )

        img2img_args = dict(
            fn=wrap_gradio_gpu_call(apps.img2img),
            _js="submit_img2img",
            inputs=[
                lbl_empty,
                tx_prompt_pos,
                tx_prompt_neg,
                cb_style_pos,
                cb_style_neg,
                init_img,
                init_img_with_mask,
                init_img_inpaint,
                init_mask_inpaint,
                mask_mode,
                s_steps,
                r_sampler,
                mask_blur,
                inpainting_fill,
                c_restore_faces,
                c_tiling,
                batch_count,
                s_count,
                s_cfg_scale,
                s_denoising_strength,
                num_seed,
                subseed, subseed_strength, seed_resize_from_h, seed_resize_from_w, c_seed_extra,
                s_height,
                s_width,
                resize_mode,
                inpaint_full_res,
                inpaint_full_res_padding,
                inpainting_mask_invert,
            ] + custom_inputs,
            outputs=[
                pv_gallery,
                tx_generation_info,
                html_info
            ],
            show_progress=False,
        )

        tx_prompt_pos.submit(**txt2img_args)
        submit.click(**txt2img_args)

        paste_fields = [
            (tx_prompt_pos, "Positive Prompt"),
            (tx_prompt_neg, "Negative prompt"),
            (s_count, "Count"),
            (s_steps, "Steps"),
            (r_sampler, "Sampler"),
            (s_cfg_scale, "CFG scale"),
            (num_seed, "Seed"),
            (s_width, "Size-1"),
            (s_height, "Size-2"),
            (subseed, "Variation seed"),
            (subseed_strength, "Variation seed strength"),
            (seed_resize_from_w, "Seed resize from-1"),
            (seed_resize_from_h, "Seed resize from-2"),
            (s_denoising_strength, "Denoising strength"),
            (c_mode_img2img, lambda d: "Denoising strength" in d),
            (c_restore_faces, "Face restoration"),
            (hr_options, lambda d: gr.Row.update(visible="Denoising strength" in d)),
        ]
        
        connect_paste(btn_paste, paste_fields, tx_prompt_pos)

        mask_mode.change(
            lambda mode, img: {
                init_img_with_mask: gr_show(mode == 0),
                init_img_inpaint: gr_show(mode == 1),
                init_mask_inpaint: gr_show(mode == 1),
            },
            inputs=[mask_mode, init_img_with_mask],
            outputs=[
                init_img_with_mask,
                init_img_inpaint,
                init_mask_inpaint,
            ],
        )

        interrogate.click(
            fn=interrogate,
            inputs=[init_img],
            outputs=[img2img_prompt],
        )

        if cmd_opts.deepdanbooru:
            deepbooru.click(
                fn=interrogate_deepbooru,
                inputs=[init_img],
                outputs=[img2img_prompt],
            )

        style_dropdowns = [(txt2img_prompt_style, txt2img_prompt_style2), (cb_style_pos, cb_style_neg)]

        btn_style_load.click(
            fn=_do_load_style,
            _js='update_txt2img_tokens',
            inputs=[tx_prompt_pos, tx_prompt_neg, style_dropdowns],
            outputs=[tx_prompt_pos, tx_prompt_neg, style_dropdowns],
        )

        btn_style_save.click(
            fn=_do_save_style,
            _js="ask_for_style_name",
            # Have to pass empty dummy component here, because the JavaScript and Python function have to accept
            # the same number of parameters, but we only know the style-name after the JavaScript prompt
            inputs=[lbl_empty, tx_prompt_pos, tx_prompt_neg],
            outputs=[cb_style_pos, cb_style_neg],
        )


    with gr.Blocks(analytics_enabled=False) as postprocess_interface:
        with gr.Row().style(equal_height=False):
            with gr.Column(variant='panel'):
                with gr.Tabs(elem_id="mode_extras"):
                    with gr.TabItem('Single Image'):
                        extras_image = gr.Image(label="Source", source="upload", interactive=True, type="pil")

                    with gr.TabItem('Batch Process'):
                        image_batch = gr.File(label="Batch Process", file_count="multiple", interactive=True, type="file")

                upscaling_resize = gr.Slider(minimum=1.0, maximum=4.0, step=0.05, label="Resize", value=2)

                with gr.Group():
                    extras_upscaler_1 = gr.Radio(label='Upscaler 1', choices=[x.name for x in cmd_opts.sd_upscalers], value=cmd_opts.sd_upscalers[0].name, type="index")

                with gr.Group():
                    extras_upscaler_2 = gr.Radio(label='Upscaler 2', choices=[x.name for x in cmd_opts.sd_upscalers], value=cmd_opts.sd_upscalers[0].name, type="index")
                    extras_upscaler_2_visibility = gr.Slider(minimum=0.0, maximum=1.0, step=0.001, label="Upscaler 2 visibility", value=1)

                with gr.Group():
                    gfpgan_visibility = gr.Slider(minimum=0.0, maximum=1.0, step=0.001, label="GFPGAN visibility", value=0, interactive=modules.gfpgan_model.have_gfpgan)

                with gr.Group():
                    codeformer_visibility = gr.Slider(minimum=0.0, maximum=1.0, step=0.001, label="CodeFormer visibility", value=0, interactive=modules.codeformer_model.have_codeformer)
                    codeformer_weight = gr.Slider(minimum=0.0, maximum=1.0, step=0.001, label="CodeFormer weight (0 = maximum effect, 1 = minimum effect)", value=0, interactive=modules.codeformer_model.have_codeformer)

                submit = gr.Button('Generate', elem_id="extras_generate", variant='primary')

            with gr.Column(variant='panel'):
                result_images = gr.Gallery(label="Result", show_label=False)
                html_info_x = gr.HTML()
                html_info = gr.HTML()
                btn_fx_send_to_img2img = gr.Button('Send to img2img')
                btx_fx_send_to_inpaint = gr.Button('Send to inpaint')

                submit.click(
                    fn=wrap_gradio_gpu_call(apps.run_extras),
                    _js="get_extras_tab_index",
                    inputs=[
                        lbl_empty,
                        extras_image,
                        image_batch,
                        gfpgan_visibility,
                        codeformer_visibility,
                        codeformer_weight,
                        upscaling_resize,
                        extras_upscaler_1,
                        extras_upscaler_2,
                        extras_upscaler_2_visibility,
                    ],
                    outputs=[
                        result_images,
                        html_info_x,
                        html_info,
                    ]
                )
            
                btn_fx_send_to_img2img.click(
                    fn=lambda x: image_from_url_text(x),
                    _js="extract_image_from_gallery_img2img",
                    inputs=[result_images],
                    outputs=[init_img],
                )
                
                btx_fx_send_to_inpaint.click(
                    fn=lambda x: image_from_url_text(x),
                    _js="extract_image_from_gallery_img2img",
                    inputs=[result_images],
                    outputs=[init_img_with_mask],
                )


    with gr.Blocks(analytics_enabled=False) as sketch_interface:
        with gr.Row().style(equal_height=False):
            with gr.Column(variant='panel'):
                img_upload = gr.Image(elem_id="pnginfo_image", label="Source", source="upload", interactive=True, type="pil")
                img_upload.change(
                    fn=wrap_gradio_call(modules.processing.run_pnginfo),
                    inputs=[img_upload],
                    outputs=[tx_generation_info, tx_info],
                )
            
            with gr.Column(variant='panel'):
                tx_generation_info = gr.Textbox(visible=False)
                tx_info = gr.HTML()

                with gr.Row():
                    btn_sketch_send_to_txt2img = gr.Button('Send to txt2img')
                    btn_sketch_send_to_img2img = gr.Button('Send to img2img')


    def create_setting_component(key):
        def fun():
            return cmd_opts.data[key] if key in cmd_opts.data else cmd_opts.data_labels[key].default

        info = cmd_opts.data_labels[key]
        t = type(info.default)

        args = info.component_args() if callable(info.component_args) else info.component_args

        if info.component is not None:
            comp = info.component
        elif t == str:
            comp = gr.Textbox
        elif t == int:
            comp = gr.Number
        elif t == bool:
            comp = gr.Checkbox
        else:
            raise Exception(f'bad options item type: {str(t)} for key {key}')

        return comp(label=info.label, value=fun, **(args or {}))

    components = []
    component_dict = {}

    def run_settings(*args):
        changed = 0

        for key, value, comp in zip(cmd_opts.data_labels.keys(), args, components):
            if comp != lbl_empty and not cmd_opts.same_type(value, cmd_opts.data_labels[key].default):
                return f"Bad value for setting {key}: {value}; expecting {type(cmd_opts.data_labels[key].default).__name__}", cmd_opts.dumpjson()

        for key, value, comp in zip(cmd_opts.data_labels.keys(), args, components):
            if comp == lbl_empty:
                continue

            comp_args = cmd_opts.data_labels[key].component_args
            if comp_args and isinstance(comp_args, dict) and comp_args.get('visible') is False:
                continue

            oldval = cmd_opts.data.get(key, None)
            cmd_opts.data[key] = value

            if oldval != value:
                if cmd_opts.data_labels[key].onchange is not None:
                    cmd_opts.data_labels[key].onchange()

                changed += 1

        cmd_opts.save(cmd_opts.config_fp)

        return f'{changed} settings changed.', cmd_opts.dumpjson()

    def run_settings_single(value, key):
        if not cmd_opts.same_type(value, cmd_opts.data_labels[key].default):
            return gr.update(visible=True), cmd_opts.dumpjson()

        oldval = cmd_opts.data.get(key, None)
        cmd_opts.data[key] = value

        if oldval != value:
            if cmd_opts.data_labels[key].onchange is not None:
                cmd_opts.data_labels[key].onchange()

        cmd_opts.save(cmd_opts.config_fp)

        return gr.update(value=value), cmd_opts.dumpjson()

    with gr.Blocks(analytics_enabled=False) as settings_interface:
        btn_settings_apply = gr.Button(value="Apply settings", variant='primary')
        p_result = gr.HTML()

        settings_cols = 3
        items_per_col = int(len(opts.options) * 0.9 / settings_cols)

        cols_displayed = 0
        items_displayed = 0
        previous_section = None
        column = None
        with gr.Row(elem_id="settings").style(equal_height=False):
            for i, (k, item) in enumerate(opts.options.items()):

                if previous_section != item.section:
                    if cols_displayed < settings_cols and (items_displayed >= items_per_col or previous_section is None):
                        if column is not None:
                            column.__exit__()

                        column = gr.Column(variant='panel')
                        column.__enter__()

                        items_displayed = 0
                        cols_displayed += 1

                    previous_section = item.section

                    gr.HTML(elem_id="settings_header_text_{}".format(item.section[0]), value='<h1 class="gr-button-lg">{}</h1>'.format(item.section[1]))

                component = create_setting_component(k)
                component_dict[k] = component
                components.append(component)
                items_displayed += 1

        with gr.Row():
            btn_reload_script_bodies = gr.Button(value='Reload custom script bodies (No ui updates, No restart)', variant='secondary')
            btn_reload_script_bodies.click(
                fn=modules.scripts.reload_script_body_only,
                inputs=[],
                outputs=[],
                _js='function(){}'
            )

            def request_restart():
                state.interrupt()
                settings_interface.gradio_ref.do_restart = True

            btn_restart_gradio = gr.Button(value='Restart Gradio and Refresh components (Custom Scripts, ui.py, js and css only)', variant='primary')
            btn_restart_gradio.click(
                fn=request_restart,
                inputs=[],
                outputs=[],
                _js='function(){restart_reload()}'
            )
        
        if column is not None:
            column.__exit__()

    interfaces = [
        # ('interface', 'label', 'ifid')
        (txt2img_interface,  'Txt2Img',  'txt2img'),
        (postprocess_interface, 'Postprocess', 'postprocess'),
        (sketch_interface,  'Sketch', 'sketch'),
        (settings_interface, 'Settings', 'settings'),
    ]

    with gr.Blocks(css=get_css(), analytics_enabled=False, title='Stable Diffusion') as demo:
        # for set 'do_restart'
        settings_interface.gradio_ref = demo
        
        with gr.Tabs():
            for interface, label, ifid in interfaces:
                with gr.TabItem(label, id=ifid):
                    interface.render()
        
        notify_audio_fp = os.path.join(STATIC_PATH, 'notification.mp3')
        if os.path.exists(notify_audio_fp):
            gr.Audio(interactive=False, value=notify_audio_fp, elem_id="audio_notification", visible=False)

        tx_settings = gr.Textbox(elem_id="settings_json", value=lambda: cmd_opts.dumpjson(), visible=False)
        btn_settings_apply.click(
            fn=run_settings,
            inputs=components,
            outputs=[p_result, tx_settings],
        )

        paste_field_names = ['Positive Prompt', 'Negative prompt', 'Steps', 'Face restoration', 'Seed', 'Size-1', 'Size-2']
        txt2img_fields = [field for field, name in paste_fields if name in paste_field_names]
        img2img_fields = [field for field, name in paste_fields if name in paste_field_names]

        btn_send_to_img2img.click(
            fn=lambda img, *args: (image_from_url_text(img),*args),
            _js="(gallery, ...args) => [extract_image_from_gallery_img2img(gallery), ...args]",
            inputs=[pv_gallery] + txt2img_fields,
            outputs=[init_img] + img2img_fields,
        )

        btn_send_to_inpaint.click(
            fn=lambda x, *args: (image_from_url_text(x), *args),
            _js="(gallery, ...args) => [extract_image_from_gallery_inpaint(gallery), ...args]",
            inputs=[pv_gallery] + txt2img_fields,
            outputs=[init_img_with_mask] + img2img_fields,
        )

        btn_send_to_skecth.click(
            fn=lambda x: image_from_url_text(x),
            _js="extract_image_from_gallery_extras",
            inputs=[pv_gallery],
            outputs=[extras_image],
        )

        connect_paste(btn_sketch_send_to_txt2img, paste_fields, tx_generation_info, 'switch_to_txt2img')
        connect_paste(btn_sketch_send_to_img2img, paste_fields, tx_generation_info, 'switch_to_img2img_img2img')

    def loadsave(path, x):
        nonlocal ui_settings

        def apply_field(obj, field, condition=None):
            key = path + "/" + field

            if getattr(obj,'custom_script_source'):
              key = 'customscript/' + obj.custom_script_source + '/' + key
            
            if getattr(obj, 'do_not_save_to_config'):return
            
            saved_value = ui_settings.get(key, None)
            if saved_value is None:
                ui_settings[key] = getattr(obj, field)
            elif condition is None or condition(saved_value):
                setattr(obj, field, saved_value)

        if type(x) in [gr.Slider, gr.Radio, gr.Checkbox, gr.Textbox, gr.Number] and x.visible:
            apply_field(x, 'visible')
        
        if type(x) in [gr.Checkbox, gr.Textbox, gr.Number]:
            apply_field(x, 'value')
        elif type(x) == gr.Slider:
            apply_field(x, 'value')
            apply_field(x, 'minimum')
            apply_field(x, 'maximum')
            apply_field(x, 'step')
        elif type(x) == gr.Radio:
            apply_field(x, 'value', lambda val: val in x.choices)

    def visit(x, func, path=''):
        if hasattr(x, 'children'):
            for c in x.children:
                visit(c, func, path)
        elif x.label is not None:
            func(path + "/" + str(x.label), x)

    visit(txt2img_interface, loadsave, "txt2img")
    visit(postprocess_interface, loadsave, "postprocess")

    ui_settings = {}

    try:
        ui_config_file = cmd_opts.ui_config_file
        if os.path.exists(ui_config_file):
            with open(ui_config_file) as fh:
                ui_settings = json.load(fh)
        else:
            with open(ui_config_file, 'w') as fh:
                json.dump(ui_settings, fh, indent=4)
    except Exception:
        print("Error loading settings:", file=sys.stderr)
        print(traceback.format_exc(), file=sys.stderr)

    return demo


# this is a fix for Windows users. Without it, javascript files will be served with content-type 'text/html' and the browser will not show any UI
mimetypes.init()
mimetypes.add_type('application/javascript', '.js')

# this is also a fix?
if 'gradio_routes_templates_response' not in globals():
    def template_response(*args, **kwargs):
        res = gradio_routes_templates_response(*args, **kwargs)
        res.body = res.body.replace(b'</head>', f'{get_javascript()}</head>'.encode('utf8'))
        res.init_headers()
        return res

    gradio_routes_templates_response = gradio.routes.templates.TemplateResponse
    gradio.routes.templates.TemplateResponse = template_response
