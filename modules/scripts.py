import os
import sys
import traceback
from types import ModuleType

import gradio as gr

from modules import ui
from modules.paths import SCRIPT_PATH
from modules import runtime
from modules.runtime import state
from modules.processing import StableDiffusionProcessing


scripts_data = []


class Script:

    filename = None
    args_from = None
    args_to = None

    # The title of the script. This is what will be displayed in the dropdown menu.
    def title(self):
        raise NotImplementedError()

    # How the script is displayed in the UI. See https://gradio.app/docs/#components
    # for the different UI components you can use and how to create them.
    # Most UI components can return a value, such as a boolean for a checkbox.
    # The returned values are passed to the run method as parameters.
    def ui(self, is_img2img):
        pass

    # Determines when the script should be shown in the dropdown menu via the 
    # returned value. As an example:
    # is_img2img is True if the current tab is img2img, and False if it is txt2img.
    # Thus, return is_img2img to only show the script on the img2img tab.
    def show(self, is_img2img):
        return True

    # This is where the additional processing is implemented. The parameters include
    # self, the model object "p" (a StableDiffusionProcessing class, see
    # processing.py), and the parameters returned by the ui method.
    # Custom functions can be defined here, and additional libraries can be imported 
    # to be used in processing. The return value should be a Processed object, which is
    # what is returned by the process_images method.
    def run(self, *args):
        raise NotImplementedError()

    # The description method is currently unused.
    # To add a description that appears when hovering over the title, amend the "titles" 
    # dict in script.js to include the script title (returned by title) as a key, and 
    # your description as the value.
    def describe(self):
        return ""


def wrap_call(func, filename, funcname, *args, default=None, **kwargs):
    try:
        return func(*args, **kwargs)
    except Exception:
        print(f"Error calling: {filename}/{funcname}", file=sys.stderr)
        print(traceback.format_exc(), file=sys.stderr)


class ScriptRunner:

    def __init__(self):
        self.scripts = []   # [Script]

    def setup_ui(self, is_img2img):
        for script_class, path in scripts_data:
            script = script_class()
            script.filename = path

            if not script.show(is_img2img):
                continue

            self.scripts.append(script)

        titles = [wrap_call(script.title, script.filename, "title") or f"{script.filename} [error]" for script in self.scripts]

        dropdown = gr.Dropdown(label="Script", choices=["None"] + titles, value="None", type="index")
        inputs = [dropdown]

        for script in self.scripts:
            script.args_from = len(inputs)
            script.args_to = len(inputs)

            controls = wrap_call(script.ui, script.filename, "ui", is_img2img)

            if controls is None:
                continue

            for control in controls:
                control.custom_script_source = os.path.basename(script.filename)
                control.visible = False

            inputs += controls
            script.args_to = len(inputs)

        def select_script(script_index):
            if 0 < script_index <= len(self.scripts):
                script = self.scripts[script_index-1]
                args_from = script.args_from
                args_to = script.args_to
            else:
                args_from = 0
                args_to = 0

            return [ui.gr_show(True if i == 0 else args_from <= i < args_to) for i in range(len(inputs))]

        dropdown.change(
            fn=select_script,
            inputs=[dropdown],
            outputs=inputs
        )

        return inputs

    def run(self, p: StableDiffusionProcessing, *args):
        script_index = args[0]

        if script_index == 0:
            return None

        script = self.scripts[script_index-1]

        if script is None:
            return None

        script_args = args[script.args_from:script.args_to]
        processed = script.run(p, *script_args)

        state.total_tqdm.clear()

        return processed

    def reload_sources(self):
        for i, script in list(enumerate(self.scripts)):
            script_fn = script.filename
            with open(script_fn, "r", encoding="utf8") as fh:
                text = fh.read()

                from types import ModuleType

                compiled = compile(text, script_fn, 'exec')
                module = ModuleType(script_fn)
                exec(compiled, module.__dict__)

                for key, script_class in module.__dict__.items():
                    if type(script_class) == type and issubclass(script_class, Script):
                        self.scripts[i] = script_class()
                        self.scripts[i].filename = script_fn
                        self.scripts[i].args_from = script.args_from
                        self.scripts[i].args_to = script.args_to


def load_scripts():
    if not os.path.exists(SCRIPT_PATH): return

    for fn in sorted(os.listdir(SCRIPT_PATH)):
        fp = os.path.join(SCRIPT_PATH, fn)
        if not os.path.isfile(fp): continue

        try:
            with open(fp, "r", encoding="utf8") as file:
                text = file.read()

            compiled = compile(text, fp, 'exec')
            module = ModuleType(fn)
            exec(compiled, module.__dict__)

            for key, script_class in module.__dict__.items():
                if type(script_class) == type and issubclass(script_class, Script):
                    scripts_data.append((script_class, fp))
        except Exception:
            print(f"Error loading script: {fn}", file=sys.stderr)
            print(traceback.format_exc(), file=sys.stderr)


def reload_script_body_only():
    runtime.script_runner.reload_sources()


def reload_scripts():
    scripts_data.clear()
    load_scripts()

    runtime.script_runner = ScriptRunner()
