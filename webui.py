import sys
import threading
import time
import importlib
import threading

from fastapi.middleware.gzip import GZipMiddleware

import modules.ui
import modules.extras
import modules.face_restoration
import modules.codeformer_model as codeformer
import modules.gfpgan_model as gfpgan
import modules.txt2img
import modules.img2img
import modules.lowvram
import modules.scripts
import modules.sd_hijack
import modules.sd_models
import modules.shared as shared
from modules import modelloader
from modules import devices, sd_samplers
from modules.shared import cmd_opts
from modules.paths import SCRIPT_PATH

modelloader.cleanup_models()
modules.sd_models.setup_model()
codeformer.setup_model(cmd_opts.codeformer_models_path)
gfpgan.setup_model(cmd_opts.gfpgan_models_path)
shared.face_restorers.append(modules.face_restoration.FaceRestoration())
modelloader.load_upscalers()

lock = threading.Lock()


def wrap_queued_call(func):
  def f(*args, **kwargs):
    with lock:
      res = func(*args, **kwargs)
    return res
  return f


def wrap_gradio_gpu_call(func, extra_outputs=None):
  def f(*args, **kwargs):
    devices.torch_gc()

    shared.state.sampling_step = 0
    shared.state.job_count = -1
    shared.state.job_no = 0
    shared.state.job_timestamp = shared.state.get_job_timestamp()
    shared.state.current_latent = None
    shared.state.current_image = None
    shared.state.current_image_sampling_step = 0
    shared.state.skipped = False
    shared.state.interrupted = False
    shared.state.textinfo = None

    with lock:
      res = func(*args, **kwargs)

    shared.state.job = ""
    shared.state.job_count = 0

    devices.torch_gc()

    return res

  return modules.ui.wrap_gradio_call(f, extra_outputs=extra_outputs)

modules.scripts.load_scripts()

shared.sd_model = modules.sd_models.load_model()
shared.opts.onchange("sd_model_checkpoint", wrap_queued_call(lambda: modules.sd_models.reload_model_weights(shared.sd_model)))

loaded_hypernetwork = modules.hypernetwork.load_hypernetwork(shared.opts.sd_hypernetwork)
shared.opts.onchange("sd_hypernetwork", wrap_queued_call(lambda: modules.hypernetwork.load_hypernetwork(shared.opts.sd_hypernetwork)))


if __name__ == '__main__':
	print(f"Launching Web UI with arguments: {' '.join(sys.argv[1:])}")

  try:
		while True:
			demo = modules.ui.create_ui(wrap_gradio_gpu_call=wrap_gradio_gpu_call)
			
			app,local_url,share_url = demo.launch(
				share=True,
				server_name='0.0.0.0',
				server_port=cmd_opts.port,
				debug=cmd_opts.gradio_debug,
				inbrowser=cmd_opts.autolaunch,
				prevent_thread_lock=True
			)
			app.add_middleware(GZipMiddleware, minimum_size=1000)

			while 1:
				time.sleep(0.5)
				if getattr(demo, 'do_restart', False):
					time.sleep(0.5)
					demo.close()
					time.sleep(0.5)
					break

			sd_samplers.set_samplers()

			print('Reloading Custom Scripts')
			modules.scripts.reload_scripts(SCRIPT_PATH)
			print('Reloading modules: modules.ui')
			importlib.reload(modules.ui)
			print('Restarting Gradio')
  except KeyboardInterrupt:
    print('Exit by Ctrl+C')
  except Exception:
    print_exc()
  finally:
    ngrok.stop()
    webui.stop()



