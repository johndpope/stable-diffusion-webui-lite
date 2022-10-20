import sys
import importlib
from time import sleep
from traceback import print_exc

from fastapi.middleware.gzip import GZipMiddleware

from modules import ui
from modules import runtime
from modules.cmd_opts import cmd_opts


if __name__ == '__main__':
    raw_cmd_args = ' '.join(sys.argv[1:])
    print(f'Launching webui app with cmd_args: {raw_cmd_args}')
    
    try:
        print('Starting Gradio')
        runtime.init()
        while True:
            server = ui.create_ui()
            app, local_url, share_url = server.launch(
                share=True,
                server_name='0.0.0.0',
                server_port=cmd_opts.port,
                debug=cmd_opts.debug,
                prevent_thread_lock=True,
            )
            app.add_middleware(GZipMiddleware, minimum_size=1000)
            print(f'local_url: {local_url}')
            print(f'share_url: {share_url}')
            
            while True:
                sleep(1)
                if hasattr(server, 'do_restart'):
                    server.close()
                    sleep(1)
                    break
            
            print('Reloading modules.runtime models & states')
            runtime.reload()
            print('Reloading modules.ui')
            importlib.reload(ui)
            print('Restarting Gradio')
    except KeyboardInterrupt:
        print('Exit by Ctrl+C')
    except Exception:
        print_exc()
