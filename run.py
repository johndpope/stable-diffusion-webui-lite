#!/usr/bin/env python3
# Author: Armit
# Create Time: 2022/10/11 

import os
import subprocess
from abc import ABCMeta, abstractmethod
from time import sleep
from traceback import print_exc

from modules.paths import TMP_PATH

CHECK_INTERVAL = 60


class SingletonPorcess(metaclass=ABCMeta):

    def __init__(self, name:str):
        self.is_restarting = False
        self.pid_fp = os.path.join(TMP_PATH, f'{name}.pid')

    def start(self):
        if os.path.exists(self.pid_fp):
            with open(self.pid_fp, 'r', encoding='utf-8') as fh:
                pid = int(fh.read().strip())

            r = subprocess.check_output(['TASKLIST', '/FI', f'"PID eq {pid}"']).strip()
            if 'No tasks are running' not in r:
                print(f'The service is running at pid = {pid}!')
                print(f'If this is a mistake, manaully remove lock file {self.pid_fp!r} and start again :)')
                exit(-1)
        
        pid = os.getpid()
        with open(self.pid_fp, 'w', encoding='utf-8') as fh:
            fh.write(pid)

        self.do_start()

    def stop(self):
        self.do_stop()

        if os.path.exists(self.pid_fp):
            os.unlink(self.pid_fp)

    def restart(self):
        # avoid double restart
        if self.is_restarting: return

        self.is_restarting = True
        self.stop()
        sleep(1)
        self.start()
        sleep(1)
        self.is_restarting = False

    @abstractmethod
    def do_start(self):
        pass
    
    @abstractmethod
    def do_stop(self):
        pass

    @abstractmethod
    def is_alive(self) -> bool:
        pass


class WebUIPorcess(SingletonPorcess):

    def __init__(self):
       super().__init__('webui')

       self.cmd = ['webui.cmd']

    def do_start(self):
        pass
    
    def do_stop(self):
        pass

    def is_alive(self) -> bool:
        pass


class NgrokPorcess(SingletonPorcess):

    def __init__(self):
        super().__init__('ngrok')

        self.cmd = ['ngrok.exe', 'http', '7860']

    def do_start(self):
        pass
    
    def do_stop(self):
        pass

    def is_alive(self) -> bool:
        pass


if __name__ == '__main__':
  # monitor worker processes
  webui = WebUIPorcess() ; webui.start()
  ngrok = NgrokPorcess() ; ngrok.start()

  # begin daemon
  try:
    while True:
      # check alive
      if not webui.is_alive(): webui.restart()
      if not ngrok.is_alive(): ngrok.restart()
      # sleep
      sleep(CHECK_INTERVAL)
  except KeyboardInterrupt:
    print('Exit by Ctrl+C')
  except Exception:
    print_exc()
  finally:
    ngrok.stop()
    webui.stop()
