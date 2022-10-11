#!/usr/bin/env python3
# Author: Armit
# Create Time: 2022/10/11 

from time import sleep
from traceback import print_exc

CHECK_INTERVAL = 60


class WebUIPorcess:

  def __init__(self):
    self.is_restarting = False

  def start(self):
    pass

  def stop(self):
    pass

  def restart(self):
    # avoid double restart
    if self.is_restarting: return

    self.is_restarting = True
    self.stop()
    self.start()
    self.is_restarting = False

  def is_alive(self) -> bool:
    pass


class NgrokPorcess:

  def __init__(self):
    self.is_restarting = False

  def start(self):
    pass

  def stop(self):
    pass

  def restart(self):
    # avoid double restart
    if self.is_restarting: return

    self.is_restarting = True
    self.stop()
    self.start()
    self.is_restarting = False

  def is_alive(self) -> bool:
    pass


if __name__ == "__main__":
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
