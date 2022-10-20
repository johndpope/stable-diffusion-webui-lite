import time
from collections import defaultdict
from threading import Thread, Event

import torch


class HardwareMonitor(Thread):

    disabled = False
    run_flag = None
    update_interval = None
    device = None
    data = None

    def __init__(self, update_interval):
        super().__init__(self)

        self.device = 'cuda'
        self.update_interval = update_interval

        self.daemon = True
        self.run_flag = Event()
        self.data = defaultdict(int)

        try:
            torch.cuda.mem_get_info()
            torch.cuda.memory_stats(self.device)
        except Exception as e:  # AMD or whatever
            print(f"Warning: caught exception '{e}', memory monitor disabled")
            self.disabled = True

    def run(self):
        if self.disabled: return

        while True:
            self.run_flag.wait()

            torch.cuda.reset_peak_memory_stats()
            self.data.clear()

            if self.update_interval <= 0:
                self.run_flag.clear()
                continue

            self.data["min_free"] = torch.cuda.mem_get_info()[0]

            while self.run_flag.is_set():
                free, total = torch.cuda.mem_get_info()  # calling with self.device errors, torch bug?
                self.data["min_free"] = min(self.data["min_free"], free)

                time.sleep(1 / self.update_interval)

    def dump_debug(self):
        print(self, 'recorded data:')
        for k, v in self.read().items():
            print(k, -(v // -(1024 ** 2)))

        print(self, 'raw torch memory stats:')
        tm = torch.cuda.memory_stats(self.device)
        for k, v in tm.items():
            if 'bytes' not in k:
                continue
            print('\t' if 'peak' in k else '', k, -(v // -(1024 ** 2)))

        print(torch.cuda.memory_summary())

    def monitor(self):
        self.run_flag.set()

    def read(self):
        if not self.disabled:
            free, total = torch.cuda.mem_get_info()
            self.data["total"] = total

            torch_stats = torch.cuda.memory_stats(self.device)
            self.data["active_peak"] = torch_stats["active_bytes.all.peak"]
            self.data["reserved_peak"] = torch_stats["reserved_bytes.all.peak"]
            self.data["system_peak"] = total - self.data["min_free"]

        return self.data

    def stop(self):
        self.run_flag.clear()
        return self.read()
