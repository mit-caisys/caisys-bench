"""
Process wrapper for system monitoring commands.

Provides a simple interface to start, stop, restart, and check
status of subprocesses (used by SAR and DCGMI monitors).
"""

import subprocess


class Proc:
    def __init__(self, cmd, output_file):
        self.cmd = cmd
        self.output_file = output_file
        self.process = None

    def start(self):
        self.process = subprocess.Popen(
            self.cmd,
            shell=False,
            stdout=self.output_file,
            stderr=self.output_file,
        )

    def stop(self):
        self.process.kill()

    def restart(self):
        self.process.kill()
        self.start()

    def check(self):
        return self.process.poll() is None
