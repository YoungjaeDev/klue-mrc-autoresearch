import json
import os
from pathlib import Path
import sys
import tempfile
import time
import unittest

from autoresearch_lab.run import run_command, GpuLock


class ControlTests(unittest.TestCase):
    def test_utf8_split_chunks_and_environment_credential_redaction(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            env = {**os.environ, "EXAMPLE_API_KEY": "abcdef-private-token"}
            code = "import os,sys,time; b='한글'.encode(); os.write(1,b[:1]); time.sleep(.1); os.write(1,b[1:]); print(os.environ['EXAMPLE_API_KEY'])"
            run_command([sys.executable, "-c", code], root / "run", root / "state", 10, env=env)
            log = (root / "run/process.log").read_text(encoding="utf-8")
            self.assertIn("한글", log)
            self.assertNotIn("abcdef-private-token", log)
            self.assertIn("[REDACTED]", log)
    def test_normal_run_and_exclusive_evidence(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            args = [sys.executable, "-c", "print('done')"]
            result = run_command(args, root / "run", root / "state", 10)
            self.assertEqual(result["status"], "completed")
            with self.assertRaises(FileExistsError):
                run_command(args, root / "run", root / "state", 10)

    def test_timeout_kills_owned_descendants(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            marker = root / "escaped.txt"
            child = "import time,pathlib; time.sleep(2); pathlib.Path(%r).write_text('escaped')" % str(marker)
            parent = "import subprocess,sys,time; subprocess.Popen([sys.executable,'-c',%r]); time.sleep(30)" % child
            result = run_command([sys.executable, "-c", parent], root / "run", root / "state", .4)
            self.assertEqual(result["status"], "timeout")
            time.sleep(2.2)
            self.assertFalse(marker.exists())

    def test_gpu_error_latches_and_blocks_next_run(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            result = run_command([sys.executable, "-c", "import sys,time; sys.stdout.write('CUDA error: illegal memory access'); sys.stdout.flush(); time.sleep(3)"],
                root / "run", root / "state", 10, gpu=True)
            self.assertEqual(result["status"], "gpu_error")
            self.assertTrue((root / "state" / "STOP_GPU.json").is_file())
            with self.assertRaises(RuntimeError):
                run_command([sys.executable, "-c", "pass"], root / "run2", root / "state", 10, gpu=True)

    def test_frozen_change_invalidates_success(self):
        import hashlib
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            frozen = root / "fixed.txt"
            frozen.write_text("before")
            hashes = {str(frozen): hashlib.sha256(frozen.read_bytes()).hexdigest()}
            code = "from pathlib import Path; Path(%r).write_text('after')" % str(frozen)
            result = run_command([sys.executable, "-c", code], root / "run", root / "state", 10,
                frozen_files=hashes)
            self.assertEqual(result["status"], "frozen_input_changed")

    def test_lock_blocks_other_process(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            with GpuLock(root):
                import subprocess
                code = "from pathlib import Path; from autoresearch_lab.run import GpuLock; lock=GpuLock(Path(%r)); lock.__enter__()" % str(root)
                process = subprocess.run([sys.executable, "-c", code], capture_output=True)
                self.assertNotEqual(process.returncode, 0)

    @unittest.skipUnless(os.name == "nt", "Windows native exit code")
    def test_native_crash_latches_gpu(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            code = "import ctypes; ctypes.windll.kernel32.ExitProcess(0xC0000005)"
            result = run_command([sys.executable, "-c", code], root / "run", root / "state", 10, gpu=True)
            self.assertEqual(result["status"], "native_crash")
            self.assertTrue((root / "state" / "STOP_GPU.json").exists())


if __name__ == "__main__":
    unittest.main()
