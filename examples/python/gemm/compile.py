from concurrent.futures import ThreadPoolExecutor

import os
import importlib.util
from collections import defaultdict

import subprocess
import ctypes
import torch

__all__ = [
    "Compile",
]

cutlass_include_dir = os.path.join(os.path.dirname(__file__),
                                   "../../../3rd-party/cutlass/include")
tiledcuda_include_dir = os.path.join(os.path.dirname(__file__),
                                     "../../../include/")
csrc_include_dir = os.path.join(os.path.dirname(__file__), "csrc")


class Compile:

    def __init__(self, file_name, tmp_dir):
        self.tmp_dir = tmp_dir
        self.file_name = file_name

        if not os.path.exists(self.tmp_dir):
            os.makedirs(self.tmp_dir)

        compute_capability = torch.cuda.get_device_capability()
        self.cc = f"{compute_capability[0]}{compute_capability[1]}"

        self.nvcc_path = self._find_nvcc_path()

    def _find_nvcc_path(self):

        def py_str(x):
            return x.decode('utf-8')

        if "CUDA_PATH" in os.environ:
            return os.environ["CUDA_PATH"]

        cmd = ["which", "nvcc"]
        proc = subprocess.Popen(cmd,
                                stdout=subprocess.PIPE,
                                stderr=subprocess.STDOUT)
        (out, _) = proc.communicate()

        if proc.returncode == 0:
            return py_str(out.strip())
        else:
            raise RuntimeError("Cannot find cuda path")

    def _create_entry_code(self, M: int, N: int, K: int):
        entry_code_path = "entry.py"
        spec = importlib.util.spec_from_file_location("entry_code",
                                                      entry_code_path)
        foo = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(foo)

        shape = defaultdict(int)
        shape["kM"] = M
        shape["kN"] = N
        shape["kK"] = K

        return foo.types.format_map(shape) + foo.entry

    def compile(self, M, N, K, compute_capability="80", timeout: float = None):
        temp_dir = self.tmp_dir
        lib_name = self.file_name.replace(".cu", ".so")
        lib_path = os.path.join(temp_dir, lib_name)

        entry_code = self._create_entry_code(M, N, K)

        source_path = os.path.join(temp_dir, self.file_name)
        with open(source_path, "w") as f:
            f.write(entry_code)

        if os.path.exists(lib_path):
            return lib_path

        command = [
            self.nvcc_path, "-std=c++20", "-O3", "--use_fast_math",
            "--expt-relaxed-constexpr", "--disable-warnings",
            "--compiler-options", "'-fPIC'", "--shared", source_path, "-lcuda",
            f"-gencode=arch=compute_{self.cc},code=sm_{self.cc}",
            f"-I{cutlass_include_dir}", f"-I{tiledcuda_include_dir}",
            f"-I{csrc_include_dir}", "-o", lib_path
        ]
        try:
            ret = subprocess.run(command, timeout=timeout)
        except subprocess.TimeoutExpired:
            return None
        if ret.returncode == 0:
            return lib_path
        else:
            raise RuntimeError("Compilation failed")

    def apply(self, lib_path, torch_array: list, device: int):
        lib = ctypes.CDLL(lib_path)

        lib.kernel_entry.restype = ctypes.c_int
        torch.cuda.set_device(device)

        ret = lib.kernel_entry(
            *[ctypes.c_void_p(arr.data_ptr()) for arr in torch_array])
        return ret
