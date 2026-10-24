# 统一编译入口：load() 只根据 sources 和编译选项判断是否要重新编译，
# 看不到 rmsnorm_ext.cu include 进来的 rmsNormWarp.cu。这里把它的哈希放进编译选项，
# kernel 一改就会重新编译。
import hashlib, pathlib
from torch.utils.cpp_extension import load

_DIR = pathlib.Path(__file__).resolve().parent
_DEPS = [_DIR / "rmsnorm_ext.cu", _DIR.parent / "rmsNormWarp.cu"]

def load_rmsnorm(verbose=False):
    digest = hashlib.sha1(b"".join(p.read_bytes() for p in _DEPS)).hexdigest()[:12]
    return load(name="rmsnorm_cuda", sources=[str(_DIR / "rmsnorm_ext.cu")],
                extra_cuda_cflags=["-O2", "-lineinfo", f"-DRMSNORM_SRC_HASH={digest}"],
                verbose=verbose)
