"""
Importing this module, before anything else, works around a real hang: on
macOS, LightGBM and PyTorch each bundle their own OpenMP runtime
(libomp/libiomp). When both get initialized in the same process, the second
one to spin up its thread pool can deadlock silently -- no crash, no
traceback, the process just sits at 0% CPU forever. It reproduced 100% of
the time in this repo whenever GBDT training was followed by two-tower
(PyTorch) training in the same process.

The fix is exactly two environment variables, but they only take effect if
set *before* either library's native extension is imported -- so every
entry point does `from src import _threading_fix` as its first import.
"""
import os

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
os.environ.setdefault("OMP_NUM_THREADS", "1")
