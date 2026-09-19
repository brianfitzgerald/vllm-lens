"""Put the examples/ directory on sys.path so example modules import as
top-level names (mirrors running a script with ``python examples/foo.py``)."""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# vLLM forks the engine core unless CUDA is initialized in this process. A fork
# of this process after it ran multi-threaded torch CPU ops can deadlock.
os.environ.setdefault("VLLM_WORKER_MULTIPROC_METHOD", "spawn")
