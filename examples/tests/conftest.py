"""Put the examples/ directory on sys.path so example modules import as
top-level names (mirrors running a script with ``python examples/foo.py``)."""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# A forked engine core can deadlock after torch CPU ops in this process.
os.environ.setdefault("VLLM_WORKER_MULTIPROC_METHOD", "spawn")
