from __future__ import annotations

import json

import paddle

try:
    import paddleslim
except ImportError:
    paddleslim = None

print(
    json.dumps(
        {
            "paddle": paddle.__version__,
            "paddleslim": getattr(paddleslim, "__version__", None),
            "cuda": paddle.is_compiled_with_cuda(),
            "cuda_devices": paddle.device.cuda.device_count(),
            "pir_api": hasattr(paddle, "pir"),
        },
        sort_keys=True,
    )
)
