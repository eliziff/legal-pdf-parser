#!/usr/bin/env bash
set -Eeuo pipefail

if [[ $# -ne 1 ]]; then
  echo "usage: setup_paddlex_wsl.sh RUN_ROOT" >&2
  exit 2
fi

run_root="$1"
venv="$run_root/venv"
log="$run_root/logs/setup.log"
status="$run_root/setup_status.json"
mkdir -p "$run_root/logs"
exec > >(tee -a "$log") 2>&1

write_status() {
  local phase="$1"
  local detail="${2:-}"
  local temporary="${status}.part"
  printf '{"schema_version":"legalpdf.ppdoc_lite_training_setup.v1","phase":"%s","detail":"%s","updated_at":"%s"}\n' \
    "$phase" "${detail//\"/\\\"}" "$(date -u +%Y-%m-%dT%H:%M:%SZ)" > "$temporary"
  mv "$temporary" "$status"
  echo "[ppdoc-lite setup] phase=$phase detail=$detail"
}

trap 'write_status failed "line $LINENO: $BASH_COMMAND"' ERR

write_status creating_venv
if [[ ! -x "$venv/bin/python" ]]; then
  /usr/bin/python3.12 -m venv "$venv"
fi

python="$venv/bin/python"
write_status installing_build_tools
"$python" -m pip install --upgrade pip 'setuptools<81' wheel

write_status installing_paddle "paddlepaddle-gpu 3.2.0 CUDA 12.6"
"$python" -m pip install paddlepaddle-gpu==3.2.0 \
  -i https://www.paddlepaddle.org.cn/packages/stable/cu126/

write_status installing_paddlex "paddlex base 3.6.1"
"$python" -m pip install 'paddlex[base]==3.6.1'

write_status preparing_detection_plugin
"$python" - <<'PY'
from pathlib import Path

import paddlex

(Path(paddlex.__file__).resolve().parent / "repo_manager" / "repos").mkdir(
    parents=True,
    exist_ok=True,
)
PY

write_status installing_detection_plugin "axis-aligned layout; skip rotated-box-only custom ops"
"$python" - <<'PY'
from pathlib import Path

import paddle
import paddlex
from paddlex.repo_manager import setup

repo = Path(paddlex.__file__).resolve().parent / "repo_manager/repos/PaddleDetection"
paddle.set_device("cpu")
setup(
    ["PaddleDetection"],
    update_repos=not (repo / ".git").is_dir(),
    use_local_repos=(repo / ".git").is_dir(),
)
PY

write_status installing_registered_configs "exact PaddleX 3.6.1 tagged configs"
"$python" - "$run_root" <<'PY'
import hashlib
import json
import sys
import urllib.request
from pathlib import Path

import paddlex

version = paddlex.__version__
if version != "3.6.1":
    raise RuntimeError(f"Refusing unpinned PaddleX version: {version}")
names = (
    "PicoDet-XS.yaml",
    "PicoDet-S_layout_17cls.yaml",
    "PicoDet-L_layout_17cls.yaml",
    "PP-DocLayout-S.yaml",
    "PP-DocLayout-M.yaml",
    "PP-DocLayoutV3.yaml",
)
base = (
    "https://raw.githubusercontent.com/PaddlePaddle/PaddleX/"
    f"v{version}/paddlex/repo_apis/PaddleDetection_api/configs"
)
destination = (
    Path(paddlex.__file__).resolve().parent
    / "repo_apis/PaddleDetection_api/configs"
)
destination.mkdir(parents=True, exist_ok=True)
receipt = {"paddlex_version": version, "source_tag": f"v{version}", "files": {}}
for name in names:
    url = f"{base}/{name}"
    with urllib.request.urlopen(url, timeout=60) as response:
        data = response.read()
    if not data.strip():
        raise RuntimeError(f"Downloaded an empty config: {url}")
    path = destination / name
    path.write_bytes(data)
    receipt["files"][name] = {
        "url": url,
        "sha256": hashlib.sha256(data).hexdigest(),
        "bytes": len(data),
    }
run_root = Path(sys.argv[1])
temporary = run_root / "registered_configs.json.part"
temporary.write_text(json.dumps(receipt, indent=2) + "\n", encoding="utf-8")
temporary.replace(run_root / "registered_configs.json")
print(json.dumps(receipt, indent=2))
PY

write_status verifying
"$python" - <<'PY'
import json
from pathlib import Path

import paddle
import paddlex
from paddlex.repo_apis.base.register import get_registered_model_info

root = Path(paddlex.__file__).resolve().parent
models = {}
for name in (
    "PicoDet-XS",
    "PicoDet-S_layout_17cls",
    "PicoDet-L_layout_17cls",
    "PP-DocLayout-S",
    "PP-DocLayout-M",
    "PP-DocLayoutV3",
):
    try:
        info = dict(get_registered_model_info(name))
        config = Path(str(info["config_path"]))
        if not config.is_file():
            raise FileNotFoundError(config)
        info["config_exists"] = True
        models[name] = info
    except Exception as exc:
        models[name] = {"error": f"{type(exc).__name__}: {exc}"}
print(
    json.dumps(
        {
            "paddle": paddle.__version__,
            "paddlex": paddlex.__version__,
            "cuda_devices": paddle.device.cuda.device_count(),
            "paddlex_root": str(root),
            "models": models,
        },
        indent=2,
        default=str,
    )
)
PY

write_status complete
