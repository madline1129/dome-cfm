#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT_DIR}"

python -m pip install --upgrade pip==23.2.1 "setuptools>=68,<70" "wheel>=0.41,<0.43"
python -m pip install -r requirements_torchcfm.txt -c constraints_torchcfm.txt

# torchcfm 1.0.5 与 DOME 的 Python 3.8 / pandas 2.0.3 / torch 2.0.1 更匹配。
# 这里用 --no-deps，防止 pip 根据 torchcfm 的元数据升级已固定的核心依赖。
python -m pip install --no-deps "torchcfm @ git+https://github.com/atong01/conditional-flow-matching.git@1.0.5"

python - <<'PY'
from torchcfm.conditional_flow_matching import ConditionalFlowMatcher
import torch

print("torch:", torch.__version__)
print("torchcfm import: OK")
print("ConditionalFlowMatcher:", ConditionalFlowMatcher)
PY
