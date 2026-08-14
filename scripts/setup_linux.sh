#!/usr/bin/env bash
set -euo pipefail

python -m pip install --upgrade pip setuptools wheel ninja packaging
python -m pip install torch==2.4.0 torchvision==0.19.0 --index-url https://download.pytorch.org/whl/cu121
python -m pip install -r requirements.txt
python -m pip install mamba-ssm==2.2.4 --no-build-isolation

python smoke_test.py

