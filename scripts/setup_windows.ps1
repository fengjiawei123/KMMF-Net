$ErrorActionPreference = "Stop"

python -m pip install --upgrade pip setuptools wheel
python -m pip install torch==2.4.0 torchvision==0.19.0 --index-url https://download.pytorch.org/whl/cu121
python -m pip install -r requirements.txt

Write-Host "Windows uses the portable lite Mamba backend for smoke tests."
Write-Host "Use Linux with mamba-ssm for formal VMamba training."
python smoke_test.py

