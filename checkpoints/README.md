# Checkpoints

`kmmf_msrs_lite_best.pt` is an inference-only checkpoint trained on the MSRS
training pairs with the portable `lite` Mamba backend and the `pykan` fusion
backend. It must be loaded with `configs/msrs_lite.yaml`.

- Display epoch: 49
- Global optimizer step: 1,519
- Best validation loss: 0.087994
- Source weights: `w_vis = 0.5`, `w_ir = 0.5`
- SHA-256: `391204EB8F92667FD2954A1E58223767C8C65FB4742086C87498BF39C1ED1EC5`

The checkpoint stores only `model` and public metadata. It does not contain an
optimizer, scheduler, scaler, private path, or dataset file list. Use
`--pretrained` to initialize a new training run from it; do not use `--resume`.

The paper-style `official` VMamba configuration has a different backbone and
must be trained separately with `configs/msrs.yaml` on Linux.
