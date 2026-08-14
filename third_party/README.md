# Third-party Components

## VMamba

- Repository: https://github.com/MzeroMiko/VMamba
- Vendored files: `VMamba/vmamba.py` and `VMamba/LICENSE`
- Commit: `2ed52ead062a51a64521ed3871d52914bf532876`

The official `VSSBlock`/`SS2D` implementation is used when a selective-scan
backend is available. The local lite block supports Windows and lightweight
experiments but does not claim to reproduce SS2D.

## pykan

- Repository: https://github.com/KindXiaoming/pykan
- Package: `pykan==0.2.8`

Formal training uses `kan.KAN` with its symbolic branch, activation recording,
and automatic checkpointing disabled via speed mode. KAN is evaluated on a
compact pooled implicit grid instead of every full-resolution pixel.
