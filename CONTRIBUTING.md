# Contributing

Contributions are welcome through GitHub issues and pull requests.

Before submitting a change:

1. Create a dedicated branch.
2. Keep data, checkpoints, logs, and generated outputs out of the commit.
3. Run `python smoke_test.py`.
4. Run `python -m compileall -q datasets losses models utils train.py test.py`.
5. Document configuration or behavior changes in the pull request.

Bug reports should include the operating system, Python/PyTorch/CUDA versions,
configuration file, command, and complete error message.
