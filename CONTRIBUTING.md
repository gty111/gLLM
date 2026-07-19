# Contributing to gLLM

Thanks for your interest in contributing to gLLM. This document covers how to
set up a development environment, make changes, and open a pull request.

## Quick start

```bash
# 1. Fork on GitHub, then clone your fork
git clone git@github.com:<your-user>/gLLM.git
cd gLLM

# 2. Create an environment and install in editable mode (Linux + CUDA required)
uv pip install -e .

# 3. Create a branch
git checkout -b feat/my-change

# 4. Smoke-check the install
python -c "import gllm; print('ok')"
```

gLLM currently targets **Linux** (including WSL). See [README.md](./README.md)
for supported models, launch modes, and serving examples.

## Development workflow

1. Prefer a focused change — one feature or fix per PR.
2. Match existing style around the code you touch (`gllm/`, `examples/`,
   `benchmarks/`).
3. If you change serving behavior, verify with a small model:
   ```bash
   python -m gllm.entrypoints.api_server --model-path $MODEL_PATH
   python examples/client.py --port $PORT
   ```
4. Open a PR against `master` and link related issues (`Closes #123` /
   `Refs #123`). Roadmap tracking lives in
   [issue #20](https://github.com/gty111/gLLM/issues/20).

## Pull requests

- Describe **what** changed and **why**.
- Include a short **test plan** (commands or scenarios reviewers can reproduce).
- Update docs / examples when user-facing behavior or flags change.
- Bump the version in `setup.py` and
  `gllm/entrypoints/api_server.py` only when intentionally cutting a release.
- Keep diffs small and avoid drive-by refactors unrelated to the PR.

## Reporting issues

When filing a bug, please include:

- gLLM version (`GET /version` or the package version)
- Model name / path and relevant launch flags (`--tp`, `--pp`, `--ep`, …)
- GPU / CUDA / PyTorch versions
- Minimal reproduction steps and error logs

Feature ideas and larger design discussions are also welcome via GitHub Issues.

## License

By contributing, you agree that your contributions will be licensed under the
[Apache License 2.0](./LICENSE).
