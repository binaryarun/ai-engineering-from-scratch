# Phase 0: Setup & Tooling: my revision notes

Status: complete (concepts covered; some hands-on exercises skipped, see "Come back to").
Machine: Mac (Apple Silicon, 48 GB unified memory). Shell: zsh. Terminal: WezTerm.

## My setup at a glance
- Python 3.13 (pyenv) and Git (Homebrew) pass `verify.py --route beginner`.
- `uv` installed. Project env: `.venv` in the repo root (numpy, matplotlib, jupyter, torch, datasets).
- Activate in every new terminal: `source .venv/bin/activate`
- GPU: no CUDA on a Mac. PyTorch uses **MPS** (Apple GPU): `torch.backends.mps.is_available()` -> True.
- Git remotes: `origin` = my fork (binaryarun), `upstream` = rohitg00 (original course). Pull course updates with `git fetch upstream` then `git merge upstream/main`.

## 0.01 Dev Environment
- Four layers, built bottom-up: system -> package manager -> runtimes -> AI libraries.
- `uv venv`, `source .venv/bin/activate`, `uv pip install ...`
- On a Mac install plain `torch` (no `--index-url .../cuXXX`). Check: `python -c "import torch; print(torch.backends.mps.is_available())"`
- Node/Rust/Julia: only when a lesson needs them.

## 0.02 Git & Collaboration
- Flow: working files -> `git add` -> `git commit` -> `git push` -> GitHub.
- A commit = snapshot of the whole project. A branch = a movable label on a commit.
- I can't push to the course repo, so I work in my fork.
- `.gitignore` should exclude `.venv/`, `*.pt`, `*.pth`, `*.safetensors` (big model files).
- `git remote set-url origin git@github.com:USER/REPO.git` switches HTTPS to SSH.

## 0.03 GPU Setup & Cloud
- Options: local NVIDIA GPU, Google Colab (free T4), rented cloud GPU ($0.20-2/hr).
- My Mac: MPS gave ~5x over CPU on a 5000x5000 matrix multiply (0.157s vs 0.035s). Time GPU code with a warm-up run and `torch.mps.synchronize()`.
- Rule of thumb: fp16 = 2 bytes per parameter, so ~2 GB per billion parameters. With 48 GB, about 14-15B parameters fit comfortably in fp16.

## 0.04 APIs & Keys (skipped)
- Pattern: endpoint + API key + JSON request -> JSON response.
- Never put keys in code or git. Use env variables (`export ANTHROPIC_API_KEY=...`) or a `.env` that is git-ignored.
- Errors: 401 bad key, 429 rate limit, 400 bad request.
- TODO before Phase 11: get an API key and run `phases/00-setup-and-tooling/04-apis-and-keys/code/first_api_call.py`.

## 0.05 Jupyter Notebooks
- A notebook = cells run by one kernel (a background Python process). Variables persist; cells run in any order you click (the main trap).
- `Shift+Enter` runs a cell. `%timeit` times small snippets. Fix confusion with Kernel > Restart & Run All.
- Explore in notebooks, ship in scripts.
- My timing: list comprehension 1.92 ms vs numpy 414 us for 100,000 random numbers (~4.6x).

## 0.06 Python Environments
- One isolated env per project avoids version fights. `uv` is the fast tool; `venv` is built in; conda only when you need non-Python dependencies.
- `pyproject.toml` lists dependencies. A lockfile pins exact versions. Never commit `.venv/`.
- Don't mix pip into a conda env.

## 0.07 Docker for AI (skipped)
- Container = packaged app + libraries. Image = recipe. Volume = shared folder. Compose = several services at once.
- On a Mac: no CUDA/MPS inside Docker, so skip the GPU image.
- TODO before RAG lessons: install Docker Desktop and run Qdrant (`docker compose up -d qdrant`).

## 0.08 Editor Setup
- VS Code (or Cursor). If `code` isn't found: Command Palette -> "Shell Command: Install 'code' command in PATH".
- Worth having: Python, Pylance, Jupyter, GitLens, Ruff/Black, format on save.

## 0.09 Data Management
- Hugging Face `datasets`: caches in `~/.cache/huggingface/`, can stream huge datasets.
- Formats: CSV/JSON for sharing, **Parquet** for storage, Arrow in memory.
- Splits: train / validation / test (about 80/10/10), always with a fixed `seed`.
- The lesson's name `glue` is stale. Use `nyu-mll/glue`. Newer libraries want `namespace/name`.
- "Unauthenticated requests" warning is harmless. A free HF token (`HF_TOKEN`) is optional.

## 0.10 Terminal & Shell
- Pipes: `|`. Redirects: `>` overwrite, `>>` append, `2>&1` merge errors.
- `grep "loss:" train.log | awk '{print $NF}'` pulls the last field of matching lines.
- Long jobs: `nohup cmd > log 2>&1 &`. I use WezTerm for split panes; tmux is only needed on remote Linux GPU boxes.
- zsh gotcha: pasting a command with a `# comment` fails. Paste commands without comments.

## 0.11 Linux for AI
- Remote GPU boxes are Ubuntu. Core commands: `ls -la`, `cd`, `cp -r`, `rm -rf` (permanent!), `grep -r`, `find`, `chmod +x`, `df -h`, `du -sh`, `rsync -avz`.
- `apt` replaces `brew` there. Linux filenames are case-sensitive.

## 0.12 Debugging & Profiling
- AI bugs often don't crash: they train silently on garbage.
- Tools: print tensor shape/dtype/device/NaN, conditional `breakpoint()`, logging, a `Timer`, `cProfile`, `tracemalloc`.
- Classic bugs: shape mismatch, NaN loss, data leakage (test accuracy too good), wrong device.
- On a Mac use MPS, not `torch.cuda.*`.

## Come back to
- 0.04 get an API key (before Phase 11)
- 0.07 Docker Desktop + Qdrant (before RAG)
- 0.12 run `debug_tools.py` and the NaN exercise
