# Contributing to LumoraQ

---

## Workflow

1. Branch from `dev`, not `main`
2. Name branches: `feature/your-feature` or `fix/your-fix`
3. One experiment per notebook
4. Update `progress.md` with any new results
5. Open a PR into `dev` when ready

---

## Adding an Experiment

- Add your model class to `src/models.py`
- Create a new notebook in `notebooks/`
- Save results to `results/` as a `.json` file
- Update `progress.md`

---

## Environment Changes

### Update the environment
```bash
conda env update -f environment.yml --prune
```

### Export the lockfile
```bash
conda env export > environment.lock.yml
```

> **Windows users:** if `conda env export` produces a lockfile with Windows-specific build strings, commit it as `environment.lock.win.yml` instead so it doesn't break Mac/Linux contributors. The shared `environment.yml` remains the source of truth for all platforms.

### Commit both files
```bash
git add environment.yml environment.lock.yml
git commit -m "chore: update environment dependencies"
```

---

## Commit Convention

Every commit message follows this format:

```
type: short description in present tense
```

| Type | When to use |
|------|-------------|
| `feat` | Adding something new |
| `fix` | Fixing something broken |
| `docs` | README, CONTRIBUTING, progress.md |
| `results` | Logging experiment outcomes |
| `refactor` | Restructuring code, no behavior change |
| `chore` | Environment, .gitignore, tooling |

---

## Platform Notes

### Mac (Apple Silicon)
PyTorch will automatically use MPS acceleration. No extra steps needed — the `get_device()` utility in `src/utils.py` handles this.

### Windows
- Use **Git Bash** or **Anaconda Prompt** for all `conda` and `git` commands
- If you hit a `conda` not found error in Git Bash, run:
```bash
conda init bash
```
Then restart Git Bash and try again.
- PyTorch will use CUDA if an NVIDIA GPU is available, otherwise CPU

### Linux
PyTorch will use CUDA if available, otherwise CPU. No extra steps needed.