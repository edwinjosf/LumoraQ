## Contributing to LumoraQ

### Workflow
1. Branch from dev, not main
2. Name branches: feature/your-feature or fix/your-fix
3. One experiment per notebook
4. Update progress.md with any new results
5. Open a PR into dev when ready

### Adding an experiment
- Add model to src/models.py
- Create notebook in notebooks/
- Save results to results/ as .json
- Update progress.md

### Environment changes
- Edit environment.yml
- Run: conda env update -f environment.yml --prune
- Export lockfile: conda env export > environment.lock.yml
- Commit both files
