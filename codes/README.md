# Codes

Python scripts that reproduce the figures and tables in the slides, one script per lecture (`l01_figures.py`, `l02_...`, and so on).

Each script lists its dependencies at the top (inline script metadata), so it runs without any setup other than [uv](https://docs.astral.sh/uv/). From the `MLwPython/` folder:

```bash
uv run codes/l01_figures.py
```

The figures are written to `slides/figures/`. The scripts use `numpy`, `matplotlib` and `scikit-learn`, which are also included in the Anaconda installation used in the course; with Anaconda, run `python codes/l01_figures.py` instead.
