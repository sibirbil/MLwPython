# Codes

Python scripts that reproduce the figures and tables in the slides, one script per lecture (`l01_figures.py`, `l02_...`, and so on).

Each script lists its dependencies at the top (inline script metadata), so it runs without any setup other than [uv](https://docs.astral.sh/uv/). From the `MLwPython/` folder:

```bash
uv run codes/l01_figures.py
```

The figures are written to `slides/figures/`. Datasets used in the lectures are in `codes/data/`; for example, `l02_figures.py` creates the synthetic student survey (`student_survey.csv`, and `student_survey_raw.csv` with typical data-quality problems) that the code on the Lecture 2 slides reads. The scripts use `numpy`, `pandas`, `matplotlib`, `seaborn` and `scikit-learn`, which are also included in the Anaconda installation used in the course; with Anaconda, run `python codes/l01_figures.py` instead.
