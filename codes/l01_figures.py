# /// script
# requires-python = ">=3.11"
# dependencies = ["numpy", "matplotlib"]
# ///
"""Figures for Lecture 1: Introduction to Machine Learning.

Run from any folder with: uv run codes/l01_figures.py
The figures are written to slides/figures/.
"""

from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

RED, BLUE, ORANGE, GREY = "#BC0031", "#25567B", "#D98C00", "#9A9A9A"
plt.rcParams.update({
    "font.family": "sans-serif",
    "font.size": 13,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.titlesize": 13,
})
rng = np.random.default_rng(0)
OUT = Path(__file__).resolve().parent.parent / "slides" / "figures"


def save(fig, name):
    fig.savefig(OUT / name, bbox_inches="tight", metadata={"CreationDate": None})
    plt.close(fig)


# Supervised vs. unsupervised: the same points, with and without labels.
a = rng.normal([1.5, 1.5], 0.55, size=(40, 2))
b = rng.normal([3.5, 3.2], 0.55, size=(40, 2))
fig, axes = plt.subplots(1, 2, figsize=(8.0, 3.2))
axes[0].scatter(*a.T, c=RED, marker="o", label="class A", s=22)
axes[0].scatter(*b.T, c=BLUE, marker="^", label="class B", s=22)
axes[0].set_title("Supervised: with labels")
axes[0].legend(frameon=False, loc="upper left")
axes[1].scatter(*np.vstack([a, b]).T, c=GREY, s=22)
axes[1].set_title("Unsupervised: no labels")
for ax in axes:
    ax.set_xlabel("feature 1")
    ax.set_ylabel("feature 2")
    ax.set_xticks([])
    ax.set_yticks([])
save(fig, "l01_supervised_unsupervised.pdf")

# Classification vs. regression: study hours as the single feature.
hours = rng.uniform(0, 10, 60)
passed = (hours + rng.normal(0, 1.5, 60) > 5).astype(int)
grade = np.clip(1 + 0.8 * hours + rng.normal(0, 0.8, 60), 1, 10)
fig, axes = plt.subplots(1, 2, figsize=(8.0, 3.2))
axes[0].scatter(hours[passed == 1], np.ones(passed.sum()), c=BLUE, marker="^", s=22, label="pass")
axes[0].scatter(hours[passed == 0], np.zeros((1 - passed).sum()), c=RED, s=22, label="fail")
axes[0].axvline(5, color=GREY, ls="--", lw=1)
axes[0].set_yticks([0, 1], ["fail", "pass"])
axes[0].set_title("Classification: pass or fail?")
axes[1].scatter(hours, grade, c=BLUE, s=22)
xs = np.linspace(0, 10, 2)
axes[1].plot(xs, 1 + 0.8 * xs, color=RED, lw=2)
axes[1].set_ylabel("exam grade")
axes[1].set_title("Regression: which grade?")
for ax in axes:
    ax.set_xlabel("hours of study per week")
save(fig, "l01_classification_regression.pdf")

# Underfitting, a good fit, overfitting: polynomials of degree 1, 3, 15.
x = np.sort(rng.uniform(0, 1, 20))
y = np.sin(2 * np.pi * x) + rng.normal(0, 0.25, x.size)
grid = np.linspace(0, 1, 400)
fig, axes = plt.subplots(1, 3, figsize=(9.0, 2.8), sharey=True)
titles = ["Too simple (underfitting)", "About right", "Too complex (overfitting)"]
for ax, deg, title in zip(axes, [1, 3, 15], titles):
    coef = np.polynomial.polynomial.Polynomial.fit(x, y, deg)
    ax.plot(grid, np.sin(2 * np.pi * grid), color=GREY, ls="--", lw=1, label="true pattern")
    ax.plot(grid, coef(grid), color=RED, lw=2, label="model")
    ax.scatter(x, y, c=BLUE, s=18, zorder=3, label="training data")
    ax.set_ylim(-2, 2)
    ax.set_title(title)
    ax.set_xticks([])
    ax.set_yticks([])
axes[0].legend(frameon=False, loc="lower left", fontsize=8)
save(fig, "l01_fitting.pdf")
