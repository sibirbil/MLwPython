# /// script
# requires-python = ">=3.11"
# dependencies = ["numpy", "pandas", "matplotlib", "scikit-learn"]
# ///
"""Figures for Lecture 3: Introduction to Supervised Learning.

Run from any folder with: uv run codes/l03_figures.py

The script reads the synthetic student survey codes/data/student_survey.csv,
which codes/l02_figures.py creates, and draws every figure of Lecture 3 into
slides/figures/.  It also prints the numbers quoted on the slides.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import ListedColormap
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor

ROOT = Path(__file__).resolve().parent.parent
FIG = ROOT / "slides" / "figures"
df = pd.read_csv(ROOT / "codes" / "data" / "student_survey.csv")

RED, BLUE, GREY = "#BC0031", "#25567B", "#9A9A9A"
plt.rcParams.update({
    "font.family": "sans-serif",
    "font.size": 12,
    "axes.titlesize": 12,
    "axes.spines.top": False,
    "axes.spines.right": False,
})


def save(fig, name):
    fig.savefig(FIG / name, bbox_inches="tight", metadata={"CreationDate": None})
    plt.close(fig)


# The task: predict whether a student passes from two features.
X = df[["study_hours", "social_media_hours"]]
y = df["passed"]
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)
XL = ("study hours per week", "social media (hours per day)")


def scatter(ax, X, y, s=22):
    ax.scatter(X.iloc[:, 0][y], X.iloc[:, 1][y], c=BLUE, marker="^", s=s, label="passed")
    ax.scatter(X.iloc[:, 0][~y], X.iloc[:, 1][~y], c=RED, marker="o", s=s, label="failed")
    ax.set_xlabel(XL[0])
    ax.set_ylabel(XL[1])


fig, ax = plt.subplots(figsize=(5.5, 3.4))
scatter(ax, X, y)
ax.legend(frameon=False, loc="upper right")
save(fig, "l03_task.pdf")

# 1-NN and 5-NN: the nearest neighbors of three new students.
new = pd.DataFrame([[8.0, 2.0], [13.0, 3.5], [19.0, 5.5]], columns=X.columns)
fig, axes = plt.subplots(1, 2, figsize=(9.5, 3.3), sharey=True)
for ax, k in zip(axes, [1, 5]):
    scatter(ax, X_train, y_train, s=16)
    nn = KNeighborsClassifier(n_neighbors=k).fit(X_train, y_train)
    dist, ind = nn.kneighbors(new)
    for point, neighbors in zip(new.to_numpy(), ind):
        for j in neighbors:
            ax.plot([point[0], X_train.iloc[j, 0]], [point[1], X_train.iloc[j, 1]],
                    color="black", lw=1.5)
    pred = nn.predict(new)
    ax.scatter(new.iloc[:, 0], new.iloc[:, 1], marker="*", s=320, c=[BLUE if p else RED for p in pred],
               edgecolor="black", zorder=4)
    ax.set_title(f"{k} nearest neighbor{'s' if k > 1 else ''}")
    ax.set_ylabel(XL[1] if k == 1 else "")
axes[0].legend(frameon=False, loc="upper right", fontsize=9)
save(fig, "l03_neighbors.pdf")

# Units matter: with study time in minutes, social media hardly counts.
point = pd.DataFrame([[12.0, 3.0]], columns=X.columns)
fig, axes = plt.subplots(1, 2, figsize=(9.5, 3.0))
for ax, factor, title in zip(axes, [1, 60], ["study time in hours", "study time in minutes"]):
    Xs = X_train.copy()
    Xs["study_hours"] = Xs["study_hours"] * factor
    p = point.copy()
    p["study_hours"] = p["study_hours"] * factor
    nn = KNeighborsClassifier(n_neighbors=5).fit(Xs, y_train)
    _, ind = nn.kneighbors(p)
    ax.scatter(X_train.iloc[:, 0], X_train.iloc[:, 1], c=GREY, s=12)
    ax.scatter(X_train.iloc[ind[0], 0], X_train.iloc[ind[0], 1], c=BLUE, s=40,
               label="5 nearest neighbors")
    ax.scatter(point.iloc[:, 0], point.iloc[:, 1], marker="*", s=260, c="black", zorder=4)
    ax.set_title(title)
    ax.set_xlabel("study hours per week")
axes[0].set_ylabel(XL[1])
axes[0].legend(frameon=False, loc="upper right", fontsize=9)
save(fig, "l03_units.pdf")

# Decision boundaries for k = 1, 5, 30.
xx, yy = np.meshgrid(np.linspace(2, 28, 400), np.linspace(0, 7.5, 400))
grid = pd.DataFrame({"study_hours": xx.ravel(), "social_media_hours": yy.ravel()})
fig, axes = plt.subplots(1, 3, figsize=(11.5, 3.2), sharey=True)
for ax, k in zip(axes, [1, 5, 30]):
    nn = KNeighborsClassifier(n_neighbors=k).fit(X_train, y_train)
    zz = nn.predict(grid).reshape(xx.shape)
    ax.contourf(xx, yy, zz, levels=[-0.5, 0.5, 1.5],
                cmap=ListedColormap(["#E8A6B4", "#A9C1D6"]))
    scatter(ax, X_train, y_train, s=12)
    ax.set_title(f"k = {k}")
    ax.set_ylabel(XL[1] if k == 1 else "")
save(fig, "l03_boundaries.pdf")

# Training and test accuracy against k.
ks = np.arange(1, 61)
train_acc, test_acc = [], []
for k in ks:
    nn = KNeighborsClassifier(n_neighbors=k).fit(X_train, y_train)
    train_acc.append(nn.score(X_train, y_train))
    test_acc.append(nn.score(X_test, y_test))
fig, ax = plt.subplots(figsize=(6.0, 3.2))
ax.plot(ks, train_acc, color=BLUE, lw=2, label="training data")
ax.plot(ks, test_acc, color=RED, lw=2, label="test data")
ax.set_xlabel("number of neighbors k")
ax.set_ylabel("accuracy")
ax.legend(frameon=False)
save(fig, "l03_accuracy_k.pdf")
print(f"1-NN training accuracy {train_acc[0]:.3f}, test accuracy {test_acc[0]:.3f}")
best = int(ks[np.argmax(test_acc)])
print(f"best k on the test set: {best}, test accuracy {max(test_acc):.3f}")
print(f"5-NN: training {train_acc[4]:.3f}, test {test_acc[4]:.3f}")

# The sweet spot (schematic).
c = np.linspace(0, 1, 200)
fig, ax = plt.subplots(figsize=(6.0, 3.2))
ax.plot(c, 0.55 + 0.43 * (1 - np.exp(-4 * c)), color=BLUE, lw=2, label="training data")
ax.plot(c, 0.55 + 0.3 * np.exp(-((c - 0.45) / 0.3) ** 2), color=RED, lw=2, label="new data")
ax.axvline(0.45, color=GREY, ls="--", lw=1)
ax.text(0.47, 0.47, "sweet spot", color=GREY)
ax.text(0.03, 0.47, "underfitting", fontsize=11)
ax.text(0.78, 0.47, "overfitting", fontsize=11)
ax.set_xticks([])
ax.set_yticks([])
ax.set_xlabel("model complexity")
ax.set_ylabel("accuracy")
ax.set_ylim(0.44, 1.0)
ax.legend(frameon=False, loc="center right")
save(fig, "l03_sweetspot.pdf")

# kNN regression: predict the grade from study hours.
Xr = df[["study_hours"]]
yr = df["grade"]
line = pd.DataFrame({"study_hours": np.linspace(0, 30, 500)})
fig, axes = plt.subplots(1, 2, figsize=(9.5, 3.0), sharey=True)
for ax, k in zip(axes, [1, 20]):
    reg = KNeighborsRegressor(n_neighbors=k).fit(Xr, yr)
    ax.scatter(Xr["study_hours"], yr, c=GREY, s=12)
    ax.plot(line["study_hours"], reg.predict(line), color=RED, lw=2)
    ax.set_title(f"k = {k}: average grade of the {k} nearest" if k > 1
                 else "k = 1: grade of the nearest student")
    ax.set_xlabel("study hours per week")
axes[0].set_ylabel("grade")
save(fig, "l03_regression.pdf")

# Grid search with cross-validation.
grid_search = GridSearchCV(KNeighborsClassifier(), {"n_neighbors": range(1, 61)}, cv=5)
grid_search.fit(X_train, y_train)
res = pd.DataFrame(grid_search.cv_results_)
mean, std = res["mean_test_score"].to_numpy(), res["std_test_score"].to_numpy()
fig, ax = plt.subplots(figsize=(6.0, 3.2))
ax.plot(ks, mean, color=BLUE, lw=2, label="mean over 5 folds")
ax.fill_between(ks, mean - std, mean + std, color=BLUE, alpha=0.2, label="± one std. dev.")
ax.axvline(grid_search.best_params_["n_neighbors"], color=GREY, ls="--", lw=1)
ax.set_xlabel("number of neighbors k")
ax.set_ylabel("cross-validation accuracy")
ax.legend(frameon=False, loc="lower right")
save(fig, "l03_gridsearch.pdf")
print(f"grid search: best k {grid_search.best_params_['n_neighbors']}, "
      f"CV accuracy {grid_search.best_score_:.3f}, "
      f"test accuracy {grid_search.score(X_test, y_test):.3f}")
print(f"class balance: {y.mean():.2f} passed")
