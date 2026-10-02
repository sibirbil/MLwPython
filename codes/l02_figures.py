# /// script
# requires-python = ">=3.11"
# dependencies = ["numpy", "pandas", "matplotlib", "seaborn"]
# ///
"""Data and figures for Lecture 2: Visualization and Matplotlib.

Run from any folder with: uv run codes/l02_figures.py

The script
  1. creates a synthetic student survey and saves it to codes/data/:
       student_survey.csv      the clean version, used in the code on the slides
       student_survey_raw.csv  the same survey with typical data-quality problems
  2. draws every figure of Lecture 2 into slides/figures/.

All data are synthetic, except Anscombe's quartet (Anscombe, 1973) and the
PhD and arcade figures (as used in Mueller's 2017 course notebook).
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

ROOT = Path(__file__).resolve().parent.parent
FIG = ROOT / "slides" / "figures"
DATA = ROOT / "codes" / "data"
DATA.mkdir(exist_ok=True)

RED, BLUE, ORANGE, GREEN, GREY = "#BC0031", "#25567B", "#D98C00", "#2E7D5B", "#9A9A9A"
plt.rcParams.update({
    "font.family": "sans-serif",
    "font.size": 12,
    "axes.titlesize": 12,
    "axes.spines.top": False,
    "axes.spines.right": False,
})
rng = np.random.default_rng(2026)


def save(fig, name):
    fig.savefig(FIG / name, bbox_inches="tight", metadata={"CreationDate": None})
    plt.close(fig)


# ---------------------------------------------------------------- the data --

def make_survey(n=300):
    """A synthetic survey of n students in an introductory course."""
    programs = ["Economics", "Business", "Communication", "Psychology", "Other"]
    program = rng.choice(programs, size=n, p=[0.30, 0.30, 0.15, 0.15, 0.10])
    age = rng.integers(19, 27, size=n)
    study = np.round(rng.gamma(shape=6, scale=2, size=n), 1)          # hours per week
    social = np.round(np.clip(rng.normal(3, 1.2, size=n), 0.2, 9), 1)  # hours per day
    sleep = np.round(np.clip(rng.normal(7, 0.9, size=n), 4, 10), 1)    # hours per night
    grade = 3.0 + 0.25 * study - 0.35 * social + 0.3 * (sleep - 7) + rng.normal(0, 1.0, size=n)
    grade = np.round(np.clip(grade, 1, 10), 1)
    satisfaction = np.clip(np.round(1 + 0.4 * grade + rng.normal(0, 0.8, size=n)), 1, 5).astype(int)
    return pd.DataFrame({
        "program": program,
        "age": age,
        "study_hours": study,
        "social_media_hours": social,
        "sleep_hours": sleep,
        "satisfaction": satisfaction,
        "grade": grade,
        "passed": grade >= 5.5,
    })


def make_raw(df):
    """The same survey with the problems that real survey exports have."""
    raw = df.copy()
    idx = rng.permutation(len(raw))
    # Definition mismatch: 40 respondents reported study hours per day, not per week.
    raw.loc[idx[:40], "study_hours"] = np.round(raw.loc[idx[:40], "study_hours"] / 7, 1)
    # Missing values.
    raw.loc[idx[40:58], "sleep_hours"] = np.nan
    raw.loc[idx[58:68], "social_media_hours"] = np.nan
    # Typing errors: 70 instead of 7.0 hours of sleep, 210 instead of 21 years.
    raw.loc[idx[68], "sleep_hours"] = 70.0
    raw.loc[idx[69], "age"] = 210
    # Wrong data type: grades exported as text with a decimal comma.
    raw["grade"] = raw["grade"].map(lambda g: f"{g:.1f}".replace(".", ","))
    # Duplicate records.
    raw = pd.concat([raw, raw.iloc[idx[70:75]]], ignore_index=True)
    return raw


df = make_survey()
raw = make_raw(df)
df.to_csv(DATA / "student_survey.csv", index=False)
raw.to_csv(DATA / "student_survey_raw.csv", index=False)

# ------------------------------------------------- session 1: why visualize --

# Anscombe's quartet (Anscombe, 1973).
x123 = [10, 8, 13, 9, 11, 14, 6, 4, 12, 7, 5]
anscombe = {
    "I": (x123, [8.04, 6.95, 7.58, 8.81, 8.33, 9.96, 7.24, 4.26, 10.84, 4.82, 5.68]),
    "II": (x123, [9.14, 8.14, 8.74, 8.77, 9.26, 8.10, 6.13, 3.10, 9.13, 7.26, 4.74]),
    "III": (x123, [7.46, 6.77, 12.74, 7.11, 7.81, 8.84, 6.08, 5.39, 8.15, 6.42, 5.73]),
    "IV": ([8, 8, 8, 8, 8, 8, 8, 19, 8, 8, 8],
           [6.58, 5.76, 7.71, 8.84, 8.47, 7.04, 5.25, 12.50, 5.56, 7.91, 6.89]),
}
fig, axes = plt.subplots(1, 4, figsize=(12, 2.9), sharex=True, sharey=True)
grid = np.array([3, 20])
for ax, (name, (x, y)) in zip(axes, anscombe.items()):
    x, y = np.array(x, float), np.array(y)
    slope, intercept = np.polyfit(x, y, 1)
    ax.plot(grid, intercept + slope * grid, color=GREY, lw=1.5)
    ax.scatter(x, y, color=BLUE, s=30, zorder=3)
    ax.set_title(f"Dataset {name}")
    print(f"Anscombe {name}: mean x {x.mean():.2f}, mean y {y.mean():.2f}, "
          f"corr {np.corrcoef(x, y)[0, 1]:.3f}, line y = {intercept:.2f} + {slope:.3f} x")
save(fig, "l02_anscombe.pdf")

# Questionable causality: two series on one axis, then on twin axes.
years = np.arange(2000, 2010)
phds = np.array([1050, 1010, 919, 993, 1076, 1205, 1325, 1393, 1399, 1554])
arcades = np.array([1196, 1176, 1269, 1240, 1307, 1435, 1601, 1654, 1803, 1734])  # million USD
fig, ax1 = plt.subplots(figsize=(6.0, 3.2))
ax1.plot(years, phds, color=BLUE, lw=2, marker="o")
ax1.set_ylabel("math PhDs awarded", color=BLUE)
ax2 = ax1.twinx()
ax2.plot(years, arcades, color=RED, lw=2, marker="s")
ax2.set_ylabel("arcade revenue (million USD)", color=RED)
ax2.spines["right"].set_visible(True)
print(f"PhDs vs arcades: corr {np.corrcoef(phds, arcades)[0, 1]:.3f}")
save(fig, "l02_spurious.pdf")

# Wrong precision: the Kansas default location (synthetic illustration).
lon = rng.uniform(-124, -70, 400)
lat = rng.uniform(26, 48, 400)
fig, ax = plt.subplots(figsize=(6.0, 3.2))
ax.scatter(lon, lat, s=8, color=GREY, alpha=0.7, label="addresses located correctly")
ax.scatter([-97], [38], s=500, color=RED, alpha=0.85, label="country known, nothing else")
ax.annotate("38.0000, -97.0000", xy=(-97, 38), xytext=(-93, 44.5), color=RED,
            arrowprops={"arrowstyle": "->", "color": RED})
ax.set_xlabel("longitude")
ax.set_ylabel("latitude")
leg = ax.legend(frameon=False, loc="lower left", fontsize=10)
for handle in leg.legend_handles:
    handle.set_sizes([40])
save(fig, "l02_kansas.pdf")

# Wrong data type: months stored as text are sorted alphabetically (synthetic).
months = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
signups = pd.Series(np.round(100 + 60 * np.sin((np.arange(12) - 2) * np.pi / 6)
                             + rng.normal(0, 5, 12)), index=months)
fig, axes = plt.subplots(1, 2, figsize=(8.5, 2.9), sharey=True)
alpha = signups.sort_index()                                  # "Apr", "Aug", "Dec", ...
axes[0].plot(alpha.index, alpha.values, color=RED, marker="o")
axes[0].set_title("month stored as text")
axes[1].plot(signups.index, signups.values, color=BLUE, marker="o")
axes[1].set_title("month stored as a date")
axes[0].set_ylabel("course sign-ups")
for ax in axes:
    ax.tick_params(axis="x", labelrotation=90)
save(fig, "l02_datatype.pdf")

# Definition mismatch: some respondents answered per day, not per week.
fig, axes = plt.subplots(1, 2, figsize=(8.5, 2.9), sharey=True)
axes[0].hist(raw["study_hours"], bins=30, color=RED)
axes[0].set_title("as collected")
axes[1].hist(df["study_hours"], bins=30, color=BLUE)
axes[1].set_title("everybody per week")
axes[0].set_ylabel("students")
for ax in axes:
    ax.set_xlabel("study hours")
save(fig, "l02_definition.pdf")

# Visual channels: the same five numbers as angles and as lengths.
values = [21, 19, 20, 22, 18]
labels = ["A", "B", "C", "D", "E"]
colors = [BLUE, RED, ORANGE, GREEN, GREY]
fig, axes = plt.subplots(1, 2, figsize=(8.0, 3.0))
axes[0].pie(values, labels=labels, colors=colors, startangle=90,
            wedgeprops={"edgecolor": "white"})
axes[0].set_title("angle and area")
axes[1].bar(labels, values, color=colors)
axes[1].set_title("position and length")
save(fig, "l02_channels.pdf")

# Colormaps: rainbow (jet) invents boundaries; viridis does not.
u = np.linspace(-3, 3, 300)
X, Y = np.meshgrid(u, u)
Z = np.exp(-(X**2 + Y**2) / 3)
fig, axes = plt.subplots(1, 4, figsize=(11.5, 2.8))
gray = lambda cmap: plt.get_cmap(cmap)(Z)[..., :3] @ [0.299, 0.587, 0.114]
for ax, img, title, cm in [
    (axes[0], Z, "jet", "jet"),
    (axes[1], gray("jet"), "jet, printed in gray", "gray"),
    (axes[2], Z, "viridis", "viridis"),
    (axes[3], gray("viridis"), "viridis, printed in gray", "gray"),
]:
    ax.imshow(img, cmap=cm)
    ax.set_title(title)
    ax.set_xticks([])
    ax.set_yticks([])
save(fig, "l02_colormaps.pdf")

# Three kinds of colormaps.
fig, axes = plt.subplots(3, 1, figsize=(6.0, 2.2))
bar = np.linspace(0, 1, 256)[None, :]
for ax, (cm, title) in zip(axes, [("viridis", "sequential: low to high"),
                                  ("RdBu", "diverging: below and above a midpoint"),
                                  ("tab10", "qualitative: categories without order")]):
    ax.imshow(bar if cm != "tab10" else np.arange(10)[None, :], cmap=cm, aspect="auto")
    ax.set_axis_off()
    ax.set_title(title, fontsize=10, loc="left", pad=3)
fig.subplots_adjust(hspace=0.9)
save(fig, "l02_cmap_types.pdf")

# ------------------------------------------- session 2: plotting in Python --

# Anatomy of a figure.
fig, axes = plt.subplots(1, 2, figsize=(8.0, 3.0))
for ax in axes:
    ax.set_xticks([])
    ax.set_yticks([])
axes[0].plot([0, 1, 2, 3], [1, 3, 2, 4], color=BLUE, lw=2, label="a line")
axes[0].set_title("title", color=RED)
axes[0].set_xlabel("x label", color=RED)
axes[0].set_ylabel("y label", color=RED)
axes[0].legend(frameon=False, loc="upper left")
axes[1].scatter([1, 2, 3, 4], [2, 1, 4, 3], color=BLUE)
axes[1].set_title("another axes")
fig.patch.set_edgecolor(RED)
fig.patch.set_linewidth(2)
fig.suptitle("one figure, two axes: fig, axes = plt.subplots(1, 2)", color=RED, y=1.06)
save(fig, "l02_anatomy.pdf")

# Line plot: weekly study hours during the course.
weeks = np.arange(1, 8)
econ = np.array([9, 10, 10, 11, 12, 14, 18])
psych = np.array([8, 8, 9, 9, 11, 13, 17])
fig, ax = plt.subplots(figsize=(5.2, 3.0))
ax.plot(weeks, econ, marker="o", color=BLUE, label="Economics")
ax.plot(weeks, psych, marker="s", color=RED, label="Psychology")
ax.set_xlabel("week of the course")
ax.set_ylabel("study hours per week")
ax.legend(frameon=False)
save(fig, "l02_line.pdf")

# Scatter plot: study hours and grade.
fig, ax = plt.subplots(figsize=(5.2, 3.0))
ax.scatter(df["study_hours"], df["grade"], color=BLUE, alpha=0.6)
ax.set_xlabel("study hours per week")
ax.set_ylabel("grade")
save(fig, "l02_scatter.pdf")

# Histograms: the number of bins changes the picture.
fig, axes = plt.subplots(1, 3, figsize=(9.5, 2.7), sharey=False)
for ax, bins in zip(axes, [5, 20, 100]):
    ax.hist(df["sleep_hours"], bins=bins, color=BLUE)
    ax.set_title(f"bins={bins}")
    ax.set_xlabel("sleep hours per night")
axes[0].set_ylabel("students")
save(fig, "l02_hist.pdf")

# Bar chart: average grade per program, sorted and horizontal.
means = df.groupby("program")["grade"].mean().sort_values()
fig, ax = plt.subplots(figsize=(5.2, 3.0))
ax.barh(means.index, means.values, color=BLUE)
ax.set_xlabel("average grade")
save(fig, "l02_bar.pdf")

# Heatmap: correlations between the numeric columns.
corr = df[["study_hours", "social_media_hours", "sleep_hours", "grade"]].corr()
fig, ax = plt.subplots(figsize=(5.0, 3.6))
im = ax.imshow(corr, cmap="RdBu", vmin=-1, vmax=1)
ax.set_xticks(range(4), ["study", "social media", "sleep", "grade"], rotation=30)
ax.set_yticks(range(4), ["study", "social media", "sleep", "grade"])
for i in range(4):
    for j in range(4):
        r = corr.iloc[i, j]
        ax.text(j, i, f"{r:.2f}", ha="center", va="center", fontsize=10,
                color="white" if abs(r) > 0.5 else "black")
fig.colorbar(im)
save(fig, "l02_heatmap.pdf")

# From default to readable.
fig, axes = plt.subplots(1, 2, figsize=(9.0, 3.0))
plt.rcdefaults()
axes[0].plot(df["social_media_hours"], df["grade"], "o")
plt.rcParams.update({"font.family": "sans-serif", "font.size": 12,
                     "axes.spines.top": False, "axes.spines.right": False})
axes[0].set_title("default")
axes[1].scatter(df["social_media_hours"], df["grade"], color=BLUE, alpha=0.5, s=20)
axes[1].set_xlabel("social media (hours per day)")
axes[1].set_ylabel("grade (1-10)")
axes[1].set_title("More social media, lower grades")
for side in ["top", "right"]:
    axes[1].spines[side].set_visible(False)
save(fig, "l02_readable.pdf")

# seaborn: one line, with colors and a legend for free.
fig, ax = plt.subplots(figsize=(6.0, 3.2))
sns.scatterplot(data=df, x="study_hours", y="grade", hue="passed",
                palette={True: BLUE, False: RED}, ax=ax)
ax.set_xlabel("study hours per week")
save(fig, "l02_seaborn.pdf")

# Data quality: missing values and an outlier in the raw survey.
fig, axes = plt.subplots(1, 2, figsize=(9.0, 3.0))
missing = raw.isna().sum()
missing = missing[missing > 0].sort_values()
axes[0].barh(missing.index, missing.values, color=RED)
axes[0].set_title("missing values per column")
axes[1].boxplot(raw["sleep_hours"].dropna(), orientation="horizontal")
axes[1].set_yticks([])
axes[1].set_xlabel("sleep hours per night")
axes[1].set_title("one student sleeps 70 hours?")
save(fig, "l02_quality.pdf")

print(f"Clean survey: {df.shape}, raw survey: {raw.shape}, duplicates in raw: "
      f"{raw.duplicated().sum()}")
