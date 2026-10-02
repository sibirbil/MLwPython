# Please check: Lecture 3 (Introduction to Supervised Learning)

Items in `lecture03.tex` that need a check by the lecturer before the slides are final.

## Numbers on the slides
- [ ] All accuracies come from `codes/l03_figures.py` on the synthetic survey (300 students, 36% passed) with `random_state=0`:
  - 1-NN: training 99.6%, test 79%;
  - 5-NN: test 80% (the `knn.score` line on the code slide);
  - best k on the test set: k = 10, 83%;
  - grid search: best k = 13, cross-validation 82%, test 79%.

  If the survey in `l02_figures.py` changes, rerun both scripts and update these numbers.
- [ ] 1-NN is not exactly 100% on the training data because two students have identical features but different outcomes. The notes say so; the exercise solution says "about 100%".

## scikit-learn facts
- [ ] The defaults on the cross-validation slide are from memory of the current scikit-learn API, not checked against the documentation in this session:
  - `cv=5`;
  - stratified folds for classifiers in `cross_val_score` and `GridSearchCV`;
  - `train_test_split` shuffles by default and keeps 25% for testing.

## Content choices
- [ ] Dropped from last year: nearest centroid and nearest shrunken centroid, parametric vs. non-parametric, computational cost (big-O, kd-trees), nested cross-validation, leave-one-out, repeated k-fold and `ShuffleSplit`.
- [ ] Distance is shown only as a picture; the slides and notes contain no formula.
- [ ] The units frame ("Distance depends on the units") is new. It anticipates scaling in Lecture 6.

## Pacing
- [ ] Session 1 has 10 content frames, after nearest centroid was dropped. If it runs short, the units frame and the regression frame are the natural places to spend more time.
