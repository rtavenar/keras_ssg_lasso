[![Documentation Status](https://readthedocs.org/projects/keras_ssg_lasso/badge/?version=latest)](http://keras_ssg_lasso.readthedocs.io/en/latest/?badge=latest)

# Semi-sparse Group Lasso implementation in `keras`

This repository proposes a `keras` implementation of Semi-sparse Group Lasso models for logistic regression
using the popular `keras` framework.

The file `ex_classif.py` provides an example use of the `SSGL_LogisticRegression` model class.

Directory `xp/` contains code for personal experiments that relies on `tslearn`, but the core of the code only depends 
on `numpy` and `keras`.
