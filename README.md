# Semi-sparse Group Lasso implementation in `keras`

This repository proposes a `keras` implementation of Semi-sparse Group Lasso models for logistic regression
using the popular `keras` framework.

The file `ex_classif.py` provides an example use of the `SSGL_LogisticRegression` model class.

Basically, one thing that is not great up to now is that the class information should be provided as a 
`(nsamples, n_classes)` matrix of zeros and ones. One could definitely use OneHotEncoder from his favorite
ML library to generate them from a vector of class labels.