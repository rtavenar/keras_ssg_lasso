import numpy

from ssgl_classifiers import SSGL_LogisticRegression

__author__ = 'Romain Tavenard romain.tavenard[at]univ-rennes2.fr'

n = 1000
d = 20
groups = numpy.array([0] * int(d / 2) + [1] * (d - int(d / 2)))
alpha = 0.
lbda = .1

X = numpy.random.randn(n, d)
secret_beta = numpy.random.randn(d)
ind_sparse = numpy.zeros((d, ))
for i in range(d):
    if groups[i] == 0 or i % 2 == 0:
        secret_beta[i] = 0
    if i % 2 == 0:
        ind_sparse[i] = 1

y = numpy.ones((n, 2), dtype=numpy.int32)
y[numpy.exp(numpy.dot(X, secret_beta)) < 1., 0] = 0
y[:, 1] = 1 - y[:, 0]

model = SSGL_LogisticRegression(dim_input=d, n_classes=2, groups=groups, sparse_indices=ind_sparse)

model.fit(X, y)
# beta_hat = model.coef_
#
# print(numpy.linalg.norm(secret_beta - beta_hat))
# for i, (betai_hat, betai) in enumerate(zip(beta_hat, secret_beta)):
#     print("Component %02d: %.4f | %.4f" % (i, betai_hat, betai))
print("Correct classification rate: %.3f" % (numpy.sum(model.predict(X) == numpy.argmax(y, axis=1)) / n))