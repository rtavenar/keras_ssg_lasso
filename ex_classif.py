import numpy

from ssgl_classifiers import SSGL_LogisticRegression, SSGL_MultiLayerPerceptron

__author__ = 'Romain Tavenard romain.tavenard[at]univ-rennes2.fr'

n = 1000
d = 20
groups = [0] * (d // 2) + [1] * (d // 2)
alpha = 0.5
lbda = .1

X = numpy.random.randn(n, d)
secret_beta = numpy.random.randn(d)
ind_sparse = numpy.zeros((d, ), dtype=numpy.int32)
for i in range(d):
    if groups[i] == 1 or i % 2 == 0:
        secret_beta[i] = 0
    if i % 2 == 0:
        ind_sparse[i] = 1

y = numpy.ones((n, 2), dtype=numpy.int32)
y[numpy.exp(numpy.dot(X, secret_beta)) < 1., 0] = 0
y[:, 1] = 1 - y[:, 0]

model = SSGL_LogisticRegression(dim_input=d, n_classes=2, groups=groups, indices_sparse=ind_sparse,
                                n_iter=1000, alpha=alpha, lbda=lbda, optimizer="rmsprop", verbose=False)

model.fit(X, y)
beta_hat = model.weights_

for i, (betai_hat, betai) in enumerate(zip(beta_hat, secret_beta)):
    print("Component %02d: %r | %.4f" % (i, numpy.linalg.norm(betai_hat), betai))
print("Correct classification rate of Logistic Regression model: %.3f" % model.evaluate(X, y)[1])

model = SSGL_MultiLayerPerceptron(dim_input=d, n_classes=2, hidden_layers=(10, 5), groups=groups,
                                  indices_sparse=ind_sparse, n_iter=1000, alpha=alpha, lbda=lbda, optimizer="rmsprop",
                                  verbose=False)

model.fit(X, y)
beta_hat = model.weights_[0]

for i, (betai_hat, betai) in enumerate(zip(beta_hat, secret_beta)):
    print("Component %02d: %r | %.4f" % (i, numpy.linalg.norm(betai_hat), betai))
print("Correct classification rate of MLP model: %.3f" % model.evaluate(X, y)[1])

print("Weight shapes:", [w.shape for w in model.weights_])
print("Bias shapes:  ", [b.shape for b in model.biases_])

del model

