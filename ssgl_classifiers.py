from keras.models import Sequential
from keras.layers import Dense
import numpy

__author__ = 'Romain Tavenard romain.tavenard[at]univ-rennes2.fr'


class SSGL_LogisticRegression:
    def __init__(self, dim_input, n_classes, groups, sparse_indices, activation="relu", n_iter=100, batch_size=32):
        self.d = dim_input
        self.n_classes = n_classes
        self.groups = groups
        self.sparse_indices = sparse_indices
        self.activation = activation
        self.n_iter = n_iter
        self.batch_size = batch_size

        self.model = None
        self._init_model()

    def _init_model(self):
        self.model = Sequential()
        self.model.add(Dense(units=self.n_classes, input_dim=self.d, activation='softmax'))
        if self.n_classes == 2:
            loss_str = "binary_crossentropy"
        else:
            loss_str = "categorical_crossentropy"
        self.model.compile(loss=loss_str, optimizer="sgd")

    def fit(self, X, y):
        self.model.fit(X, y, epochs=self.n_iter, batch_size=self.batch_size, verbose=0)

    def predict_probas(self, X):
        return self.model.predict(X, batch_size=self.batch_size, verbose=0)

    def predict(self, X):
        probas = self.predict_probas(X)
        return numpy.argmax(probas, axis=1)
