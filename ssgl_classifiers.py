from keras.models import Sequential
from keras.layers import Dense
from keras.regularizers import Regularizer
from keras import backend as K
import numpy

__author__ = 'Romain Tavenard romain.tavenard[at]univ-rennes2.fr'


class SSGL_LogisticRegression:
    def __init__(self, dim_input, n_classes, groups, indices_sparse, alpha=0.5, lbda=0.01, activation="relu",
                 n_iter=500, batch_size=32):
        self.d = dim_input
        self.n_classes = n_classes
        self.groups = groups
        self.indices_sparse = indices_sparse
        self.activation = activation
        self.n_iter = n_iter
        self.batch_size = batch_size
        self.alpha = alpha
        self.lbda = lbda

        self.model = None
        self.regularizer = None
        self._init_model()

    @property
    def weights_(self):
        return self.model.get_weights()[0]

    def _init_model(self):
        self.regularizer = SSGL_WeightRegularizer(l1_reg=self.alpha * self.lbda, indices_sparse=self.indices_sparse,
                                                  l2_reg=(1. - self.alpha) * self.lbda, groups=self.groups)
        self.model = Sequential()
        self.model.add(Dense(units=self.n_classes, input_dim=self.d, activation="softmax",
                             kernel_regularizer=self.regularizer))
        self.model.compile(loss="categorical_crossentropy", optimizer="sgd")

    def fit(self, X, y):
        self.model.fit(X, y, epochs=self.n_iter, batch_size=self.batch_size, verbose=0)

    def predict_probas(self, X):
        return self.model.predict(X, batch_size=self.batch_size, verbose=0)

    def predict(self, X):
        probas = self.predict_probas(X)
        return numpy.argmax(probas, axis=1)


class SSGL_WeightRegularizer(Regularizer):
    def __init__(self, l1_reg=0., l2_reg=0., groups=None, indices_sparse=None):
        self.l1_reg = l1_reg
        self.l2_reg = l2_reg
        if groups is None:
            self.groups = []
        else:
            self.groups = [K.variable(gr.reshape((1, -1))) for gr in groups]
        if indices_sparse is not None:
            self.indices_sparse = K.variable(indices_sparse.reshape((1, -1)))

    def __call__(self, x):
        loss = 0.
        if self.indices_sparse is not None:
            loss += K.sum(K.dot(self.indices_sparse, K.abs(x))) * self.l1_reg
        for gr in self.groups:
            loss += K.sum(K.dot(gr, K.square(x))) * self.l2_reg
        return loss

    def get_config(self):
        return {"name": self.__class__.__name__, "l1_reg": self.l1_reg, "l2_reg": self.l2_reg}