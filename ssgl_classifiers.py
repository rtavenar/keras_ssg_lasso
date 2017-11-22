from keras.models import Sequential
from keras.layers import Dense
from keras.regularizers import Regularizer
from keras import backend as K
from keras.metrics import categorical_accuracy
import numpy
import tensorflow as tf

__author__ = 'Romain Tavenard romain.tavenard[at]univ-rennes2.fr'


class SSGL_LogisticRegression:
    """Semi-Sparse Group Lasso Logistic Regression classifier.

    The loss function to minimize is:

    :math:`L(X, y, \\beta) + (1 - \\alpha)\\lambda\\sum_{l=1}^m \\sqrt{p_l}\\|\\beta^l\\|_2 + \\alpha \\lambda \\|\\beta\\|_1`

    where :math:`L` is the logistic loss and :math:`p_l` is the number of variables in group :math:`l`.

    Parameters
    ----------
    dim_input : int
        Dimension of the input feature space.
    n_classes : int
        Number of classes for the classification problem.
    groups : list of numpy arrays
        Affiliation of input dimensions to groups. numpy array of shape `(dim_input, )`. Each group is defined by an integer,
        each input dimension is attributed to a group.
    indices_sparse : array-like
        numpy array of shape `(dim_input, )` in which a zero value means the corresponding input dimension should not
        be included in the per-dimension sparsity penalty and a one value means the corresponding input dimension should
        be included in the per-dimension sparsity penalty.
    alpha : float in the range [0, 1], default 0.5
        Relative importance of per-dimension sparsity with respect to group sparsity (parameter :math:`\\alpha` in the
        optimization problem above).
    lbda : float, default 0.01
        Regularization parameter (parameter :math:`\\lambda` in the optimization problem above).
    n_iter : int, default 500
        Number of training epochs for the gradient descent.
    batch_size : int, default 256
        Size of batches to be used during both training and test.
    optimizer : Keras Optimizer, default "sgd"
        Optimizer to be used at training time. See https://keras.io/optimizers/ for more details.
    verbose : int, default 0
        Verbose level to be used for keras model (0: silent, 1: verbose).

    Attributes
    ----------
    weights_ : numpy.ndarray of shape `(dim_input, n_classes)`
        Logistic Regression weights.
    biases_ : numpy.ndarray of shape `(n_classes, )`
        Logistic Regression biases.
    """
    def __init__(self, dim_input, n_classes, groups, indices_sparse, alpha=0.5, lbda=0.01, n_iter=500, batch_size=256,
                 optimizer="sgd", verbose=0):
        self.d = dim_input
        self.n_classes = n_classes
        self.groups = groups
        self.indices_sparse = indices_sparse
        self.n_iter = n_iter
        self.batch_size = batch_size
        self.alpha = alpha
        self.lbda = lbda
        self.optimizer = optimizer
        self.verbose = verbose

        self.model = None
        self.regularizer = None
        self._init_model()

    def __str__(self):
        return self.model.summary()

    @property
    def weights_(self):
        return self.model.get_weights()[0]

    @property
    def biases_(self):
        return self.model.get_weights()[1]

    def _init_model(self):
        self.regularizer = SSGL_WeightRegularizer(l1_reg=self.alpha * self.lbda, indices_sparse=self.indices_sparse,
                                                  l2_reg=(1. - self.alpha) * self.lbda, groups=self.groups)
        self.model = Sequential()
        self.model.add(Dense(units=self.n_classes, input_dim=self.d, activation="softmax",
                             kernel_regularizer=self.regularizer))
        self.model.compile(loss="categorical_crossentropy", optimizer=self.optimizer, metrics=[categorical_accuracy])

    def fit(self, X, y):
        """Learn Logistic Regression weights.

        Parameters
        ----------
        X : array-like, shape=(n_samples, dim_input)
            Training samples.
        y : array-like, shape=(n_samples, n_classes)
            Training labels (formatted as a binary matrix, as returned by a standard One Hot Encoder, see
            http://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.OneHotEncoder.html for more details).
        """
        assert y.shape[1] == self.n_classes and y.shape[0] == X.shape[0]
        self.model.fit(X, y, epochs=self.n_iter, batch_size=self.batch_size, verbose=self.verbose)
        return self

    def fit_predict(self, X, y):
        """Fit the model using X and y and then use the fitted model to predict X.

        Utility function equivalent to calling fit and then predict on the same data.

        Parameters
        ----------
        X : array-like, shape=(n_samples, dim_input)
            Training samples.
        y : array-like, shape=(n_samples, n_classes)
            Training labels (formatted as a binary matrix, as returned by a standard One Hot Encoder, see
            http://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.OneHotEncoder.html for more details).
        Returns
        -------
        labels : array, shape=(n_samples,)
            Array of class indices.
        """
        return self.fit(X, y).predict(X)

    def predict_probas(self, X):
        """Predict the probability of each class for samples in X.

        Parameters
        ----------
        X : array-like, shape=(n_samples, dim_input)
            Samples to predict.
        Returns
        -------
        probas : array, shape=(n_samples, n_classes)
            Array of class probabilities.
        """
        return self.model.predict(X, batch_size=self.batch_size, verbose=self.verbose)

    def predict(self, X):
        """Predict the class of samples in X.

        Parameters
        ----------
        X : array-like, shape=(n_samples, dim_input)
            Samples to predict.
        Returns
        -------
        labels : array, shape=(n_samples,)
            Array of class indices.
        """
        probas = self.predict_probas(X)
        return numpy.argmax(probas, axis=1)

    def evaluate(self, X, y):
        return self.model.evaluate(X, y, verbose=self.verbose)


class SSGL_MultiLayerPerceptron(SSGL_LogisticRegression):
    """Semi-Sparse Group Lasso Multi Layer Perceptron classifier.

    Parameters
    ----------
    dim_input : int
        Dimension of the input feature space.
    n_classes : int
        Number of classes for the classification problem.
    hidden_layers : tuple (or list) of ints
        Number of neurons in the hidden layers.
    groups : list of numpy arrays
        List of groups. Each group is defined by a numpy array of shape `(dim_input, )` in which a zero value means
        the corresponding input dimension is not included in the group and a one value means the corresponding input
        dimension is part of the group.
    indices_sparse : array-like
        numpy array of shape `(dim_input, )` in which a zero value means the corresponding input dimension should not
        be included in the per-dimension sparsity penalty and a one value means the corresponding input dimension should
        be included in the per-dimension sparsity penalty.
    alpha : float in the range [0, 1], default 0.5
        Relative importance of per-dimension sparsity with respect to group sparsity (parameter :math:`\\alpha` in the
        optimization problem above).
    lbda : float, default 0.01
        Regularization parameter (parameter :math:`\\lambda` in the optimization problem above).
    n_iter : int, default 500
        Number of training epochs for the gradient descent.
    batch_size : int, default 256
        Size of batches to be used during both training and test.
    optimizer : Keras Optimizer, default "sgd"
        Optimizer to be used at training time. See https://keras.io/optimizers/ for more details.
    activation : Keras Activation function, default "relu"
        Activation function to be used for hidden layers. See https://keras.io/activations/ for more details.
    verbose : int, default 0
        Verbose level to be used for keras model (0: silent, 1: verbose).

    Attributes
    ----------
    weights_ : list of arrays
        Multi Layer Perceptron weights.
    biases_ : list of arrays
        Multi Layer Perceptron biases.
    """
    def __init__(self, dim_input, n_classes, hidden_layers, groups, indices_sparse, alpha=0.5, lbda=0.01, n_iter=500,
                 batch_size=256, optimizer="sgd", activation="relu", verbose=0):
        self.hidden_layers = list(hidden_layers)
        self.activation = activation
        if len(self.hidden_layers) == 0:
            raise ValueError("No hidden layer given, you should use SSGL_LogisticRegression class instead")
        SSGL_LogisticRegression.__init__(self, dim_input=dim_input, n_classes=n_classes, groups=groups,
                                         indices_sparse=indices_sparse, alpha=alpha, lbda=lbda, n_iter=n_iter,
                                         batch_size=batch_size, optimizer=optimizer, verbose=verbose)

    @property
    def weights_(self):
        return self.model.get_weights()[::2]

    @property
    def biases_(self):
        return self.model.get_weights()[1::2]

    def _init_model(self):
        self.regularizer = SSGL_WeightRegularizer(l1_reg=self.alpha * self.lbda, indices_sparse=self.indices_sparse,
                                                  l2_reg=(1. - self.alpha) * self.lbda, groups=self.groups)
        self.model = Sequential()
        self.model.add(Dense(units=self.hidden_layers[0], input_dim=self.d, activation=self.activation,
                             kernel_regularizer=self.regularizer))
        for n_units in self.hidden_layers[1:]:
            self.model.add(Dense(units=n_units, activation=self.activation))
        self.model.add(Dense(units=self.n_classes, activation="softmax"))
        self.model.compile(loss="categorical_crossentropy", optimizer=self.optimizer, metrics=[categorical_accuracy])

    def evaluate(self, X, y):
        return SSGL_LogisticRegression.evaluate(self, X, y)


class SSGL_WeightRegularizer(Regularizer):
    """Semi-Sparse Group Lasso weight regularizer.

    Parameters
    ----------
    l1_reg : float, default 0.
        Per-dimension sparsity penalty parameter.
    l2_reg : float, default 0.
        Group sparsity penalty parameter.
    groups : list of numpy arrays or None, default None.
        List of groups. Each group is defined by a numpy array of shape `(dim_input, )` in which a zero value means
        the corresponding input dimension is not included in the group and a one value means the corresponding input
        dimension is part of the group. None means no group sparsity penalty.
    indices_sparse : array-like or None, default None.
        numpy array of shape `(dim_input, )` in which a zero value means the corresponding input dimension should not
        be included in the per-dimension sparsity penalty and a one value means the corresponding input dimension should
        be included in the per-dimension sparsity penalty. None means no per-dimension sparsity penalty.
    """
    def __init__(self, l1_reg=0., l2_reg=0., groups=None, indices_sparse=None):
        self.l1_reg = l1_reg
        self.l2_reg = l2_reg
        if groups is None:
            self.groups = None
        else:
            groups = numpy.array(groups).astype('int32')
            self.groups = K.variable(groups, 'int32')
        if indices_sparse is not None:
            self.indices_sparse = K.variable(indices_sparse.reshape((1, -1)))

    def __call__(self, x):
        loss = 0.
        if self.indices_sparse is not None:
            loss += K.sum(K.dot(self.indices_sparse, K.abs(x))) * self.l1_reg
        if self.groups is not None:
            loss += K.sum(tf.segment_sum(K.square(x), self.groups) * self.l2_reg)
        return loss

    def get_config(self):
        return {"name": self.__class__.__name__, "l1_reg": self.l1_reg, "l2_reg": self.l2_reg}
