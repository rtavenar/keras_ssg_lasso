import numpy
from tslearn.datasets import UCR_UEA_datasets
from tslearn.utils import save_timeseries_txt

__author__ = 'Romain Tavenard romain.tavenard[at]univ-rennes2.fr'

numpy.random.seed(0)
shapelet_size_ratio = 0.1
n_shapelets = 10 * 1000

for dataset_name in UCR_UEA_datasets().list_datasets():
    X_train, y_train, X_test, y_test = UCR_UEA_datasets().load_dataset(dataset_name)
    if X_train is None:
        print("Skipping dataset %s: invalid files" % dataset_name)
        continue
    n_ts, sz, d = X_train.shape
    shapelet_size = int(shapelet_size_ratio * sz)

    shapelets = []
    indices_ts = numpy.random.randint(low=0, high=n_ts, size=n_shapelets)
    timestamps_ts = numpy.random.randint(low=0, high=sz - shapelet_size + 1, size=n_shapelets)
    for i in range(n_shapelets):
        idx = indices_ts[i]
        t = timestamps_ts[i]
        shapelets.append(X_train[idx, t:t+shapelet_size])

    save_timeseries_txt("shapelets/%s_%s.txt" % (dataset_name, str(shapelet_size_ratio)), shapelets)
    print("Wrote file for dataset %s" % dataset_name)
