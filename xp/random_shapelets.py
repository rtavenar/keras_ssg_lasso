import numpy
from tslearn.datasets import UCR_UEA_datasets
from tslearn.utils import save_timeseries_txt

__author__ = 'Romain Tavenard romain.tavenard[at]univ-rennes2.fr'

numpy.random.seed(0)
shapelet_sizes = {20: 100}
dataset_name = "Trace"

X_train, y_train, X_test, y_test = UCR_UEA_datasets().load_dataset(dataset_name)
n_ts, sz, d = X_train.shape

shapelets = []
for shp_sz, n_shp in shapelet_sizes.items():
    indices_ts = numpy.random.randint(low=0, high=n_ts, size=n_shp)
    timestamps_ts = numpy.random.randint(low=0, high=sz - shp_sz + 1, size=n_shp)
    for i in range(n_shp):
        idx = indices_ts[i]
        t = timestamps_ts[i]
        shapelets.append(X_train[idx, t:t+shp_sz])

save_timeseries_txt("shapelets/%s_%s.txt" % (dataset_name, str(shapelet_sizes)), shapelets)
