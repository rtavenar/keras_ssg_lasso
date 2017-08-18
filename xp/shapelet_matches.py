import numpy
from tslearn.datasets import UCR_UEA_datasets
from tslearn.utils import load_timeseries_txt

__author__ = 'Romain Tavenard romain.tavenard[at]univ-rennes2.fr'


def extract_shapelet_match(ts, shapelets):
    ts = ts.reshape((-1, ))
    n_shapelets = len(shapelets)
    shp_sz = shapelets[0].shape[0]
    elem_size = ts.strides[0]
    Xi_reshaped = numpy.lib.stride_tricks.as_strided(ts, strides=(elem_size, elem_size),
                                                     shape=(ts.shape[0] - shp_sz + 1, shp_sz))
    feature = numpy.zeros((2 * n_shapelets, ))
    for k in range(n_shapelets):
        shp = shapelets[k].reshape((-1, ))
        distances = numpy.linalg.norm(Xi_reshaped - shp, axis=1) ** 2
        idx = numpy.argmin(distances)
        feature[2 * k] = distances[idx]
        feature[2 * k + 1] = float(idx) / ts.shape[0]
    return feature


def transform_dataset(time_series, shapelets):
    n_ts, sz, d = time_series.shape
    n_shapelets = len(shapelets)
    X = numpy.zeros((n_ts, 2 * n_shapelets))
    for i in range(n_ts):
        X[i] = extract_shapelet_match(time_series[i], shapelets)
    return X


shapelet_size_ratio = 0.1

for dataset_name in UCR_UEA_datasets().list_datasets():
    X_train, y_train, X_test, y_test = UCR_UEA_datasets().load_dataset(dataset_name)
    if X_train is None or X_test is None:
        print("Skipping dataset %s: invalid files" % dataset_name)
        continue
    n_ts, sz, d = X_train.shape
    shapelets = load_timeseries_txt("shapelets/%s_%s.txt" % (dataset_name, str(shapelet_size_ratio)))
    numpy.savetxt("features/%s_%s_TRAIN.txt" % (dataset_name, str(shapelet_size_ratio)),
                  transform_dataset(X_train, shapelets))
    numpy.savetxt("features/%s_%s_TEST.txt" % (dataset_name, str(shapelet_size_ratio)),
                  transform_dataset(X_test, shapelets))
    print("Wrote files for dataset %s" % dataset_name)

