
__version__ = '0.1.0'


__all__ = ['MRMR']


import numpy as np
from sklearn.base import BaseEstimator, MetaEstimatorMixin, clone
from sklearn.utils import check_arrays, safe_mask
from ._mrmr import _mrmr


class MRMR(BaseEstimator, MetaEstimatorMixin):

    def __init__(self, estimator, n_features_to_select=None,
            maxrel=False, mutual_info_difference=True, normalize=False,
            estimator_params={}, verbose=0):

        self.estimator = estimator
        self.n_features_to_select = n_features_to_select
        self.maxrel = maxrel
        self.mutual_info_difference = mutual_info_difference
        self.normalize = normalize
        self.verbose = verbose

    def fit(self, X, y):
        X, y = check_arrays(X, y, sparse_format="csr")
        n_samples, n_features = X.shape
        if self.n_features_to_select is None:
            n_features_to_select = n_features // 2
        else:
            n_features_to_select = self.n_features_to_select

        X_classes = np.array(list(set(X.reshape((n_samples * n_features,)))))
        y_classes = np.array(list(set(y.reshape((n_samples,)))))

        idxs, _, _ = _mrmr(n_samples, n_features, X, y,
                X_classes, y_classes, X_classes.shape[0], y_classes.shape[0],
                n_features_to_select,
                self.maxrel, self.mutual_info_difference, self.normalize)

        support_ = np.zeros(n_features, dtype=np.bool)
        support_[idxs] = True

        self.estimator_ = clone(self.estimator)
        self.estimator_.set_params(**self.estimator_params)
        self.estimator_.fit(X[:, support_], y)
        self.n_features_ = support_.sum()
        self.support_ = support_

        return self

    def predict(self, X):
        return self.estimator_.predict(self.transform(X))

    def score(self, X, y):
        return self.estimator_.score(self.transform(X), y)

    def transform(self, X):
        return X[:, safe_mask(X, self.support_)]

    def decision_function(self, X):
        return self.estimator_.decision_function(self.transform(X))

    def predict_proba(self, X):
        return self.estimator_.predict_proba(self.transform(X))
