
__version__ = '0.1.0'


__all__ = ['MRMR']


import numpy as np
from sklearn.base import BaseEstimator, MetaEstimatorMixin, clone
from sklearn.utils import check_arrays, safe_mask
from ._mrmr import _mrmr, MAXREL, MID, MIQ


class MRMR(BaseEstimator, MetaEstimatorMixin):

    def __init__(self, estimator, n_features_to_select=None,
            method='MID', normalize=False, estimator_params={}, verbose=0):

        method = method.upper()
        if method == 'MAXREL':
            method = MAXREL
        elif method == 'MID':
            method = MID
        elif method == 'MIQ':
            method = MIQ
        else:
            raise ValueError('method must be one of `MAXREL`, `MID`, or `MIQ`')

        self.estimator = estimator
        self.n_features_to_select = n_features_to_select
        self.method = method
        self.normalize = normalize
        self.estimator_params = estimator_params
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

        idxs, _ = _mrmr(n_samples, n_features, y, X,
                y_classes, X_classes, y_classes.shape[0], X_classes.shape[0],
                n_features_to_select, self.method, self.normalize)

        support_ = np.zeros(n_features, dtype=np.bool)
        ranking_ = np.zeros(n_features, dtype=np.int)

        support_[idxs] = True
        ranking_[:] = n_features_to_select + 1
        for i, idx in enumerate(idxs):
            ranking_[idx] -= n_features_to_select - i

        self.estimator_ = clone(self.estimator)
        self.estimator_.set_params(**self.estimator_params)
        self.estimator_.fit(X[:, support_], y)
        self.n_features_ = support_.sum()
        self.support_ = support_
        self.ranking_ = ranking_

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
