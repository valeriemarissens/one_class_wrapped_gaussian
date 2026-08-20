import numpy as np
from pyriemann.classification import MDM
from sklearn.base import BaseEstimator, OutlierMixin


class OCRMDM(BaseEstimator, OutlierMixin):
    r""" Classification by One-Class Riemannian Minimum Distance to the Mean [1].

    Threshold is the Riemann median + 3 times the standard deviation.

    References
    ----------
    .. [1] `Multiclass Brain-Computer Interface Classification by Riemannian
        Geometry
        <https://hal.archives-ouvertes.fr/hal-00681328>`_
        A. Barachant, S. Bonnet, M. Congedo, and C. Jutten. IEEE Transactions
        on Biomedical Engineering, vol. 59, no. 4, p. 920-928, 2012.
    """

    def __init__(self, n_jobs=-1):
        self.mdm_instance = MDM(metric=dict(mean='riemann', distance='riemann'), n_jobs=n_jobs)

    def fit(self, X, y=None):
        """
        Determine the threshold according to the training data.

        Parameters
        ----------
        X : ndarray, shape (n_matrices, n_channels, n_channels)
            Set of SPD matrices.

        y : Ignored
            Not used, present for API consistency by convention.

        Returns
        -------
        self : object
            Fitted estimator.
        """
        self.mdm_instance.fit(X=X, y=np.ones(X.shape[0]))

        dist2mean = self.mdm_instance.transform(X)
        self.threshold = np.median(dist2mean) + 3 * np.std(dist2mean)

        return self

    def predict(self, X):
        """Get the predictions.

        Parameters
        ----------
        X : ndarray, shape (n_matrices, n_channels, n_channels)
            Set of SPD matrices.

        Returns
        -------
        pred : ndarray of int, shape (n_matrices,)
            Predictions for each matrix according to the closest centroid.
        """
        dist2mean = self.mdm_instance.transform(X)
        return [-1 if d > self.threshold else 1 for d in dist2mean]

    def get_distance(self, cov):
        return self.mdm_instance.transform(cov)
