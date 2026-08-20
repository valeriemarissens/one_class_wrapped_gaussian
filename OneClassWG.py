import numpy as np

from pyriemann.utils.tangentspace import log_map_riemann, unupper, upper
from pyriemann.utils.base import invsqrtm, logm
from pyriemann.utils.mean import mean_riemann
from sklearn.base import BaseEstimator, ClassifierMixin


def estim_param(samples, cov_diag=False):
    """Estimate the parameters of a wrapped Gaussian distribution from samples.

    Parameters
    ----------
    samples : array, shape (n_matrices, n_channels, n_channels)
        The input SPD matrices.

    Returns
    -------
    X_bar : array, shape (n_channels, n_channels)
        The estimated mean SPD matrix.
    Sigma : array, shape (n_ts, n_ts)
        The estimated covariance matrix in the tangent space.
    """
    n_matrices, n_channels, _ = samples.shape
    n_ts = n_channels * (n_channels + 1) // 2

    # Estimate the mean using the Riemannian mean
    X_bar = mean_riemann(samples)

    # Map samples to the tangent space at the mean
    samples_ts = log_map_riemann(samples, X_bar, C12=True)

    # Vectorize the tangent space matrices
    X_bar_inv_sqrt = invsqrtm(X_bar)
    samples_ts_vec = upper(X_bar_inv_sqrt @ samples_ts @ X_bar_inv_sqrt)

    # Estimate the covariance matrix in the tangent space
    if cov_diag:
        Sigma = np.diag(np.var(samples_ts_vec, axis=0))
    else:
        Sigma = np.cov(samples_ts_vec, rowvar=False)

    return X_bar, Sigma


class OneClassWrappedGaussian(BaseEstimator, ClassifierMixin):
    """Classifier based on Wrapped Gaussian distributions for each class.

    Parameters
    ----------
    None

    Attributes
    ----------
    class_params_ : dict
        Dictionary storing the parameters (mean and covariance) for each class.
    """

    def __init__(self, eps=0.95, cov_diag=False):
        super().__init__()
        self.X_bar_est = None
        self.Sigma_est = None
        self.eps = eps
        self.cov_diag = cov_diag

    def fit(self, samples):
        """Fit the parameters of the Wrapped Gaussian for the class "awake"

        Parameters
        ----------
        samples : array, shape (n_samples, n_channels, n_channels)
            The input SPD matrices of class "awake".

        Returns
        -------
        self : object
            Returns the instance itself.
        """

        self.X_bar_est, self.Sigma_est = estim_param(samples, self.cov_diag)
        self.likelihoods = self.compute_log_likelihood(
            samples, self.X_bar_est, self.Sigma_est
        )
        self.threshold = np.percentile(self.likelihoods, (1 - self.eps) * 100)
        return self

    def compute_log_likelihood(self, samples, X_bar, Sigma):
        """Compute the log-likelihood of the observations X according to a Wrapped Gaussian WG(X_bar, Sigma).

        Args:
            samples (n, dim, dim): A set of SPD matrices of size dim x dim
            X_bar (dim, dim): The parameter X_bar of the Wrapped Gaussian (the mean)
            Sigma (dim*(dim+1)/2, dim*(dim+1)/2): The parameter Sigma of the Wrapped Gaussian (the covariance matrix in the tangent space)

        Returns:
            n, : The log-likelihood of each observation
        """
        n_dim = samples.shape[1]
        n_ts = n_dim * (n_dim + 1) // 2

        # Map samples to the tangent space at the mean
        samples_ts = log_map_riemann(samples, X_bar, C12=True)

        # Vectorize the tangent space matrices
        X_bar_inv_sqrt = invsqrtm(X_bar)
        samples_ts_vec = upper(X_bar_inv_sqrt @ samples_ts @ X_bar_inv_sqrt)

        # Log-density of the multivariate normal (computed entirely in log-space)
        mahalanobis = np.sum(
            np.linalg.solve(Sigma, samples_ts_vec.T).T * samples_ts_vec,
            axis=1,
        )
        # if self.full:
        _, log_det_Sigma = np.linalg.slogdet(Sigma)
        log_h = -0.5 * (n_ts * np.log(2.0 * np.pi) + log_det_Sigma + mahalanobis)
        # else:
        #    log_h = -0.5 * mahalanobis

        # Log-Jacobian of the exponential map
        # eigh is more stable than eig for symmetric matrices
        eigenvalues = np.linalg.eigh(logm(X_bar_inv_sqrt @ samples @ X_bar_inv_sqrt))[0]

        # Pairwise eigenvalue differences (i < j)
        diff_matrix = eigenvalues[:, :, np.newaxis] - eigenvalues[:, np.newaxis, :]
        triu_indices = np.triu_indices(n_dim, k=1)
        diff_triu = diff_matrix[:, triu_indices[0], triu_indices[1]]

        # Compute log(|sinh(d/2) / d|) with three regimes for stability:
        #   |d| < 1e-7  : use the limit value log(1/2)
        #   |d| >= 700  : use asymptotic form |d|/2 - log(2) - log(|d|)
        #   otherwise   : direct computation
        half_diff = diff_triu / 2
        abs_half_diff = np.abs(half_diff)
        abs_diff = np.abs(diff_triu)

        small = abs_diff < 1e-7
        large = abs_half_diff > 350
        mid = ~small & ~large

        log_sinh_term = np.empty_like(diff_triu)
        log_sinh_term[small] = np.log(0.5)
        log_sinh_term[mid] = np.log(np.sinh(abs_half_diff[mid])) - np.log(abs_diff[mid])
        log_sinh_term[large] = (
            abs_half_diff[large] - np.log(2.0) - np.log(abs_diff[large])
        )

        log_J = np.sum(log_sinh_term, axis=1) + n_dim * (n_dim - 1) / 2 * np.log(2.0)

        return log_h - log_J

    def predict(self, X, full=True):
        """Predict method is not implemented.

        Parameters
        ----------
        X : array, shape (n_samples, n_channels, n_channels)
            The input SPD matrices.

        Returns
        -------
        y : array
            Predicted class labels (0 if class "asleep" or 1 if class "awake").
        """
        log_likelihoods = self.compute_log_likelihood(X, self.X_bar_est, self.Sigma_est)

        # +1 inliers (high log_likelihood), -1 outliers (low log_likelihood)
        labels = np.where(log_likelihoods >= self.threshold, 1, -1)
        return labels, log_likelihoods
