from numpy import random
from pyriemann.datasets import simulated
from sklearn.metrics import accuracy_score

from OneClassMDM import OCRMDM
from OneClassWG import OneClassWrappedGaussian

if __name__ == '__main__':
    # Dummy covariance matrices and dummy y_test
    cov_train = simulated.make_matrices(n_matrices=50, n_dim=20, kind='spd')
    cov_test = simulated.make_matrices(n_matrices=150, n_dim=20, kind='spd')
    y_test = random.choice([-1, 1], size=150)

    # OCRWG
    ocwg = OneClassWrappedGaussian(eps=0.95, cov_diag=True)
    ocwg.fit(cov_train)
    ocwg_pred, log_likelihoods = ocwg.predict(cov_test)
    ocwg_test_acc = accuracy_score(y_test, ocwg_pred) * 100

    # OCRMDM
    ocrmdm = OCRMDM()
    ocrmdm.fit(cov_train)
    ocrmdm_pred = ocrmdm.predict(cov_test)
    ocrmdm_test_acc = accuracy_score(y_test, ocrmdm_pred) * 100

    print("One-Class Wrapped Gaussian vs One-Class Riemannian Minimum Distance to the Mean:",
          ocwg_test_acc, "vs", ocrmdm_test_acc)

