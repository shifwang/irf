import irf
from sklearn.datasets import load_boston, load_breast_cancer
import irf.ensemble.wrf as wrf
from irf.ensemble import RandomForestClassifier
from irf.ensemble.wrf import wrf_reg
import numpy as np

from sklearn.model_selection import train_test_split

from irf import irf_utils
from irf import irf_jupyter_utils

data = load_breast_cancer()
X = data['data']
y = data['target']

X_train, X_test, y_train, y_test = train_test_split(
        data.data, data.target, train_size=0.9)

a, b, c, d, e = irf_utils.run_iRF(X_train=X_train,
                                    X_test=X_test,
                                    y_train=y_train,
                                    y_test=y_test,
                                    K=5,
                                    rf=RandomForestClassifier(n_estimators=40),
                                    signed=True,
                                    weighted_by_length=False)
print(e)