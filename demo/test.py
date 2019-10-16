from sklearn.datasets import load_breast_cancer
import numpy as np
from functools import reduce

# Needed for the scikit-learn wrapper function
import irf
from irf import irf_utils
from irf import irf_jupyter_utils
from irf.ensemble.forest import RandomForestClassifier
from math import ceil
import irf.ensemble.wrf as wrf

# Import our custom utilities
from imp import reload
from sklearn.model_selection import train_test_split

from pyspark.ml.fpm import FPGrowth

print("Everything loaded properly")

data = load_breast_cancer()
X = data['data']
y = data['target']
X_train, X_test, y_train, y_test = train_test_split(
        data.data, data.target, train_size=0.9)

print("Data was split properly")

rf = wrf(n_estimators=20, max_depth=5)

print("wrf was successfully initialized")

rf.fit(X_train, y_train)
print(rf.feature_importances_)