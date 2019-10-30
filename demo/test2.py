import matplotlib.pyplot as plt

from sklearn.datasets import load_breast_cancer
import numpy as np
from functools import reduce

# Needed for the scikit-learn wrapper function
import irf
from irf import irf_utils
from irf import irf_jupyter_utils
from irf.ensemble.forest import RandomForestClassifier
from math import ceil

# Import our custom utilities
from imp import reload

load_breast_cancer = load_breast_cancer()

X_train, X_test, y_train, y_test, rf = irf_jupyter_utils.generate_rf_example(n_estimators=20, 
                                                                             feature_weight=None)
