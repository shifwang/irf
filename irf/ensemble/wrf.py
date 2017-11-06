from .forest import RandomForestClassifier
class wrf(RandomForestClassifier):
    def fit(self, X, y, sample_weight=None, feature_weight=None, K = 5):
        for k in range(K):
            if k == 0:
                # Initially feature weights are None
                feature_importances = None
                
                # Update the dictionary of all our RF weights
                all_rf_weights["rf_weight{}".format(k)] = feature_importances
                
                # fit RF feature weights i.e. initially None
                rf = RandomForestClassifier(n_estimators=n_estimators)
                
                # fit the classifier
                rf.fit(X=X,
                       y=y_train,
                       feature_weight=all_rf_weights["rf_weight{}".format(k)])
                
                # Update feature weights using the
                # new feature importance score
                feature_importances = rf.feature_importances_
                    
                # Load the weights for the next iteration
                all_rf_weights["rf_weight{}".format(k + 1)] = feature_importances

            else:
                # fit weighted RF
                # fit the classifier
                super.fit(
                        X=X_train,
                        y=y_train,
                        feature_weight=feature_importances])
                
                # Update feature weights using the
                # new feature importance score
                feature_importances = self.feature_importances_
                
