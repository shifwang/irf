from .forest import RandomForestClassifier
class wrf(RandomForestClassifier):
    def fit(self, X, y, sample_weight=None, feature_weight=None, K = 5):
        for k in range(K):
            if k == 0:
                # Initially feature weights are None
                feature_importances = None
                
                # fit the classifier
                super(wrf, self).fit(X=X,
                         y=y,
                         feature_weight=feature_importances)
                
                # Update feature weights using the
                # new feature importance score
                feature_importances = self.feature_importances_

            else:
                # fit weighted RF
                # fit the classifier
                super(wrf, self).fit(
                        X=X,
                        y=y,
                        feature_weight=feature_importances)
                
                # Update feature weights using the
                # new feature importance score
                feature_importances = self.feature_importances_
        return self
                
