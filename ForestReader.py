from abc import ABC, abstractmethod
from irf import irf_utils
import sklearn
import numpy as np
import pandas
class Reader(ABC):
    """docstring for Reader."""
    def __init__(self, options):
        super(Reader, self).__init__()
        self.options = options

    @abstractmethod
    def read_from(self, obj):
        pass
    @abstractmethod
    def summary(self):
        pass
    @abstractmethod
    def reset(self):
        pass


class TreeReader(Reader):
    """docstring for TreeReader."""
    def __init__(self,options):
        super(TreeReader, self).__init__(options)

        assert 'feature_names' in options, 'options must specific feature_names!'
        assert 'sample_names' in options, 'options must specific sample_names!'

        self.colnames = []

        self.feature_names = options['feature_names']
        self.number_of_features = len(self.feature_names)
        self.colnames += self.feature_names

        self.sample_names = options['sample_names']
        self.number_of_samples = len(self.sample_names)
        self.colnames += self.sample_names

        self.colnames.append('tree_id')
        self.colnames.append('leaf_id')

        self.colnames.append('pred_label')
        self.info = pandas.DataFrame(columns = self.colnames)

        self.new_tree_id = 0
        self.number_of_rows = self.info.shape[0]
    def read_from(self, tree, X):
        # tree must be of the type:
        assert type(tree) == sklearn.tree.tree.DecisionTreeClassifier,\
            'The type of tree must be sklearn.tree.tree.DecisionTreeClassifier but %s given.'%str(type(tree))

        # X must have the correct shape
        assert X.shape == (self.number_of_samples, self.number_of_features),\
            'The shape of X is not (%d, %d)'%(self.number_of_features, self.number_of_samples)

        # read all paths from tree
        paths = irf_utils.all_tree_paths(tree)

        # get prediction nodes
        pred_nodes = tree.tree_.apply(np.array(X, dtype = np.float32))

        # get prediction labels
        pred_labels = tree.predict(X)

        new_record = {colname:[] for colname in list(self.info)}
        # add all paths into info
        for path in paths:

            new_record['tree_id'].append(self.new_tree_id)
            new_record['leaf_id'].append(path[-1])
            new_record['pred_label'].append(None)
            for f in self.feature_names:
                new_record[f].append(False)
            for s in self.sample_names:
                new_record[s].append(False)
            for node_ind in path[:-1]:
                new_record[self.feature_names[tree.tree_.feature[node_ind]]][-1] = True
            for sample_ind in range(self.number_of_samples):
                if pred_nodes[sample_ind] == path[-1]:
                    new_record[self.sample_names[sample_ind]][-1] = True
                    new_record['pred_label'][-1] = pred_labels[sample_ind]
            #print(new_record)
        self.info = self.info.append(pandas.DataFrame(new_record), ignore_index = True)
        self.number_of_rows = self.info.shape[0]

        # update new_tree_id
        self.new_tree_id += 1


    def reset(self):
        self.info.drop(self.info.index, inplace = True)
        self.new_tree_id = 0
        self.number_of_rows = 0



    def summary(self):
        print('Here is the summary.')
        print('Number of features is %d'%(self.number_of_features))
        print('Number of samples is %d'%(self.number_of_samples))
        print('Number of paths is %d'%(self.number_of_rows))
        print(self.info)
        print('end.')
class ForestReader(TreeReader):
    """ docstring for ForestReader"""
    def read_from(self, forest, X):
        assert type(forest) == sklearn.ensemble.forest.RandomForestClassifier,\
            'The type of forest must be sklearn.ensemble.forest.RandomForestClassifier but %s given.'%str(type(forest))
        for tree in forest.estimators_:
            super(ForestReader, self).read_from(tree, X)


if __name__ == '__main__':
    options = dict()
    from sklearn.datasets import load_breast_cancer
    from sklearn.model_selection import train_test_split
    from sklearn.ensemble import RandomForestClassifier
    raw_data = load_breast_cancer()
    X_train, X_test, y_train, y_test = train_test_split(
        raw_data.data, raw_data.target, train_size=0.9,
        random_state=2017)
    rf = RandomForestClassifier(
        n_estimators=3, random_state=1231)
    rf.fit(X=X_train, y=y_train)
    e0 = rf.estimators_[0]
    options['feature_names'] = ['f%d'%i for i in range(30)]
    options['sample_names'] = ['s%d'%i for i in range(512)]
    a = TreeReader(options)
    a.read_from(e0, X_train)
    a.summary()
    b = ForestReader(options)
    b.read_from(rf, X_train)
    b.summary()
