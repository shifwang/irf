from abc import ABC, abstractmethod
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

class ForestReader(Reader):
    """docstring for ForestReader."""
    def read_from(self, forest):
        self.info = dict()
        self.info['number_of_samples'] = None
        self.info['number_of_features'] = None
        self.info['feature_importances'] = None



    def

if __name__ == '__main__':
    a = ForestReader()
