"""
File description
"""

# Authors: Gianni Barlacchi <gianni.barlacchi@gmail.com>

import six, abc
from geol.utils import utils

@six.add_metaclass(abc.ABCMeta)
class FeatureGenerator:

    def __init__(self, pois):

        self._pois = pois
        self._categories = pois["category"].drop_duplicates().values
        self._features = None

    @abc.abstractmethod
    def generate(self):
        pass

    @property
    def features(self):
        return self._features

    def write(self, outfile):

        # normalize columns name
        cols = ["f_" + utils.normalize_word(c) if c is not 'cellID' else c for c in self._features.columns]
        self._features.columns = cols

        return self._features.to_csv(outfile, index=False)