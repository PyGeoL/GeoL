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

    def write(self, outfile):

        # normalize columns name
        cols = [utils.normalize_word(c) for c in self._pois.columns if c is not 'cellID'] + ['cellID']
        self._pois.columns = cols

        return self._pois.to_csv(outfile, index=False)