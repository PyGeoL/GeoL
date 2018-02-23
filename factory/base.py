"""
All base factory classes.
"""

# Authors: Gianni Barlacchi <gianni.barlacchi@gmail.com>

import six
import abc


@six.add_metaclass(abc.ABCMeta)
class TessellationFactory():
    """
    Base class for all the factory that build the tessellation.
    """

    @abc.abstractmethod
    def set_properties(self):
        pass

    @abc.abstractmethod
    def build_tessellation(self):
        pass
