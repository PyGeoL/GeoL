"""
File description
"""

# Authors: Gianni Barlacchi <gianni.barlacchi@gmail.com>
import logging
import time
import six
import abc

from ..utils import constants
import geopandas as gpd
import sys

@six.add_metaclass(abc.ABCMeta)
class Grid:

    def __init__(self, factory=None, **kwargs):
        """

        :param kwargs:
                        logger
                        area_name (mandatory if factory is none)
                        input (mandatory if factory is none)
        """

        self.logger = kwargs.get('logger', None)

        if self.logger is None:
            logging.basicConfig(format='%(levelname)s: %(message)s')
            self.logger = logging.getLogger(__name__)

        self.__factory = factory
        self.__properties = factory.set_properties()

        self.logger.info("Start creating tessellation (" +
                         self.__properties['id'] + ")")
        start = time.time()

        self.__grid, self.__area = factory.build_tessellation()

        end = time.time()
        self.logger.info(
            "End creating grid (" + self.__properties['id'] + ") in " + str(end - start))

    @abc.abstractmethod
    def read(self, inputfile):
        pass

    @abc.abstractmethod
    def write(self, outputfile):
       pass

    @property
    def grid(self):
           return self.__grid

       with open(outputfile, 'w') as f:
            f.write(self.__tessellation.to_json())


    @property
    def area(self):
        return self.__area

    @property
    def id(self):
        if 'id' in self.__properties:
            return self.__properties['id']
        return None
