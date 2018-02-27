"""
File description
"""

# Authors: Gianni Barlacchi <gianni.barlacchi@gmail.com> Michele Ferretti <mic.ferretti@gmail.com>

import logging
import time
from ..utils import constants
import geopandas as gpd
import sys


class Tessellation:

    def __init__(self, factory=None, **kwargs):
        """

        :param factory: factory class to build the tessellation
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

        self.__tessellation, self.__area = factory.build_tessellation()

        end = time.time()
        self.logger.info(
            "End creating grid (" + self.__properties['id'] + ") in " + str(end - start))

    def write(self, outputfile):
        """
        Write the tesselation on file.
        TODO: manage the problem with crs and possibly add properties!
        """
        with open(outputfile, 'w') as f:
            f.write(self.__tessellation.to_json())

    @property
    def tessellation(self):
        return self.__tessellation

    @property
    def area(self):
        return self.__area

    @property
    def id(self):
        if 'id' in self.__properties:
            return self.__properties['id']
        return None
