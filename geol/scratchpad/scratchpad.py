# import sys
# sys.path.append(".")
# from GeoL.factory.square import Square

# sq = Square(area_name='Milan, Italy', extension='boundary')

# # sq.build_bbox()
# cc = sq.build_tessellation()

import logging
logger = logging.getLogger(__name__)


class Test():

    def __init__(self, *args, **kwargs):
        self._logger = logger

    def log(self, text):
        self._logger.info(text)

    def log_2(self, text):
        logger.info(text)

    def talk(self, text):
        print(text)
