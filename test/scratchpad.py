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
