from geol.geol_logger.geol_logger import logger


class Test():

    def __init__(self, *args, **kwargs):
        self._logger = logger

    def log(self, text):
        self._logger.info(text)

    def log_2(self, text):
        logger.info(text)

    def log_3(self, text):
        logger.debug(text)

    def talk(self, text):
        print(text)
