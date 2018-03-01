import logging
import os
import sys

LOG_FORMAT = '%(asctime)s -[%(filename)s:%(lineno)s] -  %(name)s -  %(funcName)2s() - %(levelname)s - %(message)s'

# set up logging to console
console = logging.StreamHandler(sys.stdout)
console.setLevel(logging.DEBUG)
# set a format which is simpler for console use
formatter = logging.Formatter(LOG_FORMAT)
console.setFormatter(formatter)
# add the handler to the root logger
logging.getLogger('').addHandler(console)
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
