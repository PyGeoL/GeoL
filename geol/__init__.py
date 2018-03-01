"""
File description
"""

# Authors: Gianni Barlacchi <gianni.barlacchi@gmail.com> Michele Ferretti <mic.ferretti@gmail.com>


import logging.handlers
import os
import sys

# import sys
# sys.path.append(".")
# import scratchpad

dir = os.path.dirname(__file__)

# Logger Handlers settings
#fh = logging.handlers.RotatingFileHandler(
#    os.path.join(dir, '../log/geol.log'), maxBytes=1000000, backupCount=10)
#fh.setLevel(logging.DEBUG)

#fh2 = logging.handlers.RotatingFileHandler(
#    os.path.join(dir, '../log/geol_info_only.log'), maxBytes=1000000, backupCount=5)
#fh2.setLevel(logging.INFO)

#er = logging.handlers.RotatingFileHandler(os.path.join(dir, '../log/geol_errors.log'), maxBytes=2000000, backupCount=2)
#er.setLevel(logging.WARNING)

ch = logging.StreamHandler(sys.stdout)
ch.setLevel(1)

#fh.setFormatter(logging.Formatter(
#    '%(asctime)s -[%(filename)s:%(lineno)s] -  %(name)s -  %(funcName)2s() - %(levelname)s - %(message)s'))

#fh2.setFormatter(logging.Formatter(
#    '%(asctime)s -[%(filename)s:%(lineno)s] -  %(name)s -  %(funcName)2s() - %(levelname)s - %(message)s'))

#er.setFormatter(logging.Formatter(
#    '%(asctime)s -[%(filename)s:%(lineno)s] -  %(name)s -  %(funcName)2s() - %(levelname)s - %(message)s'))

ch.setFormatter(logging.Formatter('%(name)s - %(levelname)s - %(message)s'))

# Define Root Logger to rule them all!
root = logging.getLogger()
root.setLevel(logging.DEBUG)
#root.addHandler(fh)
#root.addHandler(fh2)
#root.addHandler(ch)
#root.addHandler(er)
