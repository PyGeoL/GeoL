"""
Script to create grid(s), given input args.
"""

# Authors: Gianni Barlacchi <gianni.barlacchi@gmail.com>

import argparse
import sys
import logging
import pandas as pd
import gensim
import pkg_resources
from geol.geol_logger.geol_logger import logger
import re
import numpy as np


def main(argv):

    parser = argparse.ArgumentParser('Build your own grid.')

    parser.add_argument('-o', '--outputfolder',
                        help='Output folder where to save the matrix.',
                        action='store',
                        dest='outputfolder',
                        required=True,
                        type=str)

    parser.add_argument('-i', '--input',
                        help='Input file with point-of-interests. NOTE: in the case of strategy=nearest|alphabetically, the input file must contains the column cellID.',
                        action='store',
                        dest='inputfile',
                        required=True,
                        type=str)

    parser.add_argument('-a', '--area',
                        action='store',
                        dest='area',
                        help='Area name',
                        default=None,
                        type=str)

    parser.add_argument('-v', '--verbose',
                        help='Level of output verbosity.',
                        action='store',
                        dest='verbosity',
                        default=0,
                        type=int,
                        nargs="?")

    args = parser.parse_args()

    if(args.verbosity == 1):
        logging.basicConfig(
            format='%(levelname)s: %(message)s', level=logging.INFO)

    elif(args.verbosity == 2):
        logging.basicConfig(
            format='%(levelname)s: %(message)s', level=logging.DEBUG)

    logger.info("Loading w2v model.")

    model = gensim.models.Word2Vec.load(args.inputfile)

    tree = pd.read_csv(pkg_resources.resource_filename(
        'geol', '/resources/category_tree.csv'), encoding="iso-8859-1")

    words = tree['level1_name'].dropna().drop_duplicates().tolist() + \
        tree['level2_name'].dropna().drop_duplicates().tolist() + \
        tree['level3_name'].dropna().drop_duplicates().tolist() + \
        tree['level4_name'].dropna().drop_duplicates().tolist()

    m = re.search('_s([0-9]+)_', args.inputfile)
    if m:
        size = m.group(1)

    m = re.search('.(.+).model', args.inputfile)
    if m:
        model_details = m.group(1)

    outputfile = "matrix_" + args.area+
    "_" + model_details + ".txt"

    f = open(outputfile, 'w')

    for word in words:

        word = re.sub('[^A-Za-z0-9#]+', " ", word).lower().strip()

        w = word.split(' ')
        v = [0] * size

        if len(w) > 1:
            tmp_w2v = []
            for e in w:
                if e in model:
                    tmp_w2v.append(model[e])
            if len(tmp_w2v) > 0:
                v = np.mean(tmp_w2v, axis=0)
        elif word in model:
            v = model[word]

        v = map(str, v)
        s = ','.join(map(str, v))
        f.write(word.replace(" ", "_") + "::n" + "\t1.0\t0\t" + s + "\n")

    f.close()


if __name__ == "__main__":
    main(sys.argv[1:])
