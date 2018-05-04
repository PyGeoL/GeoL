
# coding: utf-8


# Load FourSquare MAPPED dataset and assign to each square in the grid the
# number of POI of each FS category.

import pandas as pd
from geol.feature_extraction.various import BOC
import sys
import argparse
import os


def category(df, level):
    tmp = df['categories'].split(":")

    if len(tmp) > level:
        return tmp[level]
    else:
        return tmp[len(tmp) - 1]

def main(argv):

    parser = argparse.ArgumentParser('Generate BOC')

    parser.add_argument('-m', '--mapped_pois',
                            help='Mapped POIs.',
                            action='store',
                            dest='pois_mapped',
                            required='True',
                            type=str)

    parser.add_argument('-o', '--output',
                        help='Output folder',
                        action='store',
                        dest='output',
                        required='True',
                        type=str)

    parser.add_argument('-l', '--level',
                            help='Level to which generate the BOC',
                            action='store',
                            dest='level',
                            default=5,
                            type=int)

    args = parser.parse_args()

    input = os.path.abspath(args.pois_mapped)
    output = os.path.abspath(args.output)

    pois = BOC.from_csv(input, level=args.level)
    pois.generate()
    pois.write(output)

if __name__ == "__main__":
    main(sys.argv[1:])