
# coding: utf-8


# Load FourSquare MAPPED dataset and assign to each square in the grid the
# number of POI of each FS category.


from geol.feature_extraction.various import cell2vec
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

    parser.add_argument('-w', '--w2v_model',
                        help='Word2Vec model.',
                        action='store',
                        dest='model',
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

    parser.add_argument('-b', '--binary',
                        help='True if the word2vec model has been saved in a binary mode.',
                        dest='binary',
                        action='store_true',
                        default=False)

    args = parser.parse_args()

    input = os.path.abspath(args.pois_mapped)
    model = os.path.abspath(args.model)
    output = os.path.abspath(args.output)

    pois = cell2vec.from_csv(input, model, binary=args.binary, level=args.level)
    pois.generate()
    pois.write(output)



if __name__ == "__main__":
    main(sys.argv[1:])
