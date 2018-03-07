"""
Script to create word2vec models, given a set of mapped POIs.
"""

# Authors: Gianni Barlacchi <gianni.barlacchi@gmail.com> Michele Ferretti <mic.ferretti@gmail.com>

import argparse
import pandas as pd
import numpy as np
import gensim
import logging
import string
import os
import sys
from sklearn.manifold import TSNE
import matplotlib
matplotlib.use('Agg')  # don't use Windows by default
import matplotlib.pyplot as plt
import multiprocessing
from geol.geol_logger.geol_logger import logger
from geol.utils.utils import select_category, normalize_words, pre_processing


def run_w2v_model(outputfolder, word_list, skip_gram, prefix, size, count, window, plot):
    """
    Run Word2Vec model
    """
    output = os.path.abspath(os.path.join(outputfolder, 'models', prefix + '_s'+str(size) +
                                          '_ws'+str(window)+'_c'+str(count)+'.model'))
    model = gensim.models.Word2Vec(
        word_list, sg=skip_gram, size=size, min_count=count, window=window, workers=8)  # size 5 is default
    model.save(output)
    if plot:
        tsne_plot(model, size, window, count, outputfolder, prefix)


def tsne_plot(model, size, window, count, outputfolder, prefix):
    """
    Creates and TSNE model and plots it
    """

    labels = []
    tokens = []

    for word in model.wv.vocab:
        tokens.append(model[word])
        labels.append(word)

    tsne_model = TSNE(perplexity=40, n_components=2,
                      init='pca', n_iter=2500, random_state=23)
    new_values = tsne_model.fit_transform(tokens)

    x = []
    y = []
    for value in new_values:
        x.append(value[0])
        y.append(value[1])

    plt.figure(figsize=(16, 16))
    for i in range(len(x)):
        plt.scatter(x[i], y[i])
        plt.annotate(labels[i],
                     xy=(x[i], y[i]),
                     xytext=(5, 2),
                     textcoords='offset points',
                     ha='right',
                     va='bottom')
    plt.title('Size:'+str(size)+' Window:'+str(window)+' Count:'+str(count))
    plt.savefig(os.path.abspath(os.path.join(outputfolder, 'imgs', prefix + '_s' + str(size)+'_ws' + str(window) +
                                             '_c'+str(count)+'.png')), bbox_inches='tight')
    plt.show()


def main(argv):

    parser = argparse.ArgumentParser('Build your own Word2Vec embeddings.')

    parser.add_argument('-i', '--inputfile',
                        help='Input file.',
                        action='store',
                        dest='input_file',
                        required=True,
                        type=str)

    parser.add_argument('-o', '--outputfolder',
                        help='Output folder where to save the grids.',
                        action='store',
                        dest='outputfolder',
                        required=True,
                        type=str)

    parser.add_argument('-p', '--prefix',
                        action='store',
                        dest='prefix',
                        help='Prefix for the filename in the form <prefix>_<grid_type>_<cell_size>. By default is w2v',
                        default='w2v',
                        type=str)

    parser.add_argument('-plt', '--plot',
                        action='store_true',
                        dest='plot',
                        help='t-SNE plot',
                        default=False)

    parser.add_argument('-sg', '--skip-gram',
                        help='Defines the training algorithm. If 1, skip-gram is employed; otherwise, CBOW is used.',
                        dest='skip_gram',
                        default=0,
                        type=int)

    parser.add_argument('-s', '--size',
                        help='List of vector sizes (s1, s2, ..), default = 50.',
                        dest='sizes',
                        nargs="+",
                        default=[50],
                        type=int)

    parser.add_argument('-ws', '--window_size)',
                        help='List of window sizes (s1, s2, ..), default = 50.',
                        dest='windows',
                        nargs="+",
                        default=[50],
                        type=int)

    parser.add_argument('-c', '--min_count',
                        help='List of minimum count sizes (s1, s2, ..), default = 50.',
                        dest='counts',
                        nargs="+",
                        default=[50],
                        type=int)

    parser.add_argument('-m', '--multiprocessing',
                        help='Abilitate multiprocessing (strongly suggested when more CPUs are available)',
                        dest='mp',
                        action='store_true',
                        default=False)

    parser.add_argument('-v', '--verbose',
                        help='Level of output verbosity.',
                        action='store',
                        dest='verbosity',
                        default=0,
                        type=int,
                        nargs="?")

    args = parser.parse_args()

    if(args.verbosity == 1):
        logger.setLevel(logging.INFO)

    elif(args.verbosity == 2):
        logger.setLevel(logging.INFO)

    if args.mp == True:
        jobs = []

    # load data
    word_list = pre_processing(os.path.abspath(args.input_file))

    # create word embeddings
    for size in args.sizes:
        for window in args.windows:
            for count in args.counts:
                try:
                    # Get the factory according to the tessellation type in input
                    if args.mp == True:

                        p = multiprocessing.Process(target=run_w2v_model, args=(
                            args.outputfolder, word_list, args.skip_gram, args.prefix, size, count, window, args.plot))

                        jobs.append(p)
                        p.start()

                    else:
                        output = os.path.abspath(os.path.join(args.outputfolder, 'models', args.prefix +
                                                              '_s' + str(size) + '_ws'+str(window)+'_c'+str(count)+'.model'))
                        run_w2v_model(output, word_list, args.skip_gram, size,
                                      count, window, args.plot)

                except ValueError:
                    logger.error(
                        "Value error instantiating the grid.", exc_info=True)
                    sys.exit(1)

                except TypeError:
                    logger.error(
                        "Type error building the grid.", exc_info=True)
                    sys.exit(1)


if __name__ == "__main__":
    main(sys.argv[1:])
