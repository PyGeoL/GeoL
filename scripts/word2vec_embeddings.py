"""
Script to create word2vec models, given a set of mapped POIs.
"""

# Authors: Gianni Barlacchi <gianni.barlacchi@gmail.com> Michele Ferretti <mic.ferretti@gmail.com>

import argparse
import pandas as pd
import numpy as np
import gensim
import logging
import re
import string
import os
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt

# word_list = pre_processing(INPUT_FILE)


# windows = [3, 5, 7, 10]
# sizes = [50, 100, 200]
# counts = [2, 5, 10]


def normalize_words(words_array):
    """
    remove nasty chars
    and sets word to lowercase
    """
    pattern = re.compile('[\W_]+', re.UNICODE)
    return [pattern.sub(r'', x.lower()) for x in words_array]


def select_category(list_of_labels, level):
    """
    Aggregates all the labels for each feature
    at the given level
    separated with _
    """
    tmp = []

    for x in list_of_labels:
        norm_w = normalize_words(x.split(":"))
#         print(norm_w, level)
        if len(norm_w) > level:
            tmp.append(norm_w[level])
        else:
            tmp.append(norm_w[len(norm_w) - 1])
#             print("Selected level is too deep!")
    return tmp


def pre_processing(INPUT_FILE, depth_level=3):
    """
    Inputs a file of tab-separated labels for each grid cell
    Returns array of array of joined labels for specified depth level (default=3)
    """

    #  import text file
    with open(INPUT_FILE, 'r') as input:
        text = input.read()

    # split on new lines and remove empty lines
    labels_list = [x.split('\t') for x in list(filter(None, text.split('\n')))]

    # select labels at given depth and join them
    labels_joined_list = [select_category(x, depth_level) for x in labels_list]

    return labels_joined_list


def run_w2v_model(outputfolder, word_list, size, count, window, plot):
    """
    Run Word2Vec model
    """
    output = os.path.abspath(os.path.join(outputfolder, 'models', args.prefix + str(size) +
                                          '_'+str(window)+'_'+str(count)+'.model'))
    model = gensim.models.Word2Vec(
        word_list, size=size, min_count=count, window=window, workers=8)  # size 5 is default
    model.save(output)
    if plot:
        tsne_plot(model, size, window, count, outputfolder)


def tsne_plot(model, size, window, count, outputfolder):
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
    plt.savefig(os.path.abspath(os.path.join(outputfolder, 'imgs', 'nearest_'+str(size)+'_' + str(window) +
                                             '_'+str(count)+'.png', bbox_inches='tight')))
    plt.show()


def main(argv):

    parser = argparse.ArgumentParser('Build your own grid.')

    parser.add_argument('-i', '--inputfile',
                        help='Input file.',
                        action='store',
                        dest='input_file',
                        required='True',
                        type=str)

    parser.add_argument('-o', '--outputfolder',
                        help='Output folder where to save the grids.',
                        action='store',
                        dest='outputfolder',
                        required='True',
                        type=str)

    parser.add_argument('-p', '--prefix',
                        action='store',
                        dest='prefix',
                        help='Prefix for the filename in the form <prefix>_<grid_type>_<cell_size>. By default is "w2v."',
                        default='w2v',
                        type=str)

    parser.add_argument('-plt', '--plot',
                        action='store',
                        dest='plot',
                        help='t-SNE plot',
                        default='w2v',
                        type=str)

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

    args = parser.parse_args()

    if(args.verbosity == 1):
        logger.basicConfig(
            format='%(levelname)s: %(message)s', level=logging.INFO)

    elif(args.verbosity == 2):
        logger.basicConfig(
            format='%(levelname)s: %(message)s', level=logging.DEBUG)

    if args.mp == True:
        jobs = []

    # load data
    word_list = pre_processing(os.path.abspath(input_file))

    # create word embeddings
    for size in args.sizes:
        for window in args.windows:
            for count in counts:
                try:
                    # Get the factory according to the tessellation type in input
                    if args.mp == True:

                        p = multiprocessing.Process(target=run_w2v_model, args=(
                            args.outputfolder, word_list, size, count, window, args.plt))

                        jobs.append(p)
                        p.start()

                    else:
                        run_w2v_model(output, word_list, size, count, window)

    except ValueError:
        logger.error("Value error instantiating the grid.", exc_info=True)
        sys.exit(1)

    except TypeError:
        logger.error("Type error building the grid.", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main(sys.argv[1:])
