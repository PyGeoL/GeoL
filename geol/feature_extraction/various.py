"""
File description
"""

# Authors: Gianni Barlacchi <gianni.barlacchi@gmail.com>

from geol.feature_extraction.base import FeatureGenerator
from geol.utils import utils
import pandas as pd
import gensim
from geol.geol_logger.geol_logger import logger
import pkg_resources
import numpy as np

class BOC(FeatureGenerator):

    def __init__(self, pois):

        super(BOC, self).__init__()
        self._pois = pois
        self._categories = pois["category"].drop_duplicates().values

    @classmethod
    def from_csv(cls, input, sep='\t', category_column='categories', level=5):

        # load foursquare dataset mapped on a particular grid
        df = pd.read_csv(input, sep=sep)
        df[category_column] = df[category_column].astype(str)

        # assign category to each record of the dataset

        df.loc[:, "category"] = utils.select_category(list(df[category_column]), level)

        # drop entry with empty category
        df = df.loc[df["category"] != "nan"]

        return cls(df)

    def generate(self):

        # compute aggregation on 'cellID', summing inside the columns 'checkin' and 'usercount',

        cell_df = self._pois.groupby(['cellID']).sum()
        cell_df.reset_index(inplace=True)

        # compute aggregation on 'cellID' and 'categ' to count the number of each category in each cell
        cat_df = pd.DataFrame(self._pois.groupby(['cellID', 'category']).size(), columns=['count'])
        cat_df.reset_index(inplace=True)

        # create a table with the same information of 'cat_df' but using categories names as columns names
        r = pd.pivot_table(cat_df, values='count', index=['cellID'], columns=['category']).reset_index()

        # fill empty cells
        r[self._categories] = r[self._categories].fillna(0)

        # cast to int of every value
        r = r.astype(int)

        # merge the two table in order to get one table with all the information
        df = cell_df.merge(r, on='cellID')

        # sort columns
        df = df[list(self._categories) + ['cellID']]

        # sort rows by cell id
        df.sort_values(by='cellID', inplace=True)

        # Set Feature
        self._features = df


class cell2vec(FeatureGenerator):

    def __init__(self, pois, w2v_model, binary=False):

        super(cell2vec, self).__init__()
        logger.info("Loading w2v model")
        self._pois = pois
        self._categories = pois["category"].drop_duplicates().values

        self._model = gensim.models.KeyedVectors.load_word2vec_format(w2v_model, binary=binary)

    @classmethod
    def from_csv(cls, input, model, binary=False, sep='\t', category_column='categories', level=5):

        logger.info("Loading mapped POIs")

        # load foursquare dataset mapped on a particular grid
        df = pd.read_csv(input, sep=sep)
        df[category_column] = df[category_column].astype(str)

        # assign category to each record of the dataset

        df.loc[:, "category"] = utils.select_category(list(df[category_column]), level)

        # drop entry with empty category
        df = df.loc[df["category"] != "nan"]

        return cls(df, model,binary=binary)


    def generate(self):

        # Get embedding for each word
        tmp = self._pois.merge(self._pois.apply(utils.get_embedding, args=(self._model,), axis=1),
                               left_index=True, right_index=True)

        # Keep columns order as with w2v
        cols = [c for c in tmp.columns if type(c) is int]
        cols.sort()

        self._features = tmp[['cellID'] + cols]
        self._features = self._features.groupby("cellID").sum()

    def write(self, outfile):

        cols = ["f_w2v_" + str(c) if c is not "cellID" else c for c in self._features.columns]
        self._features.columns = cols

        return self._features.to_csv(outfile, index=True)

class SPTKMatrix(FeatureGenerator):

    def __init__(self):

        super(SPTKMatrix, self).__init__()


    def generate(self, model_path, binary=False, category_column='categories'):

        model = gensim.models.KeyedVectors.load_word2vec_format(model_path, binary=binary)
        size = model.vector_size

        tree = pd.read_csv(pkg_resources.resource_filename(
            'geol', '/resources/category_tree.csv'), encoding='iso-8859-1')

        words = tree['level1_name'].dropna().drop_duplicates().tolist() + \
                tree['level2_name'].dropna().drop_duplicates().tolist() + \
                tree['level3_name'].dropna().drop_duplicates().tolist() + \
                tree['level4_name'].dropna().drop_duplicates().tolist()

        word_vectors = {}

        for word in words:

            word = utils.normalize_word(word)

            w = word.split(' ')
            v = [0] * int(size)

            if len(w) > 1:
                tmp_w2v = []
                for e in w:
                    if e in model:
                        tmp_w2v.append(model[e])
                if len(tmp_w2v) > 0:
                    v = np.mean(tmp_w2v, axis=0)
            elif word in model:
                v = model[word]

            word_vectors[word] = v

        self._features = word_vectors

    def write(self, outfile):

        f = open(outfile, 'w')

        for k,v in self._features.iteritems():

            v = map(str, v)
            s = ','.join(map(str, v))
            f.write( (k.replace(" ", "_") + "::n" + "\t1.0\t0\t" + s + "\n").encode('utf8') )

        f.close()
