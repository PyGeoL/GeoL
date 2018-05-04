"""
File description
"""

# Authors: Gianni Barlacchi <gianni.barlacchi@gmail.com>

from geol.feature_extraction.base import FeatureGenerator
from geol.utils import utils
import pandas as pd
import gensim
from geol.geol_logger.geol_logger import logger


class BOC(FeatureGenerator):

    def __init__(self, pois):

        super(BOC, self).__init__(pois)

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

        super(cell2vec, self).__init__(pois)
        logger.info("Loading w2v model")
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