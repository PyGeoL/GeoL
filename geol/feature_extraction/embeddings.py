"""
File description
"""

# Authors: Gianni Barlacchi <gianni.barlacchi@gmail.com>


from geol.feature_extraction.base import FeatureGenerator
from geol.utils import utils
import pandas as pd
import gensim

class Embeddings(FeatureGenerator):

    def __init__(self, pois):

        FeatureGenerator.__init__(pois)



    def generate(self, model, area_based=True, strategy='avg'):

        """
        TODO: create the NOT area based function
        """

        # load w2v_model
        model = gensim.models.Word2Vec.load(model)


        # group every cell
        grouped_gdf = gdf.groupby('cellID')

        output = {}
        with open(output_file, 'w') as out:
            for cell, group in grouped_gdf:
                output[cell] = []
                for categories_raw in group['categories']:
                    # select level
                    category = utils.select_category(
                        categories_raw.split(':'), level)[-1]
                    # lookup category in w2v
                    try:
                        vector = model[category]
                        output[cell].append(np.array(vector))
                    except(KeyError):
                        pass
                if len(output[cell]) == 0:
                    output[cell] = [np.zeros(int(size))]

                # sum vectors
                sum_w = sum_vectors(output[cell])
                sum_w_str = str("\t".join(map(str, sum_w)))
                text_to_write = str(cell) + '\t' + sum_w_str + '\n'

                out.write(text_to_write)