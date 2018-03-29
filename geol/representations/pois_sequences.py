"""
File description
"""

# Authors: Gianni Barlacchi <gianni.barlacchi@gmail.com>

import pandas as pd
from geol.utils import constants
from geol.geol_logger.geol_logger import logger
import pysal
import geopandas as gpd
from shapely.ops import nearest_points
from shapely.geometry import Point
from geol.geometry.grid import Grid


class POISequences():

    def __init__(self, pois):

        self._pois = pois

    @classmethod
    def from_csv(cls, inputfile, sep='\t', crs=constants.default_crs):
        """
        Read csv file with POIs details, including latitude and longitude
        :param inputfile:
        :param sep:
        :return:
        """
        #  Read foursquare MAPPED onto the grid
        logger.info("Reading POIs dataset.")
        df = pd.read_csv(inputfile, sep=sep)

        # Create GeoDataFrame from the read DataFrame
        logger.info("Create GeoDataFrame")
        geometry = [Point(xy) for xy in zip(df.longitude, df.latitude)]
        gdf = gpd.GeoDataFrame(
            df, index=df.index, geometry=geometry, crs={'init': crs})

        return cls(gdf.to_crs({'init': constants.universal_crs}))

    def _centroid_distance(self, df):
        return df['geometry'].distance(df['centroid'])

    def _nearest(self, df):

        points = df[['categories', 'geometry']].copy()

        s = []  # str(df.iloc[0]['categories']) + "\t"
        s.append(str(df.iloc[0]['categories']))

        p = df.iloc[0]['geometry']
        points = points[points.geometry != p]

        while (len(points) > 0):
            nearest = points.geometry == nearest_points(
                p, points.geometry.unary_union)[1]

            p = points[nearest]['geometry'].iloc[0]
            s.append(points[nearest]['categories'].iloc[0])

            points = points[points['geometry'] != p]

        if len(s) > 2:
            return '\t'.join(s)
        else:
            None

    def _distance(self, band_size=500):

        logger.info("Building sequences for each point in the sapce")
        wthresh = pysal.weights.DistanceBand.from_dataframe(self._pois, band_size, p=2, binary=False, ids=self._pois.index)

        ds = []
        for index, indexes in wthresh.neighbors.items():
            if len(indexes) == 0:
                d = {}
                d['observation'] = index
                d['observed'] = index
                d['distance'] = None
                ds.append(d)
            else:
                for i in range(len(indexes)):
                    d = {}
                    d['observation'] = index
                    d['observed'] = indexes[i]
                    d['distance'] = wthresh.weights[index][i]
                    ds.append(d)

        obs = pd.DataFrame(ds)

        return obs

    def distance_based_sequence(self, band_size, outfile):

        obs = self._distance(band_size)

        # First step - get the categories for observation ID
        obs_1 = obs.merge(self._pois[['categories']], left_on='observation', right_index=True).rename(
            columns={'categories': 'cat_observation'})

        # Second step - get the categories for observed ID
        obs_2 = obs_1.merge(self._pois[['categories']], left_on='observed', right_index=True).rename(
            columns={'categories': 'cat_observed'})

        # Order by inverse of distance, which is not the real distance but the interaction value from PySal.
        # The interaction among points decreases as the distance increase.
        obs_2.sort_values(by=['observation', 'distance'], ascending=False, inplace=True)

        # Third step - build the sequence joining the words. We keep sequences with at least 3 words.
        obs_3 = obs_2.groupby(['observation', 'cat_observation']).apply(
            lambda x: '\t'.join(x['cat_observed']) if len(x) > 2 else None).reset_index().dropna().rename(
            columns={0: "sequence"})
        obs_3.loc[:, "complete"] = obs_3['cat_observation'] + "\t" + obs_3['sequence']

        # Fourth step - join the pois dataframe with the sequences and save into a csv
        logger.info("Save sequences")
        self._pois[['categories', 'geometry']].merge(obs_3, left_index=True, right_on='observation')[
            ['categories', 'geometry', 'complete']].to_csv(outfile.split(".csv")[0] + "_check.csv", sep='\t', index=False)

        obs_3[['complete']].to_csv(outfile, index=False, header=False)

    def nearest_based_sequence(self, outfile, inputgrid):

        logger.info("Load the grid.")

        # Load inputgrid
        g = Grid.from_file(inputgrid)
        grid = g.grid.to_crs({'init': constants.universal_crs})
        grid.loc[:, 'centroid'] = grid.centroid

        df = self._pois.copy()

        df = df.merge(grid[['cellID', 'centroid']], on='cellID')

        logger.info("Compute centroid for cells and build the sequences")
        df.loc[:, 'distance'] = df.apply(self._centroid_distance, axis=1)
        df.sort_values(by=['cellID', 'distance'], inplace=True, ascending=True)

        logger.info("Save sequences")
        df.groupby('cellID').apply(self._nearest).dropna().to_csv(outfile, index=False, header=None)

    def alphabetically_sequence(self, outfile):

        if('cellID' not in self._pois.columns):
            raise ValueError(
                "The input file with POIs must contains the column cellID.")

        logger.info("Build the sequences")
        self._pois.sort_values(by=["cellID", "categories"]).groupby('cellID')\
            .apply(lambda x: '\t'.join(x['categories']) if len(x) > 2 else None).dropna().to_csv(outfile, index=False, header=None)