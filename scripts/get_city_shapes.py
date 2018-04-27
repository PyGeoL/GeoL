# Authors: Gianni Barlacchi <gianni.barlacchi@gmail.com>
#          Michele Ferretti <mic.ferretti@gmail.com>

"""
Script to fetch city shapes from Nominatim
And save them as geojson
"""

import argparse
import pandas as pd
import geopandas as gpd
from geol.utils import utils
import os
import sys

# recast the columns of boolean type over to integer
# so FIONA can save the GeoDataFrame...


def gdf_bool_to_int(gdf):
    """For a given GeoDataFrame, returns a copy that
    recasts all `bool`-type columns as `int`.

    GeoDataFrame -> GeoDataFrame"""
    df = gdf.copy()
    coltypes = gpd.io.file.infer_schema(df)['properties']
    for c in coltypes.items():
        if c[1] == 'bool':
            colname = c[0]
            df[colname] = df[colname].astype('int')
    return df


def main(argv):
    parser = argparse.ArgumentParser('Build your own grid.')
    parser.add_argument('-o', '--outputfolder',
                        help='Output folder where to save the geojson.',
                        action='store',
                        dest='outputfolder',
                        required='True',
                        type=str)

    parser.add_argument('-c', '--city-name',
                        help='City name. It fetches by default the second result from Nominatim.',
                        action='store',
                        dest='city_name',
                        required="True",
                        type=str)

    args = parser.parse_args()

    OUTPUT_CITY_SHAPE = os.path.join(
        args.outputfolder, args.city_name + ".geojson")
    # recast the columns of boolean type over to integer
    city_shape = utils.get_area_boundary(args.city_name, 2)
    try:
        os.remove(OUTPUT_CITY_SHAPE)
    except OSError:
        pass
    gdf_bool_to_int(city_shape).to_file(
        OUTPUT_CITY_SHAPE, driver="GeoJSON", encoding='utf-8')


if __name__ == "__main__":
    main(sys.argv[1:])
