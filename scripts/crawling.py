# Authors: Gianni Barlacchi <gianni.barlacchi@gmail.com>

import argparse
import sys
import logging
import os
from geol.utils import utils
from geol.geometry.squaregrid import SquareGrid
from geol.crawler import foursquare_crawler

os.environ['NO_PROXY'] = "nominatim.openstreetmap.org"

logger = logging.getLogger(__name__)

def write_grid(output, size, window_size, crs, area_name, base_shape):

    # Create the tessellation and save into the outputfolder.
    try:
        grid = None

        if base_shape is not None:
            grid = SquareGrid.from_file(base_shape, meters=size, window_size=window_size, grid_crs=crs)
        else:
            grid = SquareGrid.from_name(area_name, meters=size, window_size=window_size, grid_crs=crs)

    except:
        logger.error("Error in creating tessellation " + output, exc_info=True)
        sys.exit(0)

    return grid


def main(argv):

    parser = argparse.ArgumentParser('Foursquare crawler.')

    parser.add_argument('-o', '--outputfolder',
                        help='Output folder where to save the grids.',
                        action='store',
                        dest='outputfolder',
                        required='True',
                        type=str)

    parser.add_argument('-k', '--keys',
                        help='File (Json) with Foursquare Keys.',
                        action='store',
                        dest='keys',
                        required='True',
                        type=str)

    parser.add_argument('-p', '--prefix',
                        action='store',
                        dest='prefix',
                        help='Prefix for the filename. By the default is <prefix>_<grid_type>_<cell_size>, by default is grid.',
                        default='grid',
                        type=str)

    parser.add_argument('-a', '--area',
                        action='store',
                        dest='area_name',
                        help='Area name in the format of "Area name, Country"',
                        default=None,
                        type=str)

    parser.add_argument('-ws', '--window_size',
                        help='Size of the window around the shape centroid.',
                        action='store',
                        dest='window_size',
                        default=None,
                        type=int)

    parser.add_argument('-r', '--restart',
                        help='Restarting point.',
                        action='store',
                        dest='restart',
                        default=None,
                        type=int)

    parser.add_argument('-b', '--base_shape', action='store',
                        help='Path to the shape file used as a base to build the grid over.',
                        dest='base_shape',
                        default=None,
                        type=str)

    parser.add_argument('-s', '--size',
                        help='Cell size, default = 50.',
                        dest='size',
                        default=50,
                        type=int)

    parser.add_argument('-an', '--accountNumber',
                        action='store',
                        dest='account_number',
                        help='Foursquare account number',
                        default="1",
                        type=str)

    parser.add_argument('-v', '--verbose',
                        help='Level of output verbosity.',
                        action='store',
                        dest='verbosity',
                        default=0,
                        type=int,
                        nargs="?")

    args = parser.parse_args()

    # Get foursquare key from key's file.
    foursquare_keys = utils.read_foursqaure_keys(args.keys)

    outputfile = os.path.abspath(os.path.join(args.outputfolder, args.prefix + "_foursquare_pois.csv"))

    if (args.verbosity == 1):
        logging.basicConfig(
            format='[ %(levelname)s: %(message)s ]', level=logging.INFO)

    elif (args.verbosity == 2):
        logging.basicConfig(
            format='[ %(levelname)s: %(message)s ]', level=logging.DEBUG)

    # Crete the tessellation if not passed in input. By default we use a square tessellation.
    grid = write_grid(outputfile, args.size, args.window_size, "epsg:4326", args.area_name, args.base_shape)

    logger.info("Loading Foursquare credentials")
    client_id = foursquare_keys[args.account_number]['CLIENT_ID']
    client_secret = foursquare_keys[args.account_number]['FOURSQUARE_API_TOKEN']

    logger.info("Starting the crawler")
    c = foursquare_crawler.Foursquare(client_id=client_id, client_secret=client_secret)
    c.start(grid.grid, outputfile, restart=args.restart)


if __name__ == "__main__":
    main(sys.argv[1:])
