# Authors: Gianni Barlacchi <gianni.barlacchi@gmail.com>

import argparse
import sys
import logging
import os
from geol.utils import constants
from geol.geometry.grid import Tessellation
from geol.factory.square import Square
from geol.factory.custom import Custom
from geol.crawler import foursquare_crawler as crawler

os.environ['NO_PROXY'] = "nominatim.openstreetmap.org"

logger = logging.getLogger(__name__)

def write_tessellation(factory, outputfolder, logger=logger):
    """
    Create the tessellation and save into the outputfolder.
    """
    try:

        tessellation = Tessellation(factory, logger=logger)
        outputfolder = os.path.abspath(os.path.join(outputfolder, "tessellation_" + tessellation.id + ".geojson"))
        tessellation.write(outputfolder)

        return tessellation

    except:
        logger.error("Error in creating tessellation " + outputfolder, exc_info=True)
        sys.exit(0)


def main(argv):

    parser = argparse.ArgumentParser('Build your own grid.')

    parser.add_argument('-o', '--outputfolder',
                        help='Output folder where to save the grids.',
                        action='store',
                        dest='outputfolder',
                        required='True',
                        type=str)

    parser.add_argument('-b', '--base', action='store', dest='base_shape',
                        help='Path to the shape file used as a base for the grid.',
                        type=str)

    parser.add_argument('-s', '--size',
                        help='Cell size, default = 50.',
                        dest='size',
                        default=[50],
                        type=int)

    parser.add_argument('-v', '--verbose',
                        help='Level of output verbosity.',
                        action='store',
                        dest='verbosity',
                        default=0,
                        type=int,
                        nargs="?")

    parser.add_argument('-a', '--area',
                       action='store',
                       dest='area_name',
                       help='Area name in the format of "Area name, Country"',
                       required=True,
                       type=str)

    parser.add_argument('-t', '--tessellation',
                        help='Path to the tessellation. To have an high coverage it is important a tessellation with small cell (e.g. Squared tessellation with cell of 30meters.)',
                        action='store_true',
                        dest='tessellation',
                        default=False)

    parser.add_argument('-an', '--accountNumber',
                       action='store',
                       dest='account_number',
                       help='Foursquare account number',
                       default=1,
                       type=int)

    """
    parser.add_argument('-cid', '--clientID',
                       action='store',
                       dest='client_id',
                       help='Foursquare client ID',
                       default=constants.CLIENT_ID,
                       type=str)

    parser.add_argument('-cs', '--clientSecret',
                       action='store',
                       dest='client_secret',
                       help='Foursquare client secret',
                       default=constants.CLIENT_SECRET,
                       type=str)
    """

    args = parser.parse_args()

    meters = args.size
    area_name = args.area_name
    outputfolder = args.outputfolder + "/tessellation/"

    if (args.verbosity == 1):
        logging.basicConfig(format='%(levelname)s: %(message)s', level=logging.INFO)

    elif (args.verbosity == 2):
        logging.basicConfig(format='%(levelname)s: %(message)s', level=logging.DEBUG)

    # Crete the tessellation if not passed in input. By default we use a square tessellation.

    if args.tessellation == True:
        factory = Custom(input=args.tessellation, area_name=area_name)
    else:
        factory = Square(meters=meters, area_name=area_name)

    tessellation = write_tessellation(factory, outputfolder)

    outputfile = os.path.join(args.outputfolder + "/foursquare_raw/", area_name.split(",")[0].strip() + "_poi.csv")

    #TODO read from config file
    if args.account_number == 1:
        client_id = constants.CLIENT_ID_1
        client_secret = constants.CLIENT_SECRET_1
    elif args.account_number == 2:
        client_id = constants.CLIENT_ID_2
        client_secret = constants.CLIENT_SECRET_2
    elif args.account_number == 3:
        client_id = constants.CLIENT_ID_3
        client_secret = constants.CLIENT_SECRET_3
    else:
        client_id = constants.CLIENT_ID_4
        client_secret = constants.CLIENT_SECRET_4

    c = crawler.Foursquare(client_id=client_id, client_secret=client_secret)
    c.start(tessellation.tessellation, outputfile)


if __name__ == "__main__":
    main(sys.argv[1:])
