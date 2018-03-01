"""
Script to create grid(s), given input args.
"""

# Authors: Gianni Barlacchi <gianni.barlacchi@gmail.com>

import argparse
import sys
import logging
import os
import matplotlib as mpl
mpl.use('TkAgg')
from geol.geometry import grid
import multiprocessing

#TODO switch to joblib
os.environ['NO_PROXY'] = "nominatim.openstreetmap.org"

logger = logging.getLogger(__name__)


def write_grid(outputfolder, meters, window_size, base_shape, crs):
    """
    Create the tessellation and save into the outputfolder.
    """
    try:
        if base_shape is None

        tessellation = Tessellation(factory, logger=logger)
        outputfolder = os.path.abspath(os.path.join(
            outputfolder, "tessellation_" + tessellation.id + ".geojson"))
        tessellation.write(outputfolder)

        return tessellation

    except:
        logger.error("Error in creating tessellation " +
                     outputfolder, exc_info=True)
        sys.exit(0)


def main(argv):

    parser = argparse.ArgumentParser('Build your own grid.')

    parser.add_argument('-o', '--outputfolder',
                        help='Output folder where to save the grids.',
                        action='store',
                        dest='outputfolder',
                        required='True',
                        type=str)

    parser.add_argument('-g', '--grid',
                        help='Type of grid to be used. Support types are: (i) square. It requires -a or -b.',
                        action='store',
                        dest='grid',
                        default='square',
                        type=str)

    parser.add_argument('-p', '--prefix',
                        action='store',
                        dest='prefix',
                        help='Prefix for the filename. By the default is <prefix>_<grid_type>_<cell_size>, by default is grid.',
                        default='grid',
                        type=str)

    parser.add_argument('-a', '--area',
                        action='store',
                        dest='area',
                        help='Area name in the format of "Area name, Country"',
                        type=str)

    parser.add_argument('-ws', '--window_size',
                        help='Size of the window around the shape centroid.',
                        action='store',
                        dest='window_size',
                        type=int)

    parser.add_argument('-C', '--crs',
                        help='Coordintate reference system for the output grid.',
                        action='store',
                        dest='crs',
                        default='4326',
                        type=str)

    parser.add_argument('-b', '--base_shape', action='store', dest='base_shape',
                        help='Path to the shape file used as a base to build the grid over.',
                        type=str)

    parser.add_argument('-s', '--size',
                        help='List of cell sizes (s1, s2, ..), default = 50.',
                        dest='sizes',
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
        logging.basicConfig(
            format='%(levelname)s: %(message)s', level=logging.INFO)

    elif(args.verbosity == 2):
        logging.basicConfig(
            format='%(levelname)s: %(message)s', level=logging.DEBUG)

    if args.mp == True:
        jobs = []

    for m in args.sizes:

        try:

            if (args.grid == 'square'):
                if args.base_shape:
                    factory = Square(meters=m, area_name=args.area,
                                     area=os.path.abspath(args.base_shape))
                else:
                    factory = Square(meters=m, area_name=args.area)

            # Get the factory according to the tessellation type in input
            if args.mp == True:

                os.path.join(dir_name, base_filename + "." + filename_suffix)

                outfile = os.path.abspath(args.outputfolder)
                p = multiprocessing.Process(target=write_grid, args=(args.outputfolder, logger,))
                jobs.append(p)
                p.start()

            else:
                write_tessellation(
                    factory=factory, outputfolder=args.outputfolder, logger=logger)

        except ValueError:
            logger.error("Value error instantiating the grid.", exc_info=True)
            sys.exit(1)

        except TypeError:
            logger.error("Type error building the grid.", exc_info=True)
            sys.exit(1)


if __name__ == "__main__":
    main(sys.argv[1:])
