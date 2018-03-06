"""
Script to create grid(s), given input args.
"""

# Authors: Gianni Barlacchi <gianni.barlacchi@gmail.com> Michele Ferretti <mic.ferretti@gmail.com>

import argparse
import sys
from geol.geol_logger.geol_logger import logger
import os
from geol.geometry.squaregrid import SquareGrid
import multiprocessing

# TODO switch to joblib
os.environ['NO_PROXY'] = "nominatim.openstreetmap.org"


def write_grid(output, size, type, window_size, crs,
               area_name, base_shape):
    """
    Create the tessellation and save into the outputfolder.
    """
    try:
        grid = None

        if base_shape is not None:
            grid = SquareGrid.from_file(
                base_shape, meters=size, window_size=window_size, grid_crs=crs)
        else:
            grid = SquareGrid.from_name(
                area_name, meters=size, window_size=window_size, grid_crs=crs)

        grid.write(output,)

    except:
        logger.error("Error in creating tessellation " + output, exc_info=True)
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
                        help='Prefix for the filename in the form <prefix>_<grid_type>_<cell_size>. By default is grid.',
                        default='grid',
                        type=str)

    parser.add_argument('-a', '--area',
                        action='store',
                        dest='area',
                        help='Area name in the format of "Area name, Country"',
                        default=None,
                        type=str)

    parser.add_argument('-ws', '--window_size',
                        help='Size of the window around the shape centroid.',
                        action='store',
                        dest='window_size',
                        default=None,
                        type=int)

    parser.add_argument('-C', '--crs',
                        help='Coordintate reference system for the output grid.',
                        action='store',
                        dest='crs',
                        default='epsg:4326',
                        type=str)

    parser.add_argument('-b', '--base_shape', action='store',
                        help='Path to the shape file used as a base to build the grid over.',
                        dest='base_shape',
                        default=None,
                        type=str)

    # TODO ADD CRS INPUT SHAPE

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
        logger.setLevel(logging.INFO)

    elif(args.verbosity == 2):
        logger.setLevel(logging.INFO)

    if args.mp == True:
        jobs = []

    for m in args.sizes:

        try:

            # Get the factory according to the tessellation type in input
            if args.mp == True:

                output = os.path.abspath(os.path.join(
                    args.outputfolder, args.prefix + "_" + args.grid + "_" + str(m) + ".geojson"))

                p = multiprocessing.Process(target=write_grid, args=(output, m, args.grid, args.window_size,
                                                                     args.crs, args.area, args.base_shape))
                jobs.append(p)
                p.start()

            else:
                write_grid(output, m, args.grid, args.window_size,
                           args.crs, args.area, args.base_shape)

        except ValueError:
            logger.error("Value error instantiating the grid.", exc_info=True)
            sys.exit(1)

        except TypeError:
            logger.error("Type error building the grid.", exc_info=True)
            sys.exit(1)


if __name__ == "__main__":
    main(sys.argv[1:])
