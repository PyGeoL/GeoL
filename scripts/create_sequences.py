"""
Script to create grid(s), given input args.
"""

# Authors: Gianni Barlacchi <gianni.barlacchi@gmail.com>

import argparse
import sys
import logging
import os
from geol.representations.pois_sequences import POISequences
from geol.geometry.squaregrid import SquareGrid
from geol.geol_logger.geol_logger import logger


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

    parser.add_argument('-i', '--input',
                        help='Input file with point-of-interests. NOTE: in the case of strategy=nearest|alphabetically, the input file must contains the column cellID.',
                        action='store',
                        dest='inputfile',
                        required=True,
                        type=str)

    parser.add_argument('-g', '--grid',
                        help='Input grid. This is mandatory in the case of strategy=nearest.',
                        action='store',
                        dest='inputgrid',
                        type=str)

    parser.add_argument('-p', '--prefix',
                        action='store',
                        dest='prefix',
                        help='Prefix for the filename. By the default is <prefix>_<strategy>, by default is sequences.',
                        default='sequences',
                        type=str)

    parser.add_argument('-sp', '--sep',
                        action='store',
                        dest='sep',
                        help='Separator for reading csv files. Defaults to TAB',
                        default='\t',
                        type=str)

    parser.add_argument('-s', '--strategy',
                        help='Strategy to use: 1 (alphabetically, 2 (nearest), 3 (distance). Default 1.',
                        action='store',
                        dest='strategy',
                        default=1,
                        type=int)

    parser.add_argument('-b', '--band_size',
                        help='Size of the band size, required only if strategy=distance.',
                        action='store',
                        dest='band_size',
                        default=500,
                        type=int)

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

    sequences_generator = POISequences.from_csv(args.inputfile, args.sep)

    if(args.strategy == 1):
        strStrategy = "alphabetically"
        output = os.path.abspath(os.path.join(
            args.outputfolder, args.prefix + "_" + str(strStrategy) + ".txt"))
        sequences_generator.alphabetically_sequence(output)
    elif (args.strategy == 2):

        if (args.inputgrid is None):
            raise ValueError(
                "In the case of strategy=nearest the input grid is mandatory.")

        strStrategy = "nearest"
        output = os.path.abspath(os.path.join(
            args.outputfolder, args.prefix + "_" + str(strStrategy) + ".txt"))
        sequences_generator.nearest_based_sequence(output, args.inputgrid)

    elif (args.strategy == 3):
        strStrategy = "distance"
        output = os.path.abspath(os.path.join(
            args.outputfolder, args.prefix + "_" + str(strStrategy) + ".txt"))
        sequences_generator.distance_based_sequence(args.band_size, output)
    else:
        raise ValueError(
            "Please, check the parameters as no valid configurations have been found.")


if __name__ == "__main__":
    main(sys.argv[1:])
