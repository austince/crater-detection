"""
The command line interface
"""
import argparse
import os
import sys
from scipy import misc
from termcolor import cprint
from . import __version__
from .lunar_detection import detector
from .util import logger


def main():
    """The exported main function
    :return:
    """
    parser = argparse.ArgumentParser(description='Lunar Crater Detection')
    parser.add_argument('-v', '--version', action='version', version=__version__)

    parser.add_argument('-vb', '--verbose', help="Printouts?", dest='verbose', action='store_true')
    parser.set_defaults(verbose=False)

    parser.add_argument('-do', '--display-output', help="Display output?", dest='display_output', action='store_true')
    parser.set_defaults(verbose=False)

    parser.add_argument('-i', '--input', help="The input image to detect.", type=str, required=True)
    parser.add_argument('-o', '--output', help="The destination to write output image / files.",
                        type=str,
                        required=False,
                        default=None)

    args = parser.parse_args()

    logger.set_enabled(args.verbose)

    try:
        _, image_filename = os.path.split(args.input)
        input_image = misc.imread(args.input)
        output_image, craters = detector.detect(input_image)
    except FileNotFoundError as ex:
        cprint("Can't load image file: " + ex.filename, 'red')
        sys.exit(1)
    except KeyboardInterrupt:
        cprint('Quitting before detection finished!', 'red')
        sys.exit(0)
    except Exception as ex:
        raise ex  # For Development
        logger.log('Error detecting ' + args.input + ": " + str(ex), color='red')
        sys.exit(1)

    if args.output is not None:
        out_filename = args.output
    else:
        out_filename = 'output-%s' % image_filename

    misc.imsave(out_filename, output_image)
    logger.log('Done!', color='green')

    if args.display_output:
        # print(craters)
        logger.log("Number of Craters:", len(craters))
        misc.imshow(output_image)


if __name__ == '__main__':
    main()
