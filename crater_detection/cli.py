"""
The command line interface
"""
import argparse
import os
import sys
import cv2
from scipy import misc
from termcolor import cprint
from . import __version__


def main():
    """The exported main function
    :return:
    """
    parser = argparse.ArgumentParser(description='Lunar Crater Detection')
    parser.add_argument('-v', '--version', action='version', version=__version__)

    parser.add_argument('-vb', '--verbose', help="Printouts?", dest='verbose', action='store_true')
    parser.set_defaults(verbose=False)

    # Optional to test against an input image instead of default set
    parser.add_argument('-i', '--input', help="The input image to detect.", type=str, required=True)

    args = parser.parse_args()

    try:
        _, image_filename = os.path.split(args.input)
        input_image = misc.imread(args.input)
    except FileNotFoundError as ex:
        cprint("Can't load image file: " + ex.filename, 'red')
        sys.exit(1)
    except KeyboardInterrupt:
        cprint('Quitting before detection finished!', 'red')
        sys.exit(0)
    except Exception as ex:
        # raise ex  # For Development
        cprint('Error detecting ' + args.input + ": " + str(ex), 'red')

        sys.exit(1)

    if args.verbose:
        cprint('Done!', 'green')


if __name__ == '__main__':
    main()
