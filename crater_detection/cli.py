"""
The command line interface
"""
import argparse
import os
import sys
from scipy import misc
from . import __version__
from . import detector, generator
from .util import logger


def run_detector(args):
    _, image_filename = os.path.split(args.input)
    input_image = misc.imread(args.input)
    output_image, crater_field = detector.detect(input_image)

    if args.output is not None:
        out_filename = args.output
    else:
        out_filename = f'output-{image_filename}'

    misc.imsave(out_filename, output_image)
    logger.info('Done! Saved to:', out_filename, color='green')

    stats = crater_field.stats()
    logger.info("Crater stats:", color='green')
    logger.info("Width:", stats["width"])
    logger.info("Height:", stats["height"])
    logger.info("Number of Craters:", stats["num_craters"])
    logger.info("Max Radius:", stats["max_rad"])
    logger.info("Min Radius:", stats["min_rad"])
    logger.info("Average Radius:", stats["mean_rad"])
    logger.info("Average Sun Angle (degrees):", stats["sun_angle_degrees"])

    if args.display_output:
        misc.imshow(output_image)


def detector_error_handler(ex, args):
    if type(ex) == FileNotFoundError:
        logger.error("Can't load image file: " + ex.filename)
        sys.exit(1)
    else:
        if args.debug:
            raise ex  # For Development
        logger.error('Error:' + args.input)
        sys.exit(1)


def run_generator(args):
    output_image, stats = generator.generate(
        num_craters=args.num_craters,
        width=args.width,
        height=args.height,
        min_radius=args.min_rad,
        max_radius=args.max_rad,
        shadow_factor=args.shadow_factor,
        alpha=args.alpha,
        sun_angle=args.angle,
        rand_seed=args.rand_seed,
    )

    if args.display_output:
        misc.imshow(output_image)

    if args.output is not None:
        out_filename = args.output
    else:
        out_filename = 'craters.png'

    misc.imsave(out_filename, output_image)
    logger.info('Done!', color='green')

    logger.info('Generated field stats:')
    logger.info("Width:", stats["width"])
    logger.info("Height:", stats["height"])
    logger.info("Number of Craters:", stats["num_craters"])
    logger.info("Max Radius:", stats["max_rad"])
    logger.info("Min Radius:", stats["min_rad"])
    logger.info("Average Radius:", stats["mean_rad"])
    logger.info("Sun Angle (degrees):", stats["sun_angle_degrees"])
    logger.info("Shadow Factor:", stats["shadow_factor"])
    logger.info("Alpha:", stats["alpha"])


def generator_error_handler(ex, args):
    if args.debug:
        raise ex  # For Development
    logger.error('Error generating crater.')
    sys.exit(1)


def add_common_args(parser):
    parser.add_argument('-v', '--verbose', help="Printouts?", dest='verbose', action='store_true')
    parser.set_defaults(verbose=False)

    parser.add_argument('-D', '--debug', help="Debug mode?", dest='debug', action='store_true')
    parser.set_defaults(debug=False)

    parser.add_argument('-d', '--display', help="Display output?", dest='display_output', action='store_true')
    parser.set_defaults(display=False)

    parser.add_argument('-o', '--output', help="The destination to write output image / files.",
                        type=str,
                        required=False,
                        default=None)


def main():
    """The exported main function
    :return:
    """
    parser = argparse.ArgumentParser(description='Lunar Crater Detection')
    parser.add_argument('-V', '--version', action='version', version=__version__)
    parser.set_defaults(cmd=None)
    parser.set_defaults(error_handler=None)

    # Sub parsers
    subparsers = parser.add_subparsers(help='Crater functions.')

    detection_parser = subparsers.add_parser('detect', description='To detect craters in an image.')
    detection_parser.set_defaults(cmd=run_detector)
    detection_parser.set_defaults(error_handler=detector_error_handler)

    detection_parser.add_argument('-i', '--input', help="The input image to detect.", type=str, required=True)

    add_common_args(detection_parser)

    generate_parser = subparsers.add_parser('generate', description='To detect craters in an image.')
    generate_parser.set_defaults(cmd=run_generator)
    generate_parser.set_defaults(error_handler=generator_error_handler)

    generate_parser.add_argument('--width',
                                 help="Output width (px)",
                                 default=generator.FieldX,
                                 type=int)
    generate_parser.add_argument('--height',
                                 help="Output height (px)",
                                 default=generator.FieldY,
                                 type=int)
    generate_parser.add_argument('-n', '--num-craters',
                                 help="Number of craters to draw.",
                                 default=generator.NCraters,
                                 type=int)
    generate_parser.add_argument('--min-rad',
                                 help="Min radius of a crater (px).",
                                 default=generator.MinCrater,
                                 type=int)
    generate_parser.add_argument('--max-rad',
                                 help="Max radius of a crater (px).",
                                 default=generator.MaxCrater,
                                 type=int)
    generate_parser.add_argument('--shadow-factor',
                                 help="Shadow factor.",
                                 default=generator.CraterShadowFactor,
                                 type=int)
    generate_parser.add_argument('-ca', '--alpha',
                                 help="Crater Alpha.",
                                 default=generator.Alpha,
                                 type=int)
    generate_parser.add_argument('-rs', '--rand-seed',
                                 help="Random seed for the number generator.",
                                 default=None,
                                 type=int)
    generate_parser.add_argument('-a', '--angle',
                                 help="Angle of sun (degrees).",
                                 default=generator.SunAngle,
                                 type=int)

    add_common_args(generate_parser)

    args = parser.parse_args()

    if args.cmd is None:
        parser.print_usage()
        sys.exit(1)

    logger.set_enabled(args.verbose)

    try:
        # Main
        args.cmd(args)
    except KeyboardInterrupt:
        logger.error('Quitting before detection finished!')
        sys.exit(0)
    except Exception as ex:
        if args.debug:
            raise ex  # For Development
        args.error_handler(ex, args)


if __name__ == '__main__':
    main()
