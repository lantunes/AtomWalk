import sys
sys.path.append('../atomwalk')

import argparse
from atomwalk import OneHotVectors


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Create one-hot vectors.')
    parser.add_argument('--out', '-o', type=str,
                        help='The path to the model file to be created.')
    parser.add_argument('--elems', '-e', type=str,
                        help='A comma-separated list of the elements to include (there can be no spaces between commas).')
    args = parser.parse_args()

    elems = [e for e in args.elems.split(",")]

    ohv = OneHotVectors(elems=elems)

    ohv.save(args.out)
