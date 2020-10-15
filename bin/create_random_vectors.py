import sys
sys.path.append('../atomwalk')

import argparse
from atomwalk import RandomVectors


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Create random vectors.')
    parser.add_argument('--out', '-o', type=str,
                        help='The path to the model file to be created.')
    parser.add_argument('--elems', '-e', type=str,
                        help='A comma-separated list of the elements to include (there can be no spaces between commas).')
    parser.add_argument('--dim', '-d', type=int,
                        help='The number of embedding dimensions.')
    parser.add_argument('--mean', '-m', type=float, default=0.0,
                        help='The mean.')
    parser.add_argument('--std', '-s', type=float, default=1.0,
                        help='The standard deviation.')
    args = parser.parse_args()

    elems = [e for e in args.elems.split(",")]

    rv = RandomVectors(elems=elems, dim=args.dim, mean=args.mean, std=args.std)

    rv.save(args.out)
