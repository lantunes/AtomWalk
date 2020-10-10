from pymatgen import MPRester
import pandas as pd
import argparse

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--key', '-k', type=str, help='The materialsproject.org API key')
    parser.add_argument('--out', '-o', action='store',
                        default=None,
                        help='The path to the pickle file to be created.')
    parser.add_argument('--stable', '-s', action='store_true',
                        help='If present, indicates that only stable compounds should be included.')
    args = parser.parse_args()

    client = MPRester(args.key)

    criteria = { "e_above_hull": 0.0} if args.stable else {}

    properties = ['structure', 'band_gap']

    print('Fetching data...')

    result = client.query(criteria, properties)

    print('Converting to dataframe...')

    gaps_data = pd.DataFrame(result)

    print('Pickling...')

    gaps_data.to_pickle(args.out)
