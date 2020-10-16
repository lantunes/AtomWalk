import sys
sys.path.append('../atomwalk')

import argparse
from glove import Glove
from atomwalk import LatticeCorpus
from scipy.sparse import coo_matrix

try:
    import cPickle as pickle
except ImportError:
    import pickle


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Fit a GloVe model.')
    parser.add_argument('--counts', '-c', action='store',
                        help='The filename of the counts pickle file.')
    parser.add_argument('--out', '-o', action='store',
                        help='The path to the model file to be created (without the file extension).')
    parser.add_argument('--train', '-t', action='store',
                        default=10,
                        help='Train the GloVe model with this number of epochs.')
    parser.add_argument('--learning_rate', '-l', action='store',
                        type=float, default=0.05,
                        help='Train the GloVe model with this learning rate.')
    parser.add_argument('--parallelism', '-p', action='store',
                        default=1,
                        help='Number of parallel threads to use for training.')
    parser.add_argument('--components', '-s', action='store', type=int,
                        default=100,
                        help='Number of components in the learned vectors.')
    parser.add_argument('--max_count', '-m', action='store', type=int,
                        default=100,
                        help='The max co-occurrence count.')
    args = parser.parse_args()

    with open(args.counts, "rb") as f:
        counts = pickle.load(f)
    M, dictionary = LatticeCorpus.to_cooccurrence_count_matrix(counts)

    print('Training the GloVe model')
    """
    NOTE: It appears that the Glove class will take into account self-co-occurrence counts if there are non-zero values
    in the diagonal of the co-occurrence count matrix.
    """
    glove = Glove(no_components=args.components, learning_rate=args.learning_rate, max_count=args.max_count)
    glove.fit(coo_matrix(M), epochs=int(args.train),
              no_threads=args.parallelism, verbose=True)
    glove.add_dictionary(dictionary)
    glove.save('%s.glove.model' % args.out)
