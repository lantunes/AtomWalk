import argparse
import gzip
from glove import Glove, Corpus


def read_corpus(filename):
    delchars = [chr(c) for c in range(256)]
    delchars = [x for x in delchars if not x.isalnum()]
    delchars.remove(' ')
    delchars = ''.join(delchars)
    o = gzip.open(filename, 'r') if filename.endswith(".gz") else open(filename, 'r')
    with o as datafile:
        for line in datafile:
            lower_line = line.lower()
            table = lower_line.maketrans(dict.fromkeys(delchars))
            yield lower_line.translate(table).split(' ')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Fit a GloVe model.')
    parser.add_argument('--corpus', '-c', action='store',
                        default=None,
                        help=('The filename of the corpus to pre-process. The file may be gzipped, and if so, it must '
                              'have a filename that ends with .gz.'
                              'The pre-processed corpus will be saved and will be ready for training.'))
    parser.add_argument('--out', '-o', action='store',
                        default=None,
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
    parser.add_argument('--window', '-w', action='store', type=int,
                        default=10,
                        help='The length of the (symmetric) context window used for co-occurrence.')
    args = parser.parse_args()

    print('Pre-processing corpus')
    corpus_model = Corpus()
    corpus_model.fit(read_corpus(args.corpus), window=args.window)
    corpus_model.save('%s.corpus.model' % args.out)
    print('Dict size: %s' % len(corpus_model.dictionary))
    print('Collocations: %s' % corpus_model.matrix.nnz)

    print('Training the GloVe model')
    glove = Glove(no_components=args.components, learning_rate=args.learning_rate)
    glove.fit(corpus_model.matrix, epochs=int(args.train),
              no_threads=args.parallelism, verbose=True)
    glove.add_dictionary(corpus_model.dictionary)
    glove.save('%s.glove.model' % args.out)
