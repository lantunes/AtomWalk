from gensim.models import Word2Vec
import multiprocessing
import argparse
import logging
import gzip
logging.basicConfig(format="%(levelname)s - %(asctime)s: %(message)s", datefmt= '%H:%M:%S', level=logging.INFO)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Fit a Word2Vec model.')
    parser.add_argument('--corpus', '-c', action='store',
                        type=str,
                        help=('The filename of the corpus to pre-process. The file may be gzipped, and if so, it must '
                              'have a filename that ends with .gz.'))
    parser.add_argument('--out', '-o', action='store',
                        type=str,
                        help='The path to the model file to be created.')
    parser.add_argument('--train', '-t', action='store', type=int,
                        default=10,
                        help='Train the model with this number of epochs.')
    parser.add_argument('--parallelism', '-p', action='store', type=int,
                        default=multiprocessing.cpu_count()-1,
                        help='Number of parallel threads to use for training.')
    parser.add_argument('--components', '-s', action='store', type=int,
                        default=100,
                        help='Number of components in the learned vectors.')
    parser.add_argument('--window', '-w', action='store', type=int,
                        default=10,
                        help='The length of the (symmetric) context window used for co-occurrence.')
    parser.add_argument('--alpha', '-a', action='store', type=float,
                        default=0.025,
                        help='The initial learning rate.')
    parser.add_argument('--min_alpha', '-m', action='store', type=float,
                        default=0.0001,
                        help='Learning rate will linearly drop to `min_alpha` as training progresses.')
    parser.add_argument('--batch', '-b', action='store', type=int,
                        default=10000,
                        help=('Target size (in words) for batches of examples passed to worker threads (and'
                              'thus cython routines).(Larger batches will be passed if individual'
                              'texts are longer than 10000 words, but the standard cython code truncates to that maximum.)'))
    args = parser.parse_args()

    sentences = []
    o = gzip.open(args.corpus, 'rt') if args.corpus.endswith(".gz") else open(args.corpus, 'rt')
    with o as f:
        for line in f.readlines():
            sentences.append(line.strip().split(" "))

    w2v = Word2Vec(
        size=args.components,
        window=args.window,
        sg=1,  # use Skip-gram
        workers=args.parallelism,
        alpha=args.alpha,
        min_alpha=args.min_alpha,
        batch_words=args.batch
    )

    w2v.build_vocab(
        sentences=sentences,
        progress_per=10000
    )

    w2v.train(
        sentences=sentences,
        total_words=w2v.corpus_total_words,
        epochs=args.train
    )

    w2v.save(args.out)
