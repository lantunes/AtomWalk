import sys
sys.path.append('../atomwalk')

import argparse
import warnings
import numpy as np
import pandas as pd
from tqdm import tqdm
import multiprocessing as mp
import logging

from atomwalk import LatticeCorpus

try:
    import cPickle as pickle
except ImportError:
    import pickle

warnings.simplefilter("ignore", category=UserWarning)


def listener(queue, n):
    pbar = tqdm(total=n)
    counts = {}
    while True:
        message = queue.get()
        if message == "kill":
            break

        counts = LatticeCorpus.merge(counts, message)

        pbar.update(1)
    return counts


def worker(structs, queue, verbose):
    for i in range(len(structs)):
        try:
            queue.put(LatticeCorpus.count_cooccurrences_single(structs[i]))
        except Exception as e:
            if verbose:
                logging.warning(e)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', type=str, help='The path to the dataset pickle file.')
    parser.add_argument('--out', type=str, help='The path to the counts corpus file to be created.')
    parser.add_argument('--processes', type=int, default=mp.cpu_count(),
                        help='The number of processes to create. Default is the CPU count.')
    parser.add_argument('--workers', type=int, default=1,
                        help='The number of worker processes to use. Default is 1.')
    parser.add_argument('--verbose', '-v', action='store_true',
                        help='If present, warnings will be logged.')
    args = parser.parse_args()

    print("reading pickle file...")
    df = pd.read_pickle(args.data)

    chunks = np.array_split(np.array(df), args.workers)

    print("counting co-occurrences...")

    manager = mp.Manager()
    queue = manager.Queue()
    pool = mp.Pool(args.processes)

    watcher = pool.apply_async(listener, (queue, len(df)))

    jobs = []
    for i in range(args.workers):
        chunk = chunks[i]
        job = pool.apply_async(worker, (chunk[:, 0], queue, args.verbose))
        jobs.append(job)

    for job in jobs:
        job.get()

    queue.put("kill")
    pool.close()
    pool.join()

    with open(args.out, 'wb') as pickle_file:
        pickle.dump(watcher.get(), pickle_file, protocol=pickle.HIGHEST_PROTOCOL)
