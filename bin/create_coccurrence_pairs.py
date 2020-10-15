import sys
sys.path.append('../atomwalk')

import argparse
import warnings
import numpy as np
import pandas as pd
from tqdm import tqdm
import multiprocessing as mp
import gzip
import logging

from skipatom import get_cooccurrence_pairs

warnings.simplefilter("ignore", category=UserWarning)


def listener(queue, filename, zip, n):
    pbar = tqdm(total=n)
    o = gzip.open(filename, "wt") if zip else open(filename, "w")
    with o as f:
        while True:
            message = queue.get()
            if message == "kill":
                break

            for pair in message:
                f.write("%s,%s\n" % pair)

            f.flush()

            pbar.update(1)


def worker(structs, queue, verbose):
    for i in range(len(structs)):
        try:
            queue.put(get_cooccurrence_pairs(structs[i]))
        except Exception as e:
            if verbose:
                logging.warning(e)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', type=str, help='The path to the dataset pickle file.')
    parser.add_argument('--out', '-o', type=str,
                        help='The path to the walks corpus file to be created.')
    parser.add_argument('--zip', '-z', action='store_true',
                        help='If present, indicates that the generated pairs file should be gzipped.')
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

    print("generating pairs...")

    manager = mp.Manager()
    queue = manager.Queue()
    pool = mp.Pool(args.processes)

    watcher = pool.apply_async(listener, (queue, args.out, args.zip, len(df),))

    jobs = []
    for i in range(args.workers):
        chunk = chunks[i]
        job = pool.apply_async(worker, (chunk[:,0], queue, args.verbose))
        jobs.append(job)

    for job in jobs:
        job.get()

    queue.put("kill")
    pool.close()
    pool.join()
