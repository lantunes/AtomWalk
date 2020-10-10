import sys
sys.path.append('../atomwalk')

import argparse
import warnings
import numpy as np
import pandas as pd
from tqdm import tqdm
import multiprocessing as mp
import gzip

from atomwalk import LatticeGraphWalk

warnings.simplefilter("ignore", category=UserWarning)


def listener(queue, filename, zip, n):
    pbar = tqdm(total=n)
    o = gzip.open(filename, "wt") if zip else open(filename, "w")
    with o as f:
        while True:
            message = queue.get()
            if message == "kill":
                break

            for walk in message:
                line = " ".join(walk)
                f.write(line + "\n")

            f.flush()

            pbar.update(1)


def worker(structs, queue, p, q, num_walks, walk_length):
    for i in range(len(structs)):
        struct = structs[i]
        walks = LatticeGraphWalk.walk(struct, p, q, num_walks, walk_length)
        queue.put(walks)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', type=str, help='The path to the dataset pickle file.')
    parser.add_argument('--out', '-o', action='store',
                        default=None,
                        help='The path to the walks corpus file to be created.')
    parser.add_argument('--zip', '-z', action='store_true',
                        help='If present, indicates that the generated walks corpus file should be gzipped.')
    parser.add_argument('--p', type=float, default=1,
                        help='Return parameter. Default is 1.')
    parser.add_argument('--q', type=float, default=1,
                        help='In-out parameter. Default is 1.')
    parser.add_argument('--walk-length', type=int, default=40,
                        help='Length of walk per source. Default is 80.')
    parser.add_argument('--num-walks', type=int, default=10,
                        help='Number of walks per source. Default is 10.')
    parser.add_argument('--workers', type=int, default=1,
                        help='The number of worker processes to use. Default is 1.')
    args = parser.parse_args()

    print("reading pickle file...")
    df = pd.read_pickle(args.data)

    chunks = np.array_split(np.array(df), args.workers)

    print("generating walks...")

    manager = mp.Manager()
    queue = manager.Queue()
    pool = mp.Pool(mp.cpu_count() + 2)

    watcher = pool.apply_async(listener, (queue, args.out, args.zip, len(df),))

    jobs = []
    for i in range(args.workers):
        chunk = chunks[i]
        job = pool.apply_async(worker, (chunk[:,0], queue, args.p, args.q, args.num_walks, args.walk_length))
        jobs.append(job)

    for job in jobs:
        job.get()

    queue.put("kill")
    pool.close()
    pool.join()
