from pymatgen.analysis.graphs import StructureGraph
from pymatgen.analysis.local_env import CrystalNN
import numpy as np
from numpy.linalg import svd
import scipy
from sklearn.decomposition import PCA

try:
    import cPickle as pickle
except ImportError:
    import pickle


class LatticeCorpus:

    @staticmethod
    def count_cooccurrences_single(struct):
        counts = {}
        struct_graph = StructureGraph.with_local_env_strategy(struct, CrystalNN())
        labels = {i: spec.name for i, spec in enumerate(struct.species)}
        G = struct_graph.graph.to_undirected()
        for n in labels:
            target = labels[n]
            neighbors = [labels[i] for i in G.neighbors(n)]
            for neighbor in neighbors:
                key = frozenset([target, neighbor])
                if key not in counts:
                    counts[key] = 0
                counts[key] += 1
        return counts

    @staticmethod
    def merge(counts, new_counts):
        for key in new_counts:
            if key not in counts:
                counts[key] = new_counts[key]
            else:
                counts[key] += new_counts[key]
        return counts

    @staticmethod
    def to_cooccurrence_count_matrix(counts):
        atoms = set()
        for key in counts:
            for a in key:
                atoms.add(a)

        dictionary = {a: i for i, a in enumerate(sorted(atoms))}

        M = np.zeros(shape=(len(atoms), len(atoms)), dtype=np.float)

        for key in counts:
            elements = list(key)
            if len(elements) == 1:
                element = elements[0]
                M[dictionary[element]][dictionary[element]] = counts[key]
            else:
                e1 = elements[0]
                e2 = elements[1]
                M[dictionary[e1]][dictionary[e2]] = counts[key]
                M[dictionary[e2]][dictionary[e1]] = counts[key]

        return M, dictionary


class CooccurrenceCountModel:
    def __init__(self, counts):
        self.word_vectors, self.dictionary = LatticeCorpus.to_cooccurrence_count_matrix(counts)

    @staticmethod
    def load(counts_filename):
        with open(counts_filename, "rb") as f:
            counts = pickle.load(f)
        return CooccurrenceCountModel(counts)


class EigenModel:
    def __init__(self, counts, d):
        M, self.dictionary = LatticeCorpus.to_cooccurrence_count_matrix(counts)
        eigenvalues, eigenvectors = scipy.linalg.eigh(M)
        first_d_eigenvectors = eigenvectors[:, :d]  # first_d_eigenvectors is a len(M)xd matrix
        self.word_vectors = first_d_eigenvectors

    @staticmethod
    def load(counts_filename, d):
        with open(counts_filename, "rb") as f:
            counts = pickle.load(f)
        return EigenModel(counts, d)


class SVDModel:
    def __init__(self, counts, d):
        M, self.dictionary = LatticeCorpus.to_cooccurrence_count_matrix(counts)
        U, S, V = svd(M)
        self.word_vectors = U[:, :d]

    @staticmethod
    def load(counts_filename, d):
        with open(counts_filename, "rb") as f:
            counts = pickle.load(f)
        return SVDModel(counts, d)


class PCAModel:
    def __init__(self, counts, d):
        M, self.dictionary = LatticeCorpus.to_cooccurrence_count_matrix(counts)
        pca = PCA(n_components=d)
        self.word_vectors = pca.fit(M)

    @staticmethod
    def load(counts_filename, d):
        with open(counts_filename, "rb") as f:
            counts = pickle.load(f)
        return SVDModel(counts, d)
