from pymatgen.analysis.graphs import StructureGraph
from pymatgen.analysis.local_env import CrystalNN
import numpy as np
from numpy.linalg import svd
import scipy
from sklearn.decomposition import PCA
from glove import Corpus

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
    def to_cooccurrence_count_matrix(counts, p=None):
        """
        :param counts: the co-occurrence count dictionary
        :param p: normalization factor; if None, no normalization will be performed
                  See: Zhou, Quan, et al. "Learning atoms for materials discovery." Proceedings of the National Academy
                  of Sciences 115.28 (2018): E6411-E6417
        :return: the co-occurrence count matrix and the associated dictionary of atom name to its index
        """
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

        if p:
            M = M / (np.sum(M, axis=1).reshape(-1, 1)**p) ** (1 / p)

        return M, dictionary


class CooccurrenceCountModel:
    def __init__(self, counts, p=None):
        self.vectors, self.dictionary = LatticeCorpus.to_cooccurrence_count_matrix(counts, p)

    @staticmethod
    def load(counts_filename, p=None):
        with open(counts_filename, "rb") as f:
            counts = pickle.load(f)
        return CooccurrenceCountModel(counts, p)


class EigenModel:
    def __init__(self, counts, d, p=None):
        M, self.dictionary = LatticeCorpus.to_cooccurrence_count_matrix(counts, p)
        eigenvalues, eigenvectors = scipy.linalg.eigh(M)
        first_d_eigenvectors = eigenvectors[:, :d]  # first_d_eigenvectors is a len(M)xd matrix
        self.vectors = first_d_eigenvectors

    @staticmethod
    def load(counts_filename, d, p=None):
        with open(counts_filename, "rb") as f:
            counts = pickle.load(f)
        return EigenModel(counts, d, p)


class SVDModel:
    def __init__(self, counts, d, p=None):
        M, self.dictionary = LatticeCorpus.to_cooccurrence_count_matrix(counts, p)
        U, S, V = svd(M)
        self.vectors = U[:, :d]

    @staticmethod
    def load(counts_filename, d, p=None):
        with open(counts_filename, "rb") as f:
            counts = pickle.load(f)
        return SVDModel(counts, d, p)


class SVDModelFromGloVeCorpus:
    def __init__(self, M, dictionary, d, p=None):
        self.dictionary = dictionary
        if p:
            M = M / (np.sum(M, axis=1).reshape(-1, 1) ** p) ** (1 / p)
            M = np.nan_to_num(M)  # if there are rows with all zeros, then nan values will result
        U, S, V = svd(M)
        self.vectors = U[:, :d]

    @staticmethod
    def load(glove_corpus_filename, d, p=None):
        corpus_model = Corpus.load(glove_corpus_filename)
        M = corpus_model.matrix.todense()  # an upper triangular matrix with diagonal values of zero
        M = M + M.T  # convert to a symmetric matrix
        return SVDModelFromGloVeCorpus(np.asarray(M), corpus_model.dictionary, d, p)


class PCAModel:
    def __init__(self, counts, d, p=None):
        M, self.dictionary = LatticeCorpus.to_cooccurrence_count_matrix(counts, p)
        pca = PCA(n_components=d)
        self.vectors = pca.fit(M)

    @staticmethod
    def load(counts_filename, d, p=None):
        with open(counts_filename, "rb") as f:
            counts = pickle.load(f)
        return SVDModel(counts, d, p)
