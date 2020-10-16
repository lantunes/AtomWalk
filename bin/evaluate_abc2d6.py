import numpy as np
from sklearn.neural_network import MLPRegressor
from sklearn.model_selection import cross_validate
from glove import Glove
from atomwalk import SVDModel, OneHotVectors, RandomVectors, SVDModelFromGloVeCorpus
from skipatom import SkipAtomModel, Word2VecModel

try:
    import cPickle as pickle
except ImportError:
    import pickle


if __name__ == '__main__':
    # Note: these GloVe models don't take into account self-interactions
    # model_file = "../out/all_bandgap_2020_10_09_p1_q1_walk10_len40.dim100.glove.model"
    # model_file = "../out/all_bandgap_2020_10_09_p1_q1_walk10_len40.dim20.glove.model"
    # model_file = "../out/all_stable_bandgap_2020_10_09_p1_q1_walk2_len10.dim20.glove.model"
    # model_file = "../out/all_stable_bandgap_2020_10_09_p1_q05_walk10_len40.dim20.glove.model"
    # model_file = "../out/all_stable_bandgap_2020_10_09_p4_q1_walk10_len40.dim20.glove.model"
    # model_file = "../out/all_stable_bandgap_2020_10_09_p4_q1_walk10_len40.dim20_c1.glove.model"
    # model_file = "../out/all_stable_bandgap_2020_10_09_p4_q1_walk10_len40.dim20_c1_mc1000.glove.model"
    # model_file = "../out/all_stable_bandgap_2020_10_09_p4_q1_walk10_len40.dim20_c1_mc10.glove.model"
    # model_file = "../out/all_stable_bandgap_2020_10_09_p1_q2_walk10_len40.dim20.glove.model"
    # model_file = "../out/all_stable_bandgap_2020_10_09_p1_q1_walk10_len40.dim20.glove.model"
    # model_file = "../out/all_stable_bandgap_2020_10_09_p1_q1_walk10_len40.dim20_c1.glove.model"
    # model_file = "../out/all_stable_bandgap_2020_10_09_p1_q1_walk10_len40.dim20_c2.glove.model"
    # model_file = "../out/all_stable_bandgap_2020_10_09_p1_q1_walk10_len40.dim20_c20.glove.model"
    # model = Glove.load(model_file)
    # embeddings = model.word_vectors

    # counts = "../out/all_stable_bandgap_2020_10_09.counts.pkl"
    # model = SVDModel.load(counts, d=20)       # 0.177889 +/- 0.012315
    # model = SVDModel.load(counts, d=20, p=2)  # 0.17523 +/- 0.013991
    # model = SVDModel.load(counts, d=30)       # 0.15844 +/- 0.0090904
    # model = SVDModel.load(counts, d=30, p=2)  # 0.144285 +/- 0.0077560
    # embeddings = model.vectors

    # counts = "../out/all_bandgap_2020_10_09.counts.pkl"
    # model = SVDModel.load(counts, d=30)       # 0.146637 +/- 0.0092159
    # model = SVDModel.load(counts, d=30, p=1)  # 0.149213 +/- 0.006462
    # model = SVDModel.load(counts, d=30, p=2)  # 0.153990 +/- 0.005654
    # model = SVDModel.load(counts, d=30, p=3)    # 0.152144 +/- 0.0037654
    # embeddings = model.vectors

    # one_hot = "../out/one_hot_atom2vec.model"  # 0.13927 +/- 0.00701 (dim=33)
    # model = OneHotVectors.load(one_hot)
    # embeddings = model.vectors

    # rv = "../out/random_atom2vec_dim20.model"  # 0.22476 +/- 0.010051
    # rv = "../out/random_atom2vec_dim30.model"  # 0.18327 +/- 0.007466
    # model = RandomVectors.load(rv)
    # embeddings = model.vectors

    # pairs = "../out/all_stable_bandgap_2020_10_09.pairs.dim20.model"  # 0.1494024 +/- 0.00506478
    # pairs = "../out/all_stable_bandgap_2020_10_09.pairs.dim30.model"  # 0.1378849 +/- 0.0054404
    # td = "../out/all_stable_bandgap_2020_10_09.pairs.training.data"
    # pairs = "../out/all_bandgap_2020_10_09.pairs.dim20.model"  # 0.148544 +/- 0.0060257
    # pairs = "../out/all_bandgap_2020_10_09.pairs.dim30.model"  # 0.13428 +/- 0.006687
    # td = "../out/all_bandgap_2020_10_09.pairs.training.data"
    # model = SkipAtomModel.load(pairs, td)
    # embeddings = model.vectors

    # model_file = "../out/all_stable_bandgap_2020_10_09.counts.dim20.glove.model"  # 0.2795177 +/- 0.021016494
    # model_file = "../out/all_stable_bandgap_2020_10_09.counts.dim30.glove.model"  # 0.270738 +/- 0.022560
    # model = Glove.load(model_file)
    # embeddings = model.word_vectors

    # corpus_model = "../out/all_stable_bandgap_2020_10_09_p1_q1_walk10_len40.dim20.corpus.model"
    # model = SVDModelFromGloVeCorpus.load(corpus_model, d=20)  # 0.163633 +/- 0.0102817
    # model = SVDModelFromGloVeCorpus.load(corpus_model, d=20, p=2)  # 0.157051 +/- 0.005453
    # model = SVDModelFromGloVeCorpus.load(corpus_model, d=30)  # 0.149208 +/- 0.007384
    # model = SVDModelFromGloVeCorpus.load(corpus_model, d=30, p=2)  # 0.14084955 +/- 0.00754577
    # embeddings = model.vectors

    # corpus_model = "../out/all_bandgap_2020_10_09_p1_q1_walk10_len40.dim20.corpus.model"
    # model = SVDModelFromGloVeCorpus.load(corpus_model, d=20)  # 0.160385 +/- 0.009316
    # model = SVDModelFromGloVeCorpus.load(corpus_model, d=20, p=2)  # 0.16224140 +/- 0.00850218
    # model = SVDModelFromGloVeCorpus.load(corpus_model, d=30)  # 0.1421715 +/- 0.0037470
    # model = SVDModelFromGloVeCorpus.load(corpus_model, d=30, p=2)  # 0.1447200 +/- 0.00942118
    # embeddings = model.vectors

    model_file = "../out/all_stable_bandgap_2020_10_09_p4_q1_walk10_len40.dim30.word2vec.model"  # 0.174962 +/- 0.008083
    model = Word2VecModel.load(model_file)
    embeddings = model.vectors

    with open("../out/abc2d6_training.pkl", 'rb') as pickle_file:
        X, y = pickle.load(pickle_file)

    regression = MLPRegressor(hidden_layer_sizes=(10,), activation="relu", max_iter=1000, early_stopping=True)

    X_in = []
    for x in X:
        atoms = x.split(" ")
        vectors = [[]]*4
        for atom in set(atoms):
            count = atoms.count(atom)
            if count == 6:  # D
                vectors[3] = embeddings[model.dictionary[atom]]
            elif count == 2:  # C
                vectors[2] = embeddings[model.dictionary[atom]]
            else:
                if vectors[0] == []:  # A or B TODO not sure how to determine A vs B in ABC2D6, does it matter?
                    vectors[0] = embeddings[model.dictionary[atom]]
                else:
                    vectors[1] = embeddings[model.dictionary[atom]]
        X_in.append(np.concatenate(vectors))

    cv_results = cross_validate(regression, X_in, y, cv=10, return_estimator=True,
                                scoring=('neg_mean_absolute_error', 'neg_root_mean_squared_error'))
    print(cv_results['test_neg_mean_absolute_error'])
    print(cv_results['test_neg_root_mean_squared_error'])

    print("mean fold neg_mean_absolute_error score: %s" % np.mean(cv_results['test_neg_mean_absolute_error']))
    print("std fold neg_mean_absolute_error score: %s" % np.std(cv_results['test_neg_mean_absolute_error']))
    print("mean fold neg_root_mean_squared_error score: %s" % np.mean(cv_results['test_neg_root_mean_squared_error']))
    print("std fold neg_root_mean_squared_error score: %s" % np.std(cv_results['test_neg_root_mean_squared_error']))
