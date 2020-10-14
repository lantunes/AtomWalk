import numpy as np
from sklearn.neural_network import MLPRegressor
from sklearn.model_selection import cross_validate
from glove import Glove
from atomwalk import EigenModel, SVDModel, PCAModel

try:
    import cPickle as pickle
except ImportError:
    import pickle


if __name__ == '__main__':
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

    counts = "../out/all_stable_bandgap_2020_10_09.counts.pkl"
    # model = EigenModel.load(counts, d=20)  # 0.2185
    # model = EigenModel.load(counts, d=30)  # 0.1995
    # model = SVDModel.load(counts, d=20)  # 0.1772
    # model = SVDModel.load(counts, d=30)  # 0.15993
    # model = PCAModel.load(counts, d=20)  # 0.18446
    model = PCAModel.load(counts, d=30)  # 0.1635

    # TODO try using bigger corpus for counts
    # TODO try learning the embeddings using GloVe and the co-occurrence counts

    embeddings = model.word_vectors

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
