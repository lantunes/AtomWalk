import numpy as np
import pandas as pd
from skipatom import SkipAtomModel
from sklearn.neural_network import MLPRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import cross_validate
import warnings

warnings.simplefilter("ignore", category=UserWarning)


if __name__ == '__main__':

    # max pooling:  RF: 0.816905 +/- 0.01166; MLP: 0.736002 +/- 0.0113929
    # mean pooling: RF: 0.801473 +/- 0.01158; MLP: 0.801464 +/- 0.0081636
    # model = SkipAtomModel.load("../out/all_bandgap_2020_10_09.pairs.dim20.model",
    #                            "../out/all_bandgap_2020_10_09.pairs.training.data")

    # max pooling:  RF: 0.819178 +/- 0.00760; MLP: 0.802496 +/- 0.011010
    # mean pooling: RF: 0.811296 +/- 0.01055; MLP: 0.855942 +/- 0.018847
    model = SkipAtomModel.load("../out/all_bandgap_2020_10_09.pairs.dim89.model",
                               "../out/all_bandgap_2020_10_09.pairs.training.data")

    embeddings = model.vectors

    df = pd.read_pickle("../out/all_stable_bandgap_2020_10_09.pkl.gz")

    # regression, args = RandomForestRegressor, {"n_estimators": 100, "n_jobs": 4}
    regression, args = MLPRegressor, {"hidden_layer_sizes": (100,), "max_iter": 500}

    pool = np.mean
    # pool = np.max

    exclude_zero = False
    # exclude_zero = True

    X = []
    y = []
    for i in range(len(df['structure'])):
        struct = df['structure'][i]
        band_gap = df['band_gap'][i]

        if band_gap == 0.0 and exclude_zero:
            continue

        # TODO temporary hack
        if any(s in struct.formula for s in ("Ne", "He", "Ar")):
            print("skipping: %s" % struct.formula)
            continue
        #

        vectors = []
        for element in struct.species:
            vectors.append(np.array(embeddings[model.dictionary[element.name]]))
        X.append(pool(vectors, axis=0))
        y.append(band_gap)

    cv_results = cross_validate(regression(**args), X, y, cv=10, return_estimator=True,
                                scoring=('r2', 'neg_root_mean_squared_error'))
    # the r2 score is the coefficient of determination, R^2, of the prediction
    print(cv_results['test_r2'])
    print(cv_results['test_neg_root_mean_squared_error'])

    print("mean fold r2 score: %s" % np.mean(cv_results['test_r2']))
    print("std fold r2 score: %s" % np.std(cv_results['test_r2']))
    print("mean fold neg_root_mean_squared_error score: %s" % np.mean(cv_results['test_neg_root_mean_squared_error']))
    print("std fold neg_root_mean_squared_error score: %s" % np.std(cv_results['test_neg_root_mean_squared_error']))
