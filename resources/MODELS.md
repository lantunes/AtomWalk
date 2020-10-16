Models
======

A model provides a vector representation for an atom. All models (with the exception of the Random vector and One-hot 
models) are based on the Crystal Lattice Graph associated with each structure.


## GloVe Models
- operates on either crystal lattice graph random walks or crystal lattice graph neighbors
- allows for self-interactions (if co-occurrence count matrix contains counts for self-co-occurrences and implementation supports it)

### GloVe on Crystal Lattice Graph Random Walks (AtomWalk)
- `all_bandgap_2020_10_09_p1_q1_walk10_len40.dim20.glove.model`
- `all_bandgap_2020_10_09_p1_q1_walk10_len40.dim100.glove.model`
- `all_stable_bandgap_2020_10_09_p1_q05_walk10_len40.dim100.glove.model`
- `all_stable_bandgap_2020_10_09_p1_q1_walk10_len40.dim20_c1.glove.model`
- `all_stable_bandgap_2020_10_09_p1_q1_walk10_len40.dim20_c20.glove.model`
- `all_stable_bandgap_2020_10_09_p1_q2_walk10_len40.dim100.glove.model`
- `all_stable_bandgap_2020_10_09_p1_q1_walk10_len40.dim100.glove.model`
- `all_stable_bandgap_2020_10_09_p1_q1_walk10_len40.dim20.glove.model`
- `all_stable_bandgap_2020_10_09_p1_q1_walk10_len40.dim20_c2.glove.model`
- `all_stable_bandgap_2020_10_09_p1_q05_walk10_len40.dim20.glove.model`
- `all_stable_bandgap_2020_10_09_p1_q1_walk2_len10.dim20.glove.model`
- `all_stable_bandgap_2020_10_09_p1_q2_walk10_len40.dim20.glove.model`
- `all_stable_bandgap_2020_10_09_p1_q1_walk2_len10.dim100.glove.model`
- `all_stable_bandgap_2020_10_09_p4_q1_walk10_len40.dim20.glove.model`
- `all_stable_bandgap_2020_10_09_p4_q1_walk10_len40.dim20_c1.glove.model`
- `all_stable_bandgap_2020_10_09_p4_q1_walk10_len40.dim20_c1_mc10.glove.model`
- `all_stable_bandgap_2020_10_09_p4_q1_walk10_len40.dim20_c1_mc1000.glove.model`
- `all_stable_bandgap_2020_10_09_p4_q1_walk10_len40.dim100.glove.model`

### GloVe on Crystal Lattice Graph Neighbors
- `all_stable_bandgap_2020_10_09.counts.dim20.glove.model`
- `all_stable_bandgap_2020_10_09.counts.dim30.glove.model`


## GraVe Models
- similar to GloVe models, but also jointly learn atom attributes (e.g. electronegativity)
- operates on either crystal lattice graph random walks or crystal lattice graph neighbors
- does not allow for self-interactions (since based on Factorization Machines)


## Skip-gram Models
- operates on either crystal lattice graph random walks or crystal lattice graph neighbors
- allows for self-interactions

### Skip-gram on Crystal Lattice Graph Random Walks
// all_stable_bandgap_2020_10_09_p1_q1_walk10_len40.walks.gz
dim=20, window=2, workers=1, epochs=10  # 0.20547172 +/- 0.0067635
dim=20, window=10, workers=1, epochs=10  # 0.228587 +/- 0.0135177
dim=30, window=2, workers=1, epochs=10  # 0.167361 +/- 0.00510546
dim=30, window=10, workers=1, epochs=10  # 0.177262 +/- 0.012235
// all_stable_bandgap_2020_10_09_p1_q2_walk10_len40.walks.gz
dim=30, window=2, workers=1, epochs=10  # 0.16367480 +/- 0.0100814
// all_stable_bandgap_2020_10_09_p1_q05_walk10_len40.walks.gz
dim=30, window=2, workers=1, epochs=10  # 0.181072 +/- 0.01172305
// all_stable_bandgap_2020_10_09_p4_q1_walk10_len40.walks.gz
dim=30, window=2, workers=1, epochs=10  # 0.1573134 +/- 0.009762
dim=30, window=10, workers=1, epochs=10  # 0.181542 +/- 0.014491
// all_bandgap_2020_10_09_p1_q1_walk10_len40.walks.gz
dim=30, window=2, workers=1, epochs=10  # 0.178125 +/- 0.0194813

### Skip-gram on Crystal Lattice Graph Neighbors (SkipAtom)
- `all_stable_bandgap_2020_10_09.pairs.dim20.model`+`all_stable_bandgap_2020_10_09.pairs.training.data`
- `all_stable_bandgap_2020_10_09.pairs.dim30.model`+`all_stable_bandgap_2020_10_09.pairs.training.data`
- `all_bandgap_2020_10_09.pairs.dim20.model`+`all_bandgap_2020_10_09.pairs.training.data`
- `all_bandgap_2020_10_09.pairs.dim30.model`+`all_bandgap_2020_10_09.pairs.training.data`


## SVD Models
- operates on co-occurrence count matrices
- operates on either crystal lattice graph random walks or crystal lattice graph neighbors
- allows for self-interactions (if co-occurrence count matrix contains counts for self-co-occurrences)

### SVD on Crystal Lattice Graph Random Walks
- `SVDModelFromGloVeCorpus: all_stable_bandgap_2020_10_09_p1_q1_walk10_len40.dim20.corpus.model, d=20`
- `SVDModelFromGloVeCorpus: all_stable_bandgap_2020_10_09_p1_q1_walk10_len40.dim20.corpus.model, d=20, p=2`
- `SVDModelFromGloVeCorpus: all_stable_bandgap_2020_10_09_p1_q1_walk10_len40.dim20.corpus.model, d=30`
- `SVDModelFromGloVeCorpus: all_stable_bandgap_2020_10_09_p1_q1_walk10_len40.dim20.corpus.model, d=30, p=2`
- `SVDModelFromGloVeCorpus: all_bandgap_2020_10_09_p1_q1_walk10_len40.dim20.corpus.model, d=20`
- `SVDModelFromGloVeCorpus: all_bandgap_2020_10_09_p1_q1_walk10_len40.dim20.corpus.model, d=20, p=2`
- `SVDModelFromGloVeCorpus: all_bandgap_2020_10_09_p1_q1_walk10_len40.dim20.corpus.model, d=30`
- `SVDModelFromGloVeCorpus: all_bandgap_2020_10_09_p1_q1_walk10_len40.dim20.corpus.model, d=30, p=2`

### SVD on Crystal Lattice Graph Neighbors
- `SVDModel: all_stable_bandgap_2020_10_09.counts.pkl, dim=20`
- `SVDModel: all_stable_bandgap_2020_10_09.counts.pkl, dim=20, p=2`
- `SVDModel: all_stable_bandgap_2020_10_09.counts.pkl, dim=30`
- `SVDModel: all_stable_bandgap_2020_10_09.counts.pkl, dim=30, p=2`
- `SVDModel: all_bandgap_2020_10_09.counts.pkl, dim=30`
- `SVDModel: all_bandgap_2020_10_09.counts.pkl, dim=30, p=1`
- `SVDModel: all_bandgap_2020_10_09.counts.pkl, dim=30, p=2`
- `SVDModel: all_bandgap_2020_10_09.counts.pkl, dim=30, p=3`


## Other Models

### Random vectors
- `random_atom2vec_dim20.model`
- `random_atom2vec_dim30.model`

### One-hot vectors
- when the one-hot vectors for a compound are pooled with mean pooling, they form the input representation for ElemNet
- `one_hot_atom2vec.model`

