
## Corpora

`all_bandgap_2020_10_09_p1_q1_walk10_len40.walks.gz` created with:
```
python bin/create_lattice_graph_corpus.py \
--data out/all_bandgap_2020_10_09.pkl.gz \
--out out/all_bandgap_2020_10_09_p1_q1_walk10_len40.walks.gz \
--p 1 --q 1 --num-walks 10 --walk-length 40 \
--processes 70 --workers 200 -z
```

`all_stable_bandgap_2020_10_09_p1_q1_walk10_len40.walks.gz` created with:
```
python bin/create_lattice_graph_corpus.py \
--data out/all_stable_bandgap_2020_10_09.pkl.gz \
--out out/all_stable_bandgap_2020_10_09_p1_q1_walk10_len40.walks.gz \
--p 1 --q 1 --num-walks 10 --walk-length 40 \
--processes 70 --workers 200 -z
```

`all_stable_bandgap_2020_10_09_p1_q1_walk2_len10.walks.gz` created with:
```
python bin/create_lattice_graph_corpus.py \
--data out/all_stable_bandgap_2020_10_09.pkl.gz \
--out out/all_stable_bandgap_2020_10_09_p1_q1_walk2_len10.walks.gz \
--p 1 --q 1 --num-walks 2 --walk-length 10 \
--processes 70 --workers 200 -z
```

`all_stable_bandgap_2020_10_09_p1_q05_walk10_len40.walks.gz` created with:
```
python bin/create_lattice_graph_corpus.py \
--data out/all_stable_bandgap_2020_10_09.pkl.gz \
--out out/all_stable_bandgap_2020_10_09_p1_q05_walk10_len40.walks.gz \
--p 1 --q 0.5 --num-walks 10 --walk-length 40 \
--processes 70 --workers 200 -z
```

`all_stable_bandgap_2020_10_09_p1_q2_walk10_len40.walks.gz` created with:
```
python bin/create_lattice_graph_corpus.py \
--data out/all_stable_bandgap_2020_10_09.pkl.gz \
--out out/all_stable_bandgap_2020_10_09_p1_q2_walk10_len40.walks.gz \
--p 1 --q 2 --num-walks 10 --walk-length 40 \
--processes 70 --workers 200 -z
```

`all_stable_bandgap_2020_10_09_p4_q1_walk10_len40.walks.gz` created with:
```
python bin/create_lattice_graph_corpus.py \
--data out/all_stable_bandgap_2020_10_09.pkl.gz \
--out out/all_stable_bandgap_2020_10_09_p4_q1_walk10_len40.walks.gz \
--p 4 --q 1 --num-walks 10 --walk-length 40 \
--processes 70 --workers 200 -z
```

## Models

`all_bandgap_2020_10_09_p1_q1_walk10_len40.dim100.glove.model`:
```
python bin/build_glove_model.py \
--corpus out/all_bandgap_2020_10_09_p1_q1_walk10_len40.walks.gz \
--out out/all_bandgap_2020_10_09_p1_q1_walk10_len40.dim100 \
--components 100 --train 50 --window 10
```
```
python bin/plot_atom_vectors_tsne.py \
--model out/all_bandgap_2020_10_09_p1_q1_walk10_len40.dim100.glove.model \
--iterations 5000 --perplexity 10 --learning-rate 10
```

`all_bandgap_2020_10_09_p1_q1_walk10_len40.dim20.glove.model`:
```
python bin/build_glove_model.py \
--corpus out/all_bandgap_2020_10_09_p1_q1_walk10_len40.walks.gz \
--out out/all_bandgap_2020_10_09_p1_q1_walk10_len40.dim20 \
--components 20 --train 50 --window 10
```
```
python bin/plot_atom_vectors_tsne.py \
--model out/all_bandgap_2020_10_09_p1_q1_walk10_len40.dim20.glove.model \
--iterations 5000 --perplexity 10 --learning-rate 10
```


`all_stable_bandgap_2020_10_09_p1_q1_walk2_len10.dim100.glove.model`:
```
python bin/build_glove_model.py \
--corpus out/all_stable_bandgap_2020_10_09_p1_q1_walk2_len10.walks.gz \
--out out/all_stable_bandgap_2020_10_09_p1_q1_walk2_len10.dim100 \
--components 100 --train 50 --window 10
```
```
python bin/plot_atom_vectors_tsne.py \
--model out/all_stable_bandgap_2020_10_09_p1_q1_walk2_len10.dim100.glove.model \
--iterations 5000 --perplexity 10 --learning-rate 10
```

`all_stable_bandgap_2020_10_09_p1_q1_walk2_len10.dim20.glove.model`:
```
python bin/build_glove_model.py \
--corpus out/all_stable_bandgap_2020_10_09_p1_q1_walk2_len10.walks.gz \
--out out/all_stable_bandgap_2020_10_09_p1_q1_walk2_len10.dim20 \
--components 20 --train 50 --window 10
```
```
python bin/plot_atom_vectors_tsne.py \
--model out/all_stable_bandgap_2020_10_09_p1_q1_walk2_len10.dim20.glove.model \
--iterations 5000 --perplexity 10 --learning-rate 10
```


`all_stable_bandgap_2020_10_09_p1_q05_walk10_len40.dim100.glove.model`:
```
python bin/build_glove_model.py \
--corpus out/all_stable_bandgap_2020_10_09_p1_q05_walk10_len40.walks.gz \
--out out/all_stable_bandgap_2020_10_09_p1_q05_walk10_len40.dim100 \
--components 100 --train 50 --window 10
```
```
python bin/plot_atom_vectors_tsne.py \
--model out/all_stable_bandgap_2020_10_09_p1_q05_walk10_len40.dim100.glove.model \
--iterations 5000 --perplexity 10 --learning-rate 10
```

`all_stable_bandgap_2020_10_09_p1_q05_walk10_len40.dim20.glove.model`:
```
python bin/build_glove_model.py \
--corpus out/all_stable_bandgap_2020_10_09_p1_q05_walk10_len40.walks.gz \
--out out/all_stable_bandgap_2020_10_09_p1_q05_walk10_len40.dim20 \
--components 20 --train 50 --window 10
```
```
python bin/plot_atom_vectors_tsne.py \
--model out/all_stable_bandgap_2020_10_09_p1_q05_walk10_len40.dim20.glove.model \
--iterations 5000 --perplexity 10 --learning-rate 10
```


`all_stable_bandgap_2020_10_09_p4_q1_walk10_len40.dim100.glove.model`:
```
python bin/build_glove_model.py \
--corpus out/all_stable_bandgap_2020_10_09_p4_q1_walk10_len40.walks.gz \
--out out/all_stable_bandgap_2020_10_09_p4_q1_walk10_len40.dim100 \
--components 100 --train 50 --window 10
```
```
python bin/plot_atom_vectors_tsne.py \
--model out/all_stable_bandgap_2020_10_09_p4_q1_walk10_len40.dim100.glove.model \
--iterations 5000 --perplexity 10 --learning-rate 10
```

`all_stable_bandgap_2020_10_09_p4_q1_walk10_len40.dim20.glove.model`:
```
python bin/build_glove_model.py \
--corpus out/all_stable_bandgap_2020_10_09_p4_q1_walk10_len40.walks.gz \
--out out/all_stable_bandgap_2020_10_09_p4_q1_walk10_len40.dim20 \
--components 20 --train 50 --window 10
```
```
python bin/plot_atom_vectors_tsne.py \
--model out/all_stable_bandgap_2020_10_09_p4_q1_walk10_len40.dim20.glove.model \
--iterations 5000 --perplexity 10 --learning-rate 10
```

`all_stable_bandgap_2020_10_09_p4_q1_walk10_len40.dim20_c1.glove.model`:
```
python bin/build_glove_model.py \
--corpus out/all_stable_bandgap_2020_10_09_p4_q1_walk10_len40.walks.gz \
--out out/all_stable_bandgap_2020_10_09_p4_q1_walk10_len40.dim20_c1 \
--components 20 --train 50 --window 1
```
```
python bin/plot_atom_vectors_tsne.py \
--model out/all_stable_bandgap_2020_10_09_p4_q1_walk10_len40.dim20_c1.glove.model \
--iterations 5000 --perplexity 10 --learning-rate 10
```


`all_stable_bandgap_2020_10_09_p1_q2_walk10_len40.dim100.glove.model`:
```
python bin/build_glove_model.py \
--corpus out/all_stable_bandgap_2020_10_09_p1_q2_walk10_len40.walks.gz \
--out out/all_stable_bandgap_2020_10_09_p1_q2_walk10_len40.dim100 \
--components 100 --train 50 --window 10
```
```
python bin/plot_atom_vectors_tsne.py \
--model out/all_stable_bandgap_2020_10_09_p1_q2_walk10_len40.dim100.glove.model \
--iterations 5000 --perplexity 10 --learning-rate 10
```

`all_stable_bandgap_2020_10_09_p1_q2_walk10_len40.dim20.glove.model`:
```
python bin/build_glove_model.py \
--corpus out/all_stable_bandgap_2020_10_09_p1_q2_walk10_len40.walks.gz \
--out out/all_stable_bandgap_2020_10_09_p1_q2_walk10_len40.dim20 \
--components 20 --train 50 --window 10
```
```
python bin/plot_atom_vectors_tsne.py \
--model out/all_stable_bandgap_2020_10_09_p1_q2_walk10_len40.dim20.glove.model \
--iterations 5000 --perplexity 10 --learning-rate 10
```


`all_stable_bandgap_2020_10_09_p1_q1_walk10_len40.dim100.glove.model`:
```
python bin/build_glove_model.py \
--corpus out/all_stable_bandgap_2020_10_09_p1_q1_walk10_len40.walks.gz \
--out out/all_stable_bandgap_2020_10_09_p1_q1_walk10_len40.dim100 \
--components 100 --train 50 --window 10
```
```
python bin/plot_atom_vectors_tsne.py \
--model out/all_stable_bandgap_2020_10_09_p1_q1_walk10_len40.dim100.glove.model \
--iterations 5000 --perplexity 10 --learning-rate 10
```

`all_stable_bandgap_2020_10_09_p1_q1_walk10_len40.dim20.glove.model`:
```
python bin/build_glove_model.py \
--corpus out/all_stable_bandgap_2020_10_09_p1_q1_walk10_len40.walks.gz \
--out out/all_stable_bandgap_2020_10_09_p1_q1_walk10_len40.dim20 \
--components 20 --train 50 --window 10
```
```
python bin/plot_atom_vectors_tsne.py \
--model out/all_stable_bandgap_2020_10_09_p1_q1_walk10_len40.dim20.glove.model \
--iterations 5000 --perplexity 10 --learning-rate 10
```

`all_stable_bandgap_2020_10_09_p1_q1_walk10_len40.dim20_c1.glove.model`:
```
python bin/build_glove_model.py \
--corpus out/all_stable_bandgap_2020_10_09_p1_q1_walk10_len40.walks.gz \
--out out/all_stable_bandgap_2020_10_09_p1_q1_walk10_len40.dim20_c1 \
--components 20 --train 50 --window 1
```
```
python bin/plot_atom_vectors_tsne.py \
--model out/all_stable_bandgap_2020_10_09_p1_q1_walk10_len40.dim20_c1.glove.model \
--iterations 5000 --perplexity 10 --learning-rate 10
```

`all_stable_bandgap_2020_10_09_p1_q1_walk10_len40.dim20_c2.glove.model`:
```
python bin/build_glove_model.py \
--corpus out/all_stable_bandgap_2020_10_09_p1_q1_walk10_len40.walks.gz \
--out out/all_stable_bandgap_2020_10_09_p1_q1_walk10_len40.dim20_c2 \
--components 20 --train 50 --window 2
```
```
python bin/plot_atom_vectors_tsne.py \
--model out/all_stable_bandgap_2020_10_09_p1_q1_walk10_len40.dim20_c2.glove.model \
--iterations 5000 --perplexity 10 --learning-rate 10
```

`all_stable_bandgap_2020_10_09_p1_q1_walk10_len40.dim20_c20.glove.model`:
```
python bin/build_glove_model.py \
--corpus out/all_stable_bandgap_2020_10_09_p1_q1_walk10_len40.walks.gz \
--out out/all_stable_bandgap_2020_10_09_p1_q1_walk10_len40.dim20_c20 \
--components 20 --train 50 --window 20
```
```
python bin/plot_atom_vectors_tsne.py \
--model out/all_stable_bandgap_2020_10_09_p1_q1_walk10_len40.dim20_c20.glove.model \
--iterations 5000 --perplexity 10 --learning-rate 10
```

### Co-occurrence counts

`all_stable_bandgap_2020_10_09.counts.pkl`:
```
python bin/count_cooccurrences.py --data out/all_stable_bandgap_2020_10_09.pkl.gz --out out/all_stable_bandgap_2020_10_09.counts.pkl --workers 4
```

`all_bandgap_2020_10_09.counts.pkl`:
```
python bin/count_cooccurrences.py --data out/all_bandgap_2020_10_09.pkl.gz --out out/all_bandgap_2020_10_09.counts.pkl --workers 4
```

## Dataset

Downloaded from https://qmml.org/datasets.html

ABC2D6-16

Felix A. Faber, Alexander Lindmaa, O. Anatole von Lilienfeld, Rickard Armiento: Machine Learning Energies of 2 Million 
Elpasolite (ABC2D6) Crystals, Physical Review Letters 117(13): 135502, 2016.

11k and 12k elpasolite crystals with DFT/PBE formation energies. 11,358 (12 elements, III-VI) and 10,590 (39 elements, 
I-VIII) elpasolite crystals with relaxed geometries and formation energies computed at the DFT/PBE level of theory.