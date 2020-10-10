# AtomWalk

- atoms with similar chemistry appear in similar chemo-structural environments

Notes:
- Atom2Vec uses 60,605 inorganic compounds from the Materials Project database, but they only use the compounds which
are at most quaternary, as compounds with more atom types depend more heavily on structure


`all_bandgap_2020_10_09_p1_q1_walk10_len40.walks.gz` created with:
```
python bin/create_lattice_graph_corpus.py \
--data out/all_bandgap_2020_10_09.pkl \
--out out/all_bandgap_2020_10_09_p1_q1_walk10_len40.walks.gz \
--p 1 --q 1 --num-walks 10 --walk-length 40 \
--workers 4 -z
```
