import networkx as nx
from grave import Node2VecGraph
from pymatgen.analysis.graphs import StructureGraph
from pymatgen.analysis.local_env import CrystalNN


class LatticeGraphWalk:

    @staticmethod
    def walk(struct, p, q, num_walks, walk_length):
        struct_graph = StructureGraph.with_local_env_strategy(struct, CrystalNN())
        labels = {i: spec.name for i, spec in enumerate(struct.species)}
        G = nx.Graph(struct_graph.graph)
        G = nx.relabel_nodes(G, labels)

        for source, target in G.edges():
            G[source][target]['weight'] = 1

        n2v_G = Node2VecGraph(G, is_directed=False, p=p, q=q)

        n2v_G.preprocess_transition_probs()
        walks = n2v_G.simulate_walks(num_walks=num_walks, walk_length=walk_length, verbose=False)

        return walks
