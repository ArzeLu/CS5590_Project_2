from mpi4py import MPI

import networkx as nx
import read_edges as re

def closeness_centrality():
    g = re.graph_from_edges()
    print(g.number_of_edges())
    
if __name__ == "__main__":
    closeness_centrality()