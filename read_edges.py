#consistent 0.1 seconds runtime for facebook (88234 edges)

import networkx as nx

def graph_from_edges(file_name):   
    file = open(f"assets/{file_name}", "r")
    edges = file.readlines()
    g = nx.Graph()
    
    for edge in edges:
        edge = edge.split(" ")
        g.add_edge(int(edge[0]), int(edge[1]))

    return g