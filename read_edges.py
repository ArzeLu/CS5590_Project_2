#consistent 0.1 seconds runtime for facebook (88234 edges)

import networkx as nx

def graph_from_edges():    
    file = open("assets/facebook_combined.txt", "r")
    edges = file.readlines()
    g = nx.Graph()
    
    for edge in edges:
        edge = edge.split(" ")
        a = int(edge[0])
        b = int(edge[1])
        g.add_edge(a, b)
        
    return g