import networkx as nx
import read_edges as re
import time

def generate(file_name):
    g = re.graph_from_edges(file_name) #get the graph
    cc = [] #closeness centrality of all vertices
    top_5 = [] #top 5 vertices of highest closeness centrality 
    n = g.number_of_nodes()
    
    for i in range(n):
        print(f"Processing vertex {i}")
        total_length = 0
        for j in range(n):
            if i == j:
                continue
            total_length += (len(nx.dijkstra_path(g, i, j, weight=1)) - 1)
        cc.append((i, total_length))
            
    return cc