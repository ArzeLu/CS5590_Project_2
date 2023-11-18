import networkx as nx
import read_edges as re
import time

def generate(file_name):
    g = re.graph_from_edges(file_name)  # get the graph

    cc = []  # closeness centrality of all vertices
    times = []  # runtime of each vertex

    sample_size = 5

    for i in range(sample_size):
        print(f"Processing vertex {i}")
        total_length = 0
        start_time = time.time()

        for j in range(g.number_of_nodes()):
            if i == j:
                continue
            total_length += (len(nx.dijkstra_path(g, i, j)) - 1)

        cc.append((i, 1 / (total_length / (g.number_of_nodes() - 1))))
        times.append(time.time() - start_time)

    print("----Closeness Centrality Serial----")
    print("Runtime of each vertex:")

    for t in times:
        print(t)

    print("\nAverage time:", (sum(times) / sample_size))

    return cc
