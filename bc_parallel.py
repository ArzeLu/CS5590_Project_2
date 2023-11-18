from mpi4py import MPI
import networkx as nx
import numpy as np

def read_graph(filename):
    G = nx.Graph()
    with open(filename, 'r') as file:
        for line in file:
            node_a, node_b = map(int, line.split())
            G.add_edge(node_a, node_b)
    return G

def bfs(graph, source):
    visited = set()
    levels = {source: 0}
    queue = [source]

    while queue:
        current_node = queue.pop(0)
        for neighbor in graph.neighbors(current_node):
            if neighbor not in visited:
                visited.add(neighbor)
                queue.append(neighbor)
                levels[neighbor] = levels[current_node] + 1

    return levels

def calculate_partial_betweenness_centrality(graph, nodes):
    partial_betweenness = np.zeros(graph.number_of_nodes())

    for source in nodes:
        levels = bfs(graph, source)

        shortest_paths = {node: 0 for node in graph.nodes()}
        num_paths = {node: 0 for node in graph.nodes()}
        stack = []

        for node in sorted(graph.nodes(), key=lambda x: levels[x], reverse=True):
            if node == source:
                num_paths[node] = 1
                continue

            for neighbor in graph.neighbors(node):
                if levels[neighbor] == levels[node] + 1:
                    num_paths[node] += num_paths[neighbor]
                    shortest_paths[node] += num_paths[neighbor] + shortest_paths[neighbor]

            if node != source:
                partial_betweenness[node] += shortest_paths[node] / 2.0

    return partial_betweenness

def main():
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()

    if rank == 0:
        # Read the graph from the input file
        graph_file = "assets/facebook_combined.txt"
        graph = read_graph(graph_file)
    else:
        graph = None

    # Broadcast the graph to all processes
    graph = comm.bcast(graph, root=0)

    # Split nodes among processes
    nodes_per_process = graph.number_of_nodes() // size
    start_node = rank * nodes_per_process
    end_node = (rank + 1) * nodes_per_process if rank != size - 1 else graph.number_of_nodes()

    # Calculate partial betweenness centrality
    partial_betweenness = calculate_partial_betweenness_centrality(graph, range(start_node, end_node))

    # Gather partial results to the root process
    all_partial_betweenness = comm.gather(partial_betweenness, root=0)

    if rank == 0:
        # Aggregate partial results to get the final betweenness centrality
        betweenness_centrality = np.sum(all_partial_betweenness, axis=0)
        normalization_factor = 1.0 / ((graph.number_of_nodes() - 1) * (graph.number_of_nodes() - 2))
        betweenness_centrality *= normalization_factor

        print("Betweenness Centrality:", betweenness_centrality)

if __name__ == "__main__":
    main()
