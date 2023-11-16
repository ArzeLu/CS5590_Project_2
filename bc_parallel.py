from mpi4py import MPI
import numpy as np
import read_edges as re
def load_graph_data(file_path):
    g = re.graph_from_edges(file_path) # get the graph
    print(g)
    pass
load_graph_data('sample.txt')

def dijkstra(graph, start):
    n = len(graph)
    visited = [False] * n
    distance = [float('inf')] * n
    distance[start] = 0

    for _ in range(n):
        u = min_distance(distance, visited)
        visited[u] = True

        for v in range(n):
            if not visited[v] and graph[u][v] and distance[u] + graph[u][v] < distance[v]:
                distance[v] = distance[u] + graph[u][v]
    print(distance)
    return distance

def min_distance(distance, visited):
    min_dist = float('inf')
    min_index = -1

    for i, dist in enumerate(distance):
        if not visited[i] and dist < min_dist:
            min_dist = dist
            min_index = i

    return min_index

def calculate_shortest_paths(graph):
    n = len(graph)
    all_shortest_paths = []

    for start_node in range(n):
        shortest_paths = dijkstra(graph, start_node)
        all_shortest_paths.append(shortest_paths)

    return all_shortest_paths

def calculate_closeness_centrality(graph, node, distance_matrix):
    shortest_paths = distance_matrix[node]
    closeness_centrality = 1 / np.mean(shortest_paths)
    return closeness_centrality

def calculate_betweenness_centrality(graph, node, distance_matrix):
    n = len(graph)
    betweenness_centrality = 0

    for i in range(n):
        if i != node:
            for j in range(n):
                if j != i and j != node:
                    if distance_matrix[i][j] != 0:
                        paths_ij = find_shortest_paths(graph, i, j, distance_matrix)
                        paths_i_node = find_shortest_paths(graph, i, node, distance_matrix)
                        paths_j_node = find_shortest_paths(graph, j, node, distance_matrix)

                        betweenness_centrality += (len(paths_i_node) * len(paths_j_node)) / len(paths_ij)

    return betweenness_centrality


from collections import defaultdict


def find_shortest_paths(graph, start, end, distance_matrix):
    def backtrack_paths(current, path):
        if current == start:
            paths.append(path[::-1])
            return
        for predecessor in predecessors[current]:
            backtrack_paths(predecessor, path + [predecessor])

    n = len(graph)
    visited = [False] * n
    distance = [float('inf')] * n
    distance[start] = 0
    predecessors = defaultdict(list)
    paths = []

    while True:
        u = min_distance(distance, visited)
        if u == -1 or u == end:
            break

        visited[u] = True

        for v in range(n):
            if not visited[v] and graph[u][v] and distance[u] + graph[u][v] < distance[v]:
                distance[v] = distance[u] + graph[u][v]
                predecessors[v] = [u]
            elif not visited[v] and graph[u][v] and distance[u] + graph[u][v] == distance[v]:
                predecessors[v].append(u)

    if distance[end] == float('inf'):
        # No path exists
        return []

    # Backtrack to find all shortest paths
    backtrack_paths(end, [end])

    return paths


# Example usage
if __name__ == "__main__":
    # Example graph represented as an adjacency matrix
    example_graph = [
        [0, 1, 1, 0, 0],
        [1, 0, 1, 1, 0],
        [1, 1, 0, 1, 1],
        [0, 1, 1, 0, 1],
        [0, 0, 1, 1, 0]
    ]

    start_node = 0
    end_node = 4

    # Assuming distance_matrix is already computed using Dijkstra's algorithm
    distance_matrix = calculate_shortest_paths(example_graph)

    # Find all shortest paths between start_node and end_node
    shortest_paths = find_shortest_paths(example_graph, start_node, end_node, distance_matrix)

    print(f"All Shortest Paths from {start_node} to {end_node}:")
    for path in shortest_paths:
        print(path)


def calculate_centrality(graph, n):
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()

    nodes_per_processor = n // size
    start_node = rank * nodes_per_processor
    end_node = (rank + 1) * nodes_per_processor

    # Calculate all-pairs shortest paths using Dijkstra's algorithm
    distance_matrix = calculate_shortest_paths(graph)

    # Calculate centrality for each node in the assigned range
    closeness_centrality_values = {}
    betweenness_centrality_values = {}

    for node in range(start_node, end_node):
        closeness_centrality_values[node] = calculate_closeness_centrality(graph, node, distance_matrix)
        betweenness_centrality_values[node] = calculate_betweenness_centrality(graph, node, distance_matrix)

    # Gather centrality values from all processors to the root (processor 0)
    all_closeness_centrality_values = comm.gather(closeness_centrality_values, root=0)
    all_betweenness_centrality_values = comm.gather(betweenness_centrality_values, root=0)

    # Processor 0 combines and prints the results
    if rank == 0:
        combined_closeness_centrality = {node: value for centrality_values in all_closeness_centrality_values for node, value in centrality_values.items()}
        combined_betweenness_centrality = {node: value for centrality_values in all_betweenness_centrality_values for node, value in centrality_values.items()}

        print("Closeness Centrality:")
        print(combined_closeness_centrality)

        print("Betweenness Centrality:")
        print(combined_betweenness_centrality)

'''
# Example usage
if __name__ == "__main__":
    # Specify the path to the file containing the value of n
    n_file_path = "path/to/n/file.txt"

    # Load n from the file
    with open(n_file_path, "r") as file:
        n = int(file.read().strip())

    # Specify the path to the file containing the graph data
    graph_data_file_path = "path/to/graph/data/file.txt"

    # Load your graph data from the social networks datasets
    graph_data = load_graph_data(graph_data_file_path)

    # Calculate centrality in parallel
    calculate_centrality(graph_data, n)
'''