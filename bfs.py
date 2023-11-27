import networkx as nx

# Runs BFS
# Returns shortest paths from each node to the source,
# in the form of two dictionaries in a tuple, node: length and length: nodes
def get_bfs(graph, source):
    sp = {source: 0} # shortest paths (node: path length to source)
    visited = set()
    queue = [source]
    longest = 0
    
    while queue:
        current_node = queue.pop(0)
        for neighbor in graph.neighbors(current_node):
            if neighbor not in visited:
                visited.add(neighbor)
                queue.append(neighbor)
                sp[neighbor] = sp[current_node] + 1
                
                if sp[neighbor] > longest:
                    longest = sp[neighbor]
    
    sp[source] = 0
    sp_inverse = {x: [] for x in range(longest + 1)}
    
    for a, b in sp.items():
        sp_inverse[b].append(a)
        
    return (sp, sp_inverse)