from mpi4py import MPI
import networkx as nx

def get_graph(file_name):   
    f = open(f"assets/{file_name}", "r")
    edges = f.readlines()
    g = nx.Graph()
    
    for edge in edges:
        edge = edge.split(" ")
        g.add_edge(int(edge[0]), int(edge[1]))
    
    f.close()
    return g
    
# Runs BFS
# Returns only shortest paths from each node to s
def bfs_basic(g, s):
    sp = {s: 0} # shortest paths (node: path length to s)
    visited = set()
    queue = [s]
    
    while queue:
        current_node = queue.pop(0)
        for neighbor in g.neighbors(current_node):
            if neighbor not in visited:
                visited.add(neighbor)
                queue.append(neighbor)
                sp[neighbor] = sp[current_node] + 1

    sp[s] = 0

    return sp

# BFS
# Returns: 1. stack of visited nodes in order of non-increasing distance from s
#          2. immediate parent(s) of each node of each shortest path
#          3. shortest paths from each node to s
def bfs_detailed(g, nl, nl_inv, s):
    n = g.number_of_nodes()
    si = nl_inv.get(s)
    stack = []
    queue = [s]
    prev = [[] for _ in range(n)] # previous node of each shortest path
    
    sp = [0] * n # shortest path (sigma)
    sp[si] = 1
    
    d = [-1] * n # distance
    d[si] = 0
    
    while queue:
        v = queue.pop(0)
        stack.append(v)
        vi = nl_inv.get(v)
        for w in g.neighbors(v):
            wi = nl_inv.get(w)
            if d[wi] < 0:
                queue.append(w)
                d[wi] = d[vi] + 1
            if d[wi] == d[vi] + 1:
                sp[wi] += sp[vi]
                prev[wi].append(v)

    return (stack, prev, sp)
                    
# Send data data to p_0
# Then returns the top 5
def top5_cc(data):
    comm = MPI.COMM_WORLD
    p = comm.Get_size()
    r = comm.Get_rank()
    
    if r != 0:
        comm.send(obj = data, dest = 0, tag = 3)
    else:
        for pi in range(1, p):
            data.update(comm.recv(source = pi, tag = 3))
      
        data = dict(sorted(data.items(), key = lambda x: x[1], reverse = True))
   
        # Get top 5
        top5 = [[] for _ in range(5)]
        cc_set = set()
        for x, y in data.items():
            cc_set.add(y)
            if len(cc_set) == 6:
                break
            top5[len(cc_set) - 1].append((x, y))
        
        print("\n>>>>> Top 5: <<<<<")
        for i in range(len(top5)):
            print(f"#{i}:")
            for y in top5[i]:
                print(f"    {y[0]}, {y[1]}")
                
def top5_bc(data, nl):
    comm = MPI.COMM_WORLD
    p = comm.Get_size()
    r = comm.Get_rank()
    n = len(nl)
    normalizer = 1 / ((n - 1) * (n - 2))
    
    if r != 0:
        comm.send(obj = data, dest = 0, tag = 3)
    else:
        for pi in range(1, p):
            other_data = comm.recv(source = pi, tag = 3)
            for i in range(len(data)):
                data[i] += other_data[i]
        
        data = dict(sorted(data.items(), key = lambda x: x[1], reverse = True))
        
        for i in range(len(data)):
            data[i] *= normalizer
        
        # Get top 5
        top5 = [[] for _ in range(5)]
        cc_set = set()
        for x, y in data.items():
            cc_set.add(y)
            if len(cc_set) == 6:
                break
            top5[len(cc_set) - 1].append((x, y))
        
        print("\n>>>>> Top 5: <<<<<")
        for i in range(len(top5)):
            print(f"#{i}:")
            for y in top5[i]:
                print(f"    {nl.get(y[0])}, {y[1]}")