from mpi4py import MPI
import networkx as nx
import math

# set up MPI
comm = MPI.COMM_WORLD
p = comm.Get_size()
r = comm.Get_rank()

# floyd warshall
def fw(g, assignment, dist, start, end, sample_size):
    n = g.number_of_nodes()
    
    # floyd-warshall algo
    # row based parallelization
    for k in range(sample_size):        
        impact_row = []
        
        # pre-calculate the impacting row,
        # then send the row data to all processors
        if r == assignment[k]: 
            impact_row = dist[k].copy()
            updated_impact_row = [min(dist[k][j], (dist[k][k] + dist[k][j])) for j in range(n)] #calculate updated_impact_row
            
            # Send impact row to other processors
            for pi in range(p):
                if pi < r:
                    comm.send(obj = impact_row, dest = pi, tag = 2)
                elif pi > r:
                    comm.send(obj = updated_impact_row, dest = pi, tag = 2)
        else:
            impact_row = comm.recv(source = assignment[k], tag = 2)
            
        for i in range(start, end):
            for j in range(n):
                dist[i][j] = min(dist[i][j], (dist[i][k] + impact_row[j]))
    return dist
    
# Runs BFS
# Returns only shortest paths from each node to s
def bfs_basic(g, nl, nl_inv, s):
    si = nl_inv.get(s)
    sp = {si: 0} # shortest paths (node: path length to s)
    visited = set()
    queue = [s]
    
    while queue:
        v = queue.pop(0)
        vi =  nl_inv.get(v)
        for w in g.neighbors(v):
            wi =  nl_inv.get(w)
            if w not in visited:
                visited.add(w)
                queue.append(w)
                sp[wi] = sp[vi] + 1

    sp[si] = 0
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