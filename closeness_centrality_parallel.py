import math
import time
import networkx as nx
import algos
import helpers
from mpi4py import MPI

# set up MPI
comm = MPI.COMM_WORLD
p = comm.Get_size()
r = comm.Get_rank()
    
# Closeness centrality with floyd warshall
def cc_fw(g, sample_size):
    # measures total runtime
    if r == 0:
        timer_1 = time.time() 
        print("\n==================Closeness Centrality with Floyd-Warshall===================")
        print("===============================================================================")
    
    # Set up node data
    n, nl, nl_inv = helpers.get_node_data(g)
    
    # Divide the workload for each processor
    start = int((r * sample_size) / p)
    end = int((((r + 1) * sample_size) / p))
    
    assignment = helpers.assign_tasks(r, p, start, end)
    
    if r == 0:
        print(f"\nUsed {p} processor(s). Processed {sample_size} nodes; {int(sample_size / n * 100)}% of total {n} nodes")
        timer_2 = time.time()
    
    # floyd-warshall matrix
    # filling weights (1) for each edge, and fill zeros for diagonal
    # DONT PARALLELIZE
    dist = [[math.inf] * n for _ in range(n)] #initialize the 2D array with infinity. By qtniko
    for ui in range(n):
        u = nl.get(ui)
        dist[ui][ui] = 0
        for v in g.neighbors(u):
            vi = nl_inv.get(v)
            dist[ui][vi] = 1
            
    dist = algos.fw(g, assignment, dist, start, end, sample_size)

    if r == 0:                    
        algo_runtime = time.time() - timer_2
        
    # Path lengths logic in README
    closeness_centrality = {x: 0 for x in range(start, end)}
    for ui in range(start, end):
        u = nl.get(ui)
        for vi in range(n):
            v = nl.get(vi)
            if dist[ui][vi] != math.inf:
                closeness_centrality[ui] += dist[ui][vi]
        closeness_centrality[ui] = int(((n - 1) / closeness_centrality[ui]) * 1000) / 1000
    
    helpers.top5_cc(closeness_centrality, nl)
    
    if r == 0:
        total_time = time.time() - timer_1    
        print(f"\nTotal runtime: {total_time:.4f} seconds")
        print(f"CC Floyd-Warshall Algo Runtime: {algo_runtime:.4f} seconds\n")
        print("===============================================================================")
        print("===============================================================================\n")   
    
# closeness centrality with bfs
def cc_bfs(g, sample_size):
    if r == 0:
        timer_1 = time.time()
        print("\n==================Closeness Centrality with BFS===================")
        print("===============================================================================")
        
    # Set up node data
    n, nl, nl_inv = helpers.get_node_data(g)
    
    # Divide up the workload for each processor
    start = int((r * sample_size) / p)
    end = int((((r + 1) * sample_size) / p))
    closeness_centrality = {x: 0 for x in range(start, end)}
    
    if r == 0:
        print(f"\nUsed {p} processor(s). Processed {sample_size} nodes; {int(sample_size / n * 100)}% of total {n} nodes")
        timer_2 = time.time()
    for si in range(start, end):
        s = nl.get(si)
        sp = algos.bfs_basic(g, nl, nl_inv, s)
        for a, b in sp.items():
            closeness_centrality[si] += b
        closeness_centrality[si] = int(((n - 1) / closeness_centrality[si]) * 1000) / 1000
    if r == 0:
        algo_runtime = time.time() - timer_2
        
    helpers.top5_cc(closeness_centrality, nl)
    
    if r == 0:
        total_time = time.time() - timer_1    
        print()
        print(f"Total runtime: {total_time:.4f} seconds")
        print(f"CC BFS Algo Runtime: {algo_runtime:.4f} seconds\n")
        print("===============================================================================")
        print("===============================================================================\n")