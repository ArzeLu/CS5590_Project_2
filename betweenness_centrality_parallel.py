from mpi4py import MPI
import read_edges as re
import networkx as nx
import time
import helpers

#set up MPI
comm = MPI.COMM_WORLD
p = comm.Get_size()
r = comm.Get_rank()

def bc_bfs(g, sample_size):
    # measures total runtime
    if r == 0:
        timer_1 = time.time() 
        print("\n===================== Betweenness Centrality with BFS ========================")
        print("===============================================================================")
        
    n = g.number_of_nodes()
    
    # Divide the workload for each processor
    start = int((r * sample_size) / p)
    end = int((((r + 1) * sample_size) / p))
    
    betweenness_centrality = {x: 0 for x in range(n)}
    normalizer = 1 / ((n - 1) * (n - 2))
    
    # Start runtime measurement. Only measures the main floyd-warshall part
    if r == 0:
        print(f"\nProcessed {sample_size} nodes; {int(sample_size / n * 100)}% of total {n} nodes")
        timer_2 = time.time()
    
    # Ulrik Brandes Betweenness Centrality ALgorithm
    for s in range(start, end):
        stack, prev, sp = helpers.bfs_detailed(g, s)
                
        depd = [0] * n
        while stack:
            w = stack.pop()
            for v in prev[w]:
                depd[v] += (sp[v] / sp[w]) * (1 + depd[w])
            if w != s:
                betweenness_centrality[w] += depd[w]          
        
    # End algo runtime measurement
    if r == 0:                    
        algo_runtime = time.time() - timer_2
    
    # Top 5
    helpers.top5_bc(betweenness_centrality, normalizer)

    if r == 0:
        total_time = time.time() - timer_1    
        print()
        print(f"Total runtime: {total_time:.4f} seconds")
        print(f"BC BFS Algo Runtime: {algo_runtime:.4f} seconds\n")
        print("===============================================================================")
        print("===============================================================================\n")    