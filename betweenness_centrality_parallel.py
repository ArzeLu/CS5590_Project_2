from mpi4py import MPI
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
    
    # Set up node data
    # Keeping two dictionaries just in case graph nodes aren't purely sequential
    n = g.number_of_nodes()
    nl = list(g.nodes()) # node list
    nl = {x: nl[x] for x in range(n)} # node list as dictionary (index: node)
    nl_inv = {y: x for x, y in nl.items()} # inverse node dictionary (node: index)
    
    # Divide the workload for each processor
    start = int((r * sample_size) / p)
    end = int((((r + 1) * sample_size) / p))
    
    betweenness_centrality = {x: 0 for x in range(n)}    
    
    # Start runtime measurement. Only measures the main floyd-warshall part
    if r == 0:
        print(f"\nProcessed {sample_size} nodes; {int(sample_size / n * 100)}% of total {n} nodes")
        timer_2 = time.time()
    
    # Ulrik Brandes Betweenness Centrality ALgorithm
    for si in range(start, end):
        s = nl.get(si)
        stack, prev, sp = helpers.bfs_detailed(g, nl, nl_inv, s)              
        depd = [0] * n
        while stack:
            w = stack.pop()
            wi = nl_inv.get(w)
            for v in prev[wi]:
                vi = nl_inv.get(v)
                depd[vi] += (sp[vi] / sp[wi]) * (1 + depd[wi])
            if w != s:
                betweenness_centrality[wi] += depd[wi]          
        
    # End algo runtime measurement
    if r == 0:                    
        algo_runtime = time.time() - timer_2
    
    # Top 5
    helpers.top5_bc(betweenness_centrality, nl)

    if r == 0:
        total_time = time.time() - timer_1    
        print()
        print(f"Total runtime: {total_time:.4f} seconds")
        print(f"BC BFS Algo Runtime: {algo_runtime:.4f} seconds\n")
        print("===============================================================================")
        print("===============================================================================\n")    