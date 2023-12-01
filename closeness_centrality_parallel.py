import math
import time
import networkx as nx
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
    
    n = g.number_of_nodes()

    # Divide the workload for each processor
    start = int((r * sample_size) / p)
    end = int((((r + 1) * sample_size) / p))
    
    #================== Calculate for Task Assignments ================"
    # maps each node to the processor rank that is responsible for it,
    # the node is the index,
    # then merge lists in the order of the ranks so the index matches with node,
    # then broadcast to all processors.
    # Saves calculation on determining who is responsible for an impact row later
    assignment = [r] * (end - start)
            
    if r != 0:
        comm.send(obj = assignment, dest = 0, tag = 1)
    else:
        for pi in range(1, p):
            assignment += comm.recv(source = pi, tag = 1)
            
    assignment = comm.bcast(obj = assignment, root = 0)
    #=================================================================="
    
    # Start runtime measurement. Only measures the main floyd-warshall part
    if r == 0:
        print(f"\nProcessed {sample_size} nodes; {int(sample_size / n * 100)}% of total {n} nodes")
        timer_2 = time.time()
        
    # floyd-warshall matrix
    # filling weights (1) for each edge, and fill zeros for diagonal
    # DONT PARALLELIZE
    dist = [[math.inf] * n for _ in range(n)] #initialize the 2D array with infinity. By qtniko
    for u in range(n):
        dist[u][u] = 0
        for v in g.neighbors(u):
            dist[u][v] = 1
    
    # floyd-warshall algo
    # parallelize logic in README
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
    
    # End algo runtime measurement
    if r == 0:                    
        algo_runtime = time.time() - timer_2
        
    # Path lengths logic in README
    closeness_centrality = {x: 0 for x in range(start, end)}
    
    for i in range(start, end):
        for j in range(n):
            if dist[i][j] != math.inf:
                closeness_centrality[i] += dist[i][j]
        closeness_centrality[i] = int(((n - 1) / closeness_centrality[i]) * 1000) / 1000
    
    helpers.top5_cc(closeness_centrality)
    
    if r == 0:
        total_time = time.time() - timer_1    
        print()
        print(f"Total runtime: {total_time:.4f} seconds")
        print(f"CC Floyd-Warshall Algo Runtime: {algo_runtime:.4f} seconds\n")
        print("===============================================================================")
        print("===============================================================================\n")   
    
# closeness centrality with bfs
def cc_bfs(g, sample_size):
    n = g.number_of_nodes()
    
    if r == 0:
        timer_1 = time.time()
        print("\n==================Closeness Centrality with BFS===================")
        print("===============================================================================")
    
    # Divide up the workload for each processor
    start = int((r * sample_size) / p)
    end = int((((r + 1) * sample_size) / p))
    closeness_centrality = {x: 0 for x in range(start, end)}
    
    if r == 0:
        timer_2 = time.time()
    for i in range(start, end):
        sp = helpers.bfs_basic(g, i)
        for a, b in sp.items():
            closeness_centrality[i] += b
        closeness_centrality[i] = int(((n - 1) / closeness_centrality[i]) * 1000) / 1000
    if r == 0:
        algo_runtime = time.time() - timer_2
        
    helpers.top5_cc(closeness_centrality)
    
    if r == 0:
        total_time = time.time() - timer_1    
        print()
        print(f"Total runtime: {total_time:.4f} seconds")
        print(f"CC BFS Algo Runtime: {algo_runtime:.4f} seconds\n")
        print("===============================================================================")
        print("===============================================================================\n")