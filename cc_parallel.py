import math
import time
import networkx as nx
import read_edges as re
import bfs
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
                
            
    # Send closeness_centrality data to p_0
    if r != 0:
        comm.send(obj = closeness_centrality, dest = 0, tag = 3)
    else:
        for pi in range(1, p):
            closeness_centrality.update(comm.recv(source = pi, tag = 3))
      
        closeness_centrality = dict(sorted(closeness_centrality.items(), key = lambda x: x[1], reverse = True))
   
        # Get top 5
        top5 = [[] for _ in range(5)]
        cc_set = set()
        for x, y in closeness_centrality.items():
            cc_set.add(y)
            if len(cc_set) == 6:
                break
            top5[len(cc_set) - 1].append((x, y))
        
        print("\n>>>>> Top 5: <<<<<")
        for i in range(len(top5)):
            print(f"#{i}:")
            for y in top5[i]:
                print(f"    {y[0]}, {y[1]}")
    
        total_time = time.time() - timer_1    
        print()
        print(f"Total runtime: {total_time:.4f} seconds")
        print(f"CC Floyd-Warshall Algo Runtime: {algo_runtime:.4f} seconds\n")
        print("===============================================================================")
        print("===============================================================================\n")
        
    return closeness_centrality    
    
# closeness centrality with bfs
def breadth_first_search():
    return