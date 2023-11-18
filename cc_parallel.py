import math
import time
import matplotlib.pyplot as plt
import networkx as nx
import read_edges as re
from mpi4py import MPI

def floyd_warshall(file_name):
    #set up MPI
    comm = MPI.COMM_WORLD
    p = comm.Get_size()
    r = comm.Get_rank()
    
    g = re.graph_from_edges(file_name)   
    
    #uncomment here for sample
    #g = nx.Graph()
    #edges = [(0, 3), (0, 4), (1, 2), (1, 3), (3, 6), (3, 7), (4, 8), (5, 8), (5, 7), (6, 7)]
    #g.add_edges_from(edges)
    
    n = g.number_of_nodes()
    
    dist = [[math.inf] * n for _ in range(n)] #initialize the 2D array with infinity. By qtniko
       
    #Divide the workload
    sample_size = 100
    start = int((r * sample_size) / p)
    end = int((((r + 1) * sample_size) / p))
    
    
    #================== Calculate for Task Assignments ================"
    assignment = []
    
    for idx in range(start, end):
        assignment.append((idx, r))
            
    if r != 0:
        comm.send(obj = assignment, dest = 0, tag = 1)
    else:
        for pi in range(1, p):
            assignment += comm.recv(source = pi, tag = 1)
            
    assignment = comm.bcast(obj = assignment, root = 0)
    #=================================================================="
    
    
    #filling weights (= 1) for each edge, and fill zeros for diagonal
    for u in range(n):
        dist[u][u] = 0
        for v in g.neighbors(u):
            dist[u][v] = 1
    
    #Start runtime measurement. Only measures the main floyd-warshall part
    if r == 0:
        print(f"Processed {sample_size} nodes; {int(sample_size / n * 100)}% of total nodes")
        start_time = time.time()
        
    #floyd-warshall algo
    #parallelize logic in README
    for k in range(sample_size):
        original_impact_row = []
        updated_impact_row = []
        
        #pre-calculate the impacting row,
        #then send the row data to all processors
        if r == assignment[k][1]: 
            original_impact_row = dist[k]  
            
            for j in range(n):
                updated_impact_row.append(min(dist[k][j], (dist[k][k] + dist[k][j])))
            
        original_impact_row = comm.bcast(obj = original_impact_row, root = assignment[k][1])
        updated_impact_row = comm.bcast(obj = updated_impact_row, root = assignment[k][1])
                
        for i in range(start, end):
            for j in range(n):
                if i <= k:
                    dist[i][j] = min(dist[i][j], (dist[i][k] + original_impact_row[j]))
                elif i > k:
                    dist[i][j] = min(dist[i][j], (dist[i][k] + updated_impact_row[j]))
    
    #End runtime measurement
    if r == 0:                    
        print("runtime:", time.time() - start_time)
        
    #Path lengths logic in README
    path_lengths = [0] * n
    
    for i in range(start, end):
        for j in range(n):
            path_lengths[i] += dist[i][j]
            
    #Send matrix data to p_0
    if r != 0:
        comm.send(obj = path_lengths, dest = 0, tag = 4)
    else:
        for pi in range(1, p):
            other_path_lengths = comm.recv(source = pi, tag = 4)
            for i in range(n):
                path_lengths[i] += other_path_lengths[i]
                
    #if r == 0:
        #print(path_lengths)