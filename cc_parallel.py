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
    
    n = g.number_of_nodes()
    
    dist = [[math.inf] * n for _ in range(n)] #initialize the 2D array with infinity. By qtniko
       
    #Divide the workload
    sample_size = 360
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
    for u in range(start, end):
        dist[u][u] = 0
        for v in g.neighbors(u):
            dist[u][v] = 1
    
    #Start runtime measurement. Only measures the main floyd-warshall part
    if r == 0:
        print(f"\nProcessed {sample_size} nodes; {int(sample_size / n * 100)}% of total {n} nodes")
        start_time = time.time()
        
    impact_row = []
    
    #floyd-warshall algo
    #parallelize logic in README
    for k in range(sample_size):        
    
        #pre-calculate the impacting row,
        #then send the row data to all processors
        if r == assignment[k][1]: 
            impact_row = dist[k].copy()
            updated_impact_row = [min(dist[k][j], (dist[k][k] + dist[k][j])) for j in range(n)] #calculate updated_impact_row
            
            #Send impact row to other processors
            for pi in range(p):
                if pi < r:
                    comm.send(obj = impact_row, dest = pi, tag = 2)
                elif pi > r:
                    comm.send(obj = updated_impact_row, dest = pi, tag = 2)
                
        else:
            impact_row = comm.recv(source = assignment[k][1], tag = 2)
            
        for i in range(start, end):
            for j in range(n):
                dist[i][j] = min(dist[i][j], (dist[i][k] + impact_row[j]))
    
    #End runtime measurement
    if r == 0:                    
        runtime = time.time() - start_time
        print(f"Floyd-Warshall Runtime: {runtime:.4f} seconds")
        
    #Path lengths logic in README
    path_lengths = [0] * n
    
    for i in range(start, end):
        for j in range(n):
            if dist[i][j] != math.inf:
                path_lengths[i] += dist[i][j]
            
    #Send path_lengths data to p_0
    if r != 0:
        comm.send(obj = path_lengths, dest = 0, tag = 3)
    else:
        for pi in range(1, p):
            other_path_lengths = comm.recv(source = pi, tag = 3)
            for i in range(n):
                path_lengths[i] += other_path_lengths[i]
    
    cc = {} #Store as dictionary to keep node data
    
    if r == 0:
        #Convert path_lengths to normalized closeness centrality (Keeping only three decimals)
        #print("\n=========Unsorted=========")
        #print("Node | Closeness Centrality")
        for i in range(n):
            if path_lengths[i] == 0:
                cc[i] = 0
            else:
                cc[i] = int(((n - 1) / path_lengths[i]) * 1000) / 1000
            #print(f"{i}, {cc[i]:.3f}")
        
        #Sorting cc dictionary
        cc = dict(sorted(cc.items(), key = lambda item: item[1], reverse = True))
        #print("\n==========Sorted==========")
        #print("Node | Closeness Centrality")
        #for x, y in cc.items():
            #print(f"{x}, {y:.3f}")
            
        #Get top 5
        top5 = []
        current_list = []
        current = 0
        for x, y in cc.items():
            if y != current:
                if len(current_list) != 0:
                    top5.append(current_list)
                    current_list = []
                
                if len(top5) == 5:
                    break

                current = y
            current_list.append((x, y))
        
        print("\n==========Top 5:===========")
        count = 1
        for x in top5:
            print(f"#{count}:")
            for y in x:
                print(f"    {y[0]}, {y[1]}")
            count += 1
        
    return cc    