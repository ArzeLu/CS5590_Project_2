from mpi4py import MPI
import read_edges as re
import networkx as nx
import time
import bfs

#set up MPI
comm = MPI.COMM_WORLD
p = comm.Get_size()
r = comm.Get_rank()

def bc_bfs(g):
    # measures total runtime
    if r == 0:
        timer_1 = time.time() 
        print("\n======================Betweenness Centrality with BFS========================")
        print("===============================================================================")
        
    n = g.number_of_nodes()
    
    # Divide the workload for each processor
    sample_size = n
    start = int((r * sample_size) / p)
    end = int((((r + 1) * sample_size) / p))
    
    asp = [] # all shortest paths (node: path length)
    asp_inv = [] # all shortest paths inverse (path length: node)
    
    # With BFS, calculate shortest paths of all nodes.
    for i in range(start, end):
        sp = bfs.get_bfs(g, i) # shortest paths (node: shortest path to source i)
        asp.append(sp[0])
        asp_inv.append(sp[1])
        
    
    # Send every asp to processor 0
    if r != 0:
        comm.send(obj = asp, dest = 0, tag = 1)
        comm.send(obj = asp_inv, dest = 0, tag = 2)
    else:
        for pi in range(1, p):
            asp += comm.recv(source = pi, tag = 1)
            asp_inv += comm.recv(source = pi, tag = 2)
        
    asp = comm.bcast(obj = asp, root = 0)
    asp_inv = comm.bcast(obj = asp_inv, root = 0)
        
    betweenness_centrality = {x: 0 for x in range(start, end)}
    normalizer = 2 / ((n - 1) * (n - 2))
    
    #Start runtime measurement. Only measures the main floyd-warshall part
    if r == 0:
        print(f"\nProcessed {sample_size} nodes; {int(sample_size / n * 100)}% of total {n} nodes")
        timer_2 = time.time()
        
    # Algo for extracting betweenness from BFS
    # *Ref #
    for k in range(start, end):  
        for i in range(n):
            if i == k:
                continue
            if asp_inv[i].get(1) == 1:
                continue
                
            sp_ki = asp[k].get(i) # sp of (k, i)    
            
            # Don't go through 0 to n because duplicate paths
            for j in range(i + 1, n):
                if j == k:
                    continue
                    
                sp_kj = asp[k].get(j) # sp of (k, j)
                sp_ij = asp[i].get(j) # sp of (i, j)
                
                # *Ref #
                if sp_ij < (sp_ki + sp_kj):
                    continue
                
                sps_through_k = 0
                sps = 0
                
                # sps of (i, j) through k
                for x in asp_inv[k][sp_kj - 1]:
                    if asp[x][j] == 1:
                        sps_through_k += 1
                        
                # sps of (i, j)
                # *Ref #
                for x in asp_inv[i][sp_ij - 1]:
                    if asp[x][j] == 1:
                        sps += 1
                
                #print(f"({k}, {i}, {j}) => spstk: {sps_through_k}, sps: {sps}")
                betweenness_centrality[k] += (sps_through_k / sps)
                
        betweenness_centrality[k] *= normalizer
        
    #End algo runtime measurement
    if r == 0:                    
        algo_runtime = time.time() - timer_2
    
    if r != 0:
        comm.send(obj = betweenness_centrality, dest = 0, tag = 3)
    else:
        for pi in range(1, p):
            betweenness_centrality.update(comm.recv(source = pi, tag = 3))
            
        betweenness_centrality = dict(sorted(betweenness_centrality.items(), key = lambda x: x[1], reverse = True)) #sort dictionary 
        
        # Get top 5
        top5 = [[] for _ in range(5)]
        bc_set = set()
        for x, y in betweenness_centrality.items():
            bc_set.add(y)
            if len(bc_set) == 6:
                break
            top5[len(bc_set) - 1].append((x, y))
        
        print("\n>>>>> Top 5: <<<<<")
        for i in range(len(top5)):
            print(f"#{i}:")
            for y in top5[i]:
                print(f"    {y[0]}, {y[1]}")
        
        if r == 0:
            total_time = time.time() - timer_1    
            print()
            print(f"Total runtime: {total_time:.4f} seconds")
            print(f"BC BFS Algo Runtime: {algo_runtime:.4f} seconds\n")
            print("===============================================================================")
            print("===============================================================================\n")    
        
    return betweenness_centrality