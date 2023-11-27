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
        
    betweenness_centrality = [0] * (end - start)
    normalizer = 2 / ((n - 1) * (n - 2))
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
        
    for i in range(n):
        print(f"{i}, {betweenness_centrality[i]}")
    
    return betweenness_centrality