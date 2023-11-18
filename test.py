from mpi4py import MPI

import read_edges as re
import math
import time
import networkx as nx
import random

def test():   
    comm = MPI.COMM_WORLD
    p = comm.Get_size()
    r = comm.Get_rank()
    
    n = 20
    start = int((r * n) / p)
    end = int((((r + 1) * n) / p))
    
    assignment = []
    
    for idx in range(start, end):
        assignment.append((idx, r))
    
    if r != 0:
        comm.send(obj = assignment, dest = 0, tag = 1)
    else:
        for pr in range(1, p):
            assignment += comm.recv(source = pr, tag = 1)
    
    assignment = comm.bcast(obj = assignment, root = 0)
    
    for i in range(n):
        print(assignment[i][1])
        
    if r == 1:
        print(assignment)