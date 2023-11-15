from mpi4py import MPI

import networkx as nx
import read_edges as re
import time

def generate():
    g = re.graph_from_edges() #get the graph
    
    #set up MPI
    comm = MPI.COMM_WORLD
    size = comm.Get_size()
    rank = comm.Get_rank()