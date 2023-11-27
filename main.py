import time
import cc_parallel as cp
import cc_serial as cs
import bc_p as bp
import read_edges as re
from mpi4py import MPI

import bc_test as bt

def main():
    file_name = "sample.txt"
    g = re.get_graph(file_name)   
    
    #cp.floyd_warshall(g)
    #startTime = time.time()
    #bp.bc_bfs(g)
    #print("\n\n",time.time() - startTime)
    
    bt.betweenness_centrality(g)
if __name__ == "__main__":
    main()