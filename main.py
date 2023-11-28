import cc_parallel as cp
import cc_serial as cs
import bc_parallel as bp
import read_edges as re
from mpi4py import MPI

def main():
    file_name = "sample3.txt"
    g = re.get_graph(file_name)   
    
    cp.cc_fw(g)
    bp.bc_bfs(g)

if __name__ == "__main__":
    main()