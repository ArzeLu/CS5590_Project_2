import sys
import cc_parallel as cp
import cc_serial as cs
import bc_parallel as bp
import read_edges as re
from mpi4py import MPI

def main():
    file_name = "twitter_combined.txt"
    g = re.get_graph(file_name)  
    n = g.number_of_nodes()
    
    if len(sys.argv) < 2:
        print("Please specify sample size (float number)")
        print("e.g. python main.py 0.6")
        print("Exiting...")
        sys.exit(1)
    
    portion = 0
    try:
        portion = float(sys.argv[1])
    except:
        print("Please input a float number instead.")
        sys.exit(1)
    
    cp.cc_fw(g, int(n * portion))
    bp.bc_bfs(g, int(n * portion))

if __name__ == "__main__":
    main()