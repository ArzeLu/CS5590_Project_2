from mpi4py import MPI
import sys
import closeness_centrality_parallel as cp
import betweenness_centrality_parallel as bp
import helpers

def main():
    file_name = sys.argv[2] + ".txt"
    
    g = helpers.get_graph(file_name)  
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
    
    #cp.cc_fw(g, int(n * portion))
    cp.cc_bfs(g, int(n * portion))
    bp.bc_bfs(g, int(n * portion))

if __name__ == "__main__":
    main()