from mpi4py import MPI
import networkx as nx

# set up MPI
comm = MPI.COMM_WORLD
p = comm.Get_size()
r = comm.Get_rank()

def get_graph(file_name):   
    f = open(f"assets/{file_name}", "r")
    edges = f.readlines()
    g = nx.Graph()
    
    for edge in edges:
        edge = edge.split(" ")
        g.add_edge(int(edge[0]), int(edge[1]))
    
    f.close()
    return g

# Set up node data
# Keeping two dictionaries just in case graph nodes aren't purely sequential
def get_node_data(g):
    n = g.number_of_nodes()
    nl = list(g.nodes()) # node list
    nl = {x: nl[x] for x in range(n)} # node list as dictionary (index: node)
    nl_inv = {y: x for x, y in nl.items()} # inverse node dictionary (node: index)
    return (n, nl, nl_inv)
    
# Used for floyd warshall
# maps each node to the processor rank that is responsible for it,
# the node is the index,
# then merge lists in the order of the ranks so the index matches with node,
# then broadcast to all processors.
# Saves calculation on determining who is responsible for an impact row later
def assign_tasks(r, p, start, end):
    assignment = [r] * (end - start)
            
    if r != 0:
        comm.send(obj = assignment, dest = 0, tag = 1)
    else:
        for pi in range(1, p):
            assignment += comm.recv(source = pi, tag = 1)
            
    assignment = comm.bcast(obj = assignment, root = 0)
    return assignment
    
# Calculates and prints top 5
def top5_cc(data, nl):
    comm = MPI.COMM_WORLD
    p = comm.Get_size()
    r = comm.Get_rank()
    
    if r != 0:
        comm.send(obj = data, dest = 0, tag = 3)
    else:
        for pi in range(1, p):
            data.update(comm.recv(source = pi, tag = 3))
      
        data = dict(sorted(data.items(), key = lambda x: x[1], reverse = True))
   
        # Get top 5
        top5 = [[] for _ in range(5)]
        cc_set = set()
        for x, y in data.items():
            cc_set.add(y)
            if len(cc_set) == 6:
                break
            top5[len(cc_set) - 1].append((x, y))
        
        print("\n>>>>> Top 5: <<<<<")
        for i in range(len(top5)):
            print(f"#{i}:")
            for y in top5[i]:
                print(f"    {nl.get(y[0])}, {y[1]}")
    return
    
# Calculates and prints top 5                
def top5_bc(data, nl):
    comm = MPI.COMM_WORLD
    p = comm.Get_size()
    r = comm.Get_rank()
    n = len(nl)
    normalizer = 1 / ((n - 1) * (n - 2))
    
    if r != 0:
        comm.send(obj = data, dest = 0, tag = 3)
    else:
        for pi in range(1, p):
            other_data = comm.recv(source = pi, tag = 3)
            for i in range(len(data)):
                data[i] += other_data[i]
        
        data = dict(sorted(data.items(), key = lambda x: x[1], reverse = True))
        
        for i in range(len(data)):
            data[i] *= normalizer
        
        # Get top 5
        top5 = [[] for _ in range(5)]
        cc_set = set()
        for x, y in data.items():
            cc_set.add(y)
            if len(cc_set) == 6:
                break
            top5[len(cc_set) - 1].append((x, y))
        
        print("\n>>>>> Top 5: <<<<<")
        for i in range(len(top5)):
            print(f"#{i}:")
            for y in top5[i]:
                print(f"    {nl.get(y[0])}, {y[1]}")
    return