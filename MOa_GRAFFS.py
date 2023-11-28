import networkx as nx

f = open("assets/sample3.txt", "w")

g = nx.gnp_random_graph(1107, 0.4, directed = False)

edges = list(g.edges())

for edge in edges:
    s = f"{edge[0]} {edge[1]}\n"
    f.write(s)

print(g.number_of_nodes())
print(g.number_of_edges())
    
f.close()