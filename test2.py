i = open("assets/twitter_combined.txt", "r")
o = open("assets/replaced.txt", "w")

edges = i.readlines()

for edge in edges:
    edge = edge.split(" ")
    o.write(f"{int(edge[0]) - 12} {int(edge[1]) - 12}\n")
    
    
i.close()
o.close()