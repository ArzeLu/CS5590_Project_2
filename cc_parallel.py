from mpi4py import MPI

import networkx as nx
import read_edges as re
import time


def generate(file_name):
    # set up MPI
    comm = MPI.COMM_WORLD
    p = comm.Get_size()  # number of processors
    r = comm.Get_rank()  # rank of current processor

    # split graph across processors by reading n / p lines from the file for each processor
    g = re.graph_from_edges(file_name)

    starting_index = int((r * g.number_of_nodes()) / p)
    ending_index = int((((r + 1) * g.number_of_nodes()) / p) - 1)

    if r == p - 1:
        ending_index = g.number_of_nodes()

    # begin calculating closeness centrality
    cc = []  # closeness centrality of all vertices
    times = []  # runtime of each vertex

    top_five = []

    sample_size = 5

    for i in range(starting_index, starting_index + sample_size):
        sum_of_shortest_paths = 0
        start_time = time.time()

        for j in range(g.number_of_nodes()):
            if i == j:
                continue
            sum_of_shortest_paths += (len(nx.dijkstra_path(g, i, j)) - 1)
        average_shortest_path = sum_of_shortest_paths / (g.number_of_nodes() - 1)
        closeness = 1 / average_shortest_path
        cc.append((i, closeness))
        if len(top_five) < 5:
            top_five.append((i, closeness))
        else:
            for j in top_five:
                if closeness > j[1]:
                    top_five = list(map(lambda x: x.replace(j, (i, closeness))))
                    break
        times.append(time.time() - start_time)
    '''
    print("----Closeness Centrality Parallel----")
    print("Runtime of each vertex:")

    for t in times:
        print(t)
    '''

    if r != 0:
        comm.send(cc, dest=0, tag=1)
        comm.send(top_five, dest=0, tag=2)
        comm.send(times, dest=0, tag=3)
    else:
        for i in range(1, p):
            data = comm.recv(source=i, tag=1)
            five = comm.recv(source=i, tag=2)
            t = comm.recv(source=i, tag=3)
            cc.extend(data)
            top_five.extend(five)
            times.extend(t)
        top_five.sort(reverse=True)
        print("\nAverage time:", (sum(times) / (sample_size * p)))
        f = open('output.txt', 'w')
        f.write('---Top Five---\n')
        for i in top_five[:5]:
            print(i)
            f.write(str(i) + '\n')
        f.write('\n---Results---\n')
        for i in cc:
            f.write(str(i) + '\n')
        f.close()
        return cc


generate('facebook_combined.txt')
