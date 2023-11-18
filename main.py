import time

import test
import cc_parallel as ccp
import cc_serial as ccs
from mpi4py import MPI

def main():
    #Get rank info
    comm = MPI.COMM_WORLD
    r = comm.Get_rank()
    
    file_name = "facebook_combined.txt"
    
    #ccs_result = ccs.generate(file_name)
    #ccp_result = ccp.generate(file_name)

    ccp.floyd_warshall(file_name)

    #test.test()
    
if __name__ == "__main__":
    main()