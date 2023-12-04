Notice: 
All three algorithms run upon calling main.
Comment out specific line in main.py to limit it.

Command format:
python main.py [portion] [file name without extension]

Example:
python main.py 0.3 facebook_combined <-- this runs V * 0.3, 30% of the entire process
python main.py 1 twitter_combined <-- this runs V, all the nodes in the file

MPI Example:
mpiexec -n 12 python main.py 1 sample3