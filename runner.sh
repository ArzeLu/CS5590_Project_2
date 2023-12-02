#!/bin/bash
#SBATCH --job-name=TT
#SBATCH --partition=compute
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --ntasks-per-node=1
#SBATCH --time=40:00:00
#SBATCH --mem=30GB
#SBATCH -o output/twitter/%x_%a.o

/apps/anaconda3/envs/arzeEnv/bin/mpirun -n $SLURM_ARRAY_TASK_ID python3 main.py 1 twitter_combined
##python3 main.py 1 twitter_combined
