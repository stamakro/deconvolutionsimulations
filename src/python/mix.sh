#!/bin/sh
#SBATCH -t 00:30:00
#SBATCH -p genoa
#SBATCH --nodes 1
#SBATCH --ntasks 1
#SBATCH --cpus-per-task 8
#SBATCH --tasks-per-node 1
#SBATCH --mem=10G
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=aphrodite.provolisianou@ru.nl


source activate medseq;

python mix.py files.txt output.json


