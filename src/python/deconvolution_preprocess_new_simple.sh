#!/bin/sh
#SBATCH -t 00:15:00
#SBATCH -p genoa
#SBATCH --nodes 1
#SBATCH --ntasks 1
#SBATCH --cpus-per-task 8
#SBATCH --tasks-per-node 1
#SBATCH --mem=10G
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=aphrodite.provolisianou@ru.nl

ml purge;
ml load 2023;
ml load MATLAB/2023b-upd7;
source activate frogs;


which python;
p_ython deconvolution_preprocess_new_simple.py;
