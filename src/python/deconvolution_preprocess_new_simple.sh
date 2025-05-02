#!/bin/sh
#SBATCH -t 01:30:00
#SBATCH -p genoa
#SBATCH --nodes 1
#SBATCH --ntasks 1
#SBATCH --cpus-per-task 8
#SBATCH --tasks-per-node 1
#SBATCH --mem=10G
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=s.makrodimitris@erasmusmc.nl

ml purge;
ml load 2023;
ml load MATLAB/2023b-upd7;
ml load Anaconda3/2023.07-2;


python deconvolution_preprocess_new_simple.sh;
