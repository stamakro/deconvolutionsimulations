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
python deco_1.py /projects/0/AdamsLab/Scripts/afroditi/afroditi/mixout/sim1_part0.csv outpathfordeco_1
