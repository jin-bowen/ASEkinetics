#!/bin/sh

#SBATCH -J txb
#SBATCH --time=3-12:00:00
#SBATCH -N 1
#SBATCH -n 10
#SBATCH --mem=32gb

ind=$1
dir=/mnt/pan/Data14/hPSC

# get mtx
python $HOME/ASE/lib/sparse.py $dir/${ind}/outs/filtered_feature_bc_matrix $dir/${ind}/${ind}

reads=${dir}/${ind}/${ind}.mtx
ase=${dir}/${ind}/${ind}.ase
cb=${dir}/${ind}/${ind}.cb

python $HOME/ASE/lib/txburst.py ${reads} ${ase} ${cb} ${ind} ${dir}/${ind}

