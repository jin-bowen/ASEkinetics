#!/bin/sh

#SBATCH -J txb
#SBATCH --time=12:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=16
#SBATCH --mem-per-cpu=4gb

ind=$1
dir=/mnt/pan/Data14/1kgp

reads=${dir}/${ind}/${ind}.mtx
ase=${dir}/${ind}/${ind}.ase
cb=${dir}/${ind}/${ind}.cb
ase_infer=${dir}/${ind}/${ind}.ase.infer

python $HOME/ASEkinetics/lib/ase_infer.py ${reads} ${ase} ${cb} ${ind} ${dir}/${ind}
python $HOME/ASEkinetics/lib/txburst.py ${ase_infer} ${ind} ${dir}/${ind}


