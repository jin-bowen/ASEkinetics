#!/bin/sh

#SBATCH -J kpe_est
#SBATCH -N 1 -n 4
#SBATCH --mem=32g
#SBATCH --time=4-12:00:00

ind=$1
dir=`pwd`
ase_infer=${dir}/${ind}/${ind}.ase.infer
python $HOME/ASE/lib/kpe.py ${ase_infer} full ${dir}/${ind}/${ind} &
python $HOME/ASE/lib/kpe.py ${ase_infer} pb   ${dir}/${ind}/${ind} &

wait

