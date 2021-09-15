#!/bin/sh

#SBATCH -J kpe_var
#SBATCH -N 1 -n 4
#SBATCH --mem-per-cpu=16g
#SBATCH --time=4-12:00:00

sample=$1
method=$2

dir=`pwd`
ase=${dir}/${sample}/${sample}.ase.reform
kpe=${dir}/${sample}/${sample}_${method}.est
out_var=${dir}/${sample}/${sample}_${method}.var

python $HOME/ASE/lib/variance.py ${kpe} ${ase} ${out_var} 

