#!/bin/sh

#SBATCH -J kpe_var
#SBATCH -N 1 -n 4
#SBATCH --mem-per-cpu=16g
#SBATCH --time=4-12:00:00

sample=$1
sub_sample=$2
method=$3

echo $sample $sub_sample $method

ase=${dir}/${sample}/${sub_sample}.ase.reform
kpe=${dir}/${sample}/${sub_sample}_${method}.est
out_eval=${dir}/${sub_sample}_${method}.eval

python ${dir}/ASEkinetics/lib/evaluation.py ${kpe} ${method} ${ase} ${out_eval} 
python ${dir}/ASEkinetics/lib/simulation_compare.py ${out_eval} ${dir}/${sub_sample}_${method}
