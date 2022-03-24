#!/bin/sh

#SBATCH -J validation
#SBATCH -N 2 -n 2
#SBATCH --cpus-per-task 2
#SBATCH --mem-per-cpu 6g
#SBATCH --time=5-12:00:00

method=$2
sample=$1
dir=/mnt/pan/Data14/1kgp/NA12878
#dir=/mnt/pan/Data14/1kgp/${sample}

kpe=${dir}/sub_est/${sample}_${method}.est${SLURM_ARRAY_TASK_ID}
ase=${dir}/sub_ase/${sample}.ase.reform${SLURM_ARRAY_TASK_ID}

out_var=${dir}/${sample}_${method}.var${SLURM_ARRAY_TASK_ID} 
python $HOME/ASE/lib/variance.py ${kpe} ${ase} ${out_var} &

out_eval=${dir}/${sample}_${method}.eval${SLURM_ARRAY_TASK_ID} 
python $HOME/ASE/lib/evaluation.py ${kpe} ${method} ${ase} ${out_eval} &
wait

