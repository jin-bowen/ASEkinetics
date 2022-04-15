#!/bin/sh

#SBATCH -J ase_infer
#SBATCH --time=3-12:00:00
#SBATCH -N 1
#SBATCH -n 6
#SBATCH --mem=36gb

sample=$1
dir=/mnt/pan/Data14/1kgp/${sample}
ref=${HOME}/genome_ref/ref_GRCh38_genes.bed

reads=${dir}/${sample}.mtx
ase=${dir}/${sample}.ase
cb=${dir}/${sample}.qc

python -W ignore $HOME/ASE/lib/ase_bn.py ${reads} ${ase} ${cb} ${dir}/${sample}  
python -W ignore $HOME/ASE/lib/analysis/ase_infer_evaluation.py ${dir}/${sample}.ase.record ${ref} ${dir}/${sample}.qc 

ase_infer=${dir}/${sample}.ase.infer
python $HOME/ASE/lib/ase_reform_phased.py ${ase_infer} True ${dir}/${sample}

