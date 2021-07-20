#!/bin/sh

#SBATCH -J ASE_mapping
#SBATCH --time=3-12:00:00
#SBATCH -N 1
#SBATCH -n 1
#SBATCH --mem=32gb

ind=$1
dir=/mnt/pan/Data14/1kgp
vcf=${dir}/${ind}/${ind}_wgs_grch38_het.vcf
refFlat=$HOME/refdata-cellranger-GRCh38-3.0.0/genes/genes.gtf

pileup=${dir}/${ind}/sub_parser/${ind}.parser${SLURM_ARRAY_TASK_ID}
python $HOME/ASE/lib/ase.py ${pileup} ${vcf} ${dir}/${ind}/sub_parser/${ind}_${SLURM_ARRAY_TASK_ID}
#cat ${dir}/${ind}/sub_parser/${ind}_*.ase > ${dir}/${ind}/${ind}.ase




