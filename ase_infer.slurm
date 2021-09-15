#!/bin/sh

#SBATCH -J ASE_mapping
#SBATCH --time=3-12:00:00
#SBATCH -N 1
#SBATCH -n 1
#SBATCH --mem=32gb

ind=$1
dir=/mnt/pan/Data14/1kgp
vcf=${dir}/${ind}/${ind}_wgs_grch38_het.vcf

pileup=${dir}/${ind}/${ind}.parser
python $HOME/ASE/lib/ase.py ${pileup} ${vcf} ${dir}/${ind}

reads=${dir}/${ind}/${ind}.mtx
ase=${dir}/${ind}/${ind}.ase
cb=${dir}/${ind}/${ind}.qc
python $HOME/ASE/lib/ase_infer.py ${reads} ${ase} ${cb} ${ind} ${dir}/${ind}
ase_infer=${dir}/${ind}/${ind}.ase.infer
python $HOME/ASE/lib/ase_reform.py ${ase_infer} ${dir}/${ind}/${ind}

