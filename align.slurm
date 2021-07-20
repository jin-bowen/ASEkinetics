#!/bin/sh

#SBATCH -J align
#SBATCH --time=5-12:00:00
#SBATCH -N 1
#SBATCH -n 8
#SBATCH --mem=32gb

ind=$1
dir=/mnt/pan/Data14/1kgp

cellranger count --id=sc_${ind} --nosecondary \
                   --transcriptome=$HOME/refdata-cellranger-GRCh38-3.0.0 \
                   --fastqs=$dir/fastq/${ind}

echo 'End of SLURM'

