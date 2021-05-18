#!/bin/sh

#SBATCH -J align
#SBATCH --time=5-12:00:00
#SBATCH -N 4
#SBATCH -n 10
#SBATCH --mem=32gb

ind=$1
dir=$HOME/fastq
ref=$HOME/refdata-cellranger-GRCh38-3.0.0

cellranger mkgtf Homo_sapiens.GRCh37.87.gtf Homo_sapiens.GRCh37.87.filtered.gtf \
                 --attribute=gene_biotype:protein_coding \
                 --attribute=gene_biotype:lincRNA \
                 --attribute=gene_biotype:antisense

cellranger mkref --genome=hg19 \
                 --fasta=Homo_sapiens.GRCh37.dna.primary_assembly.fa \
                 --genes=Homo_sapiens.GRCh37.87.filtered.gtf \
                 --ref-version=3.0.0

cellranger count --id=${ind} \
		--transcriptome=${ref} \
		--fastqs=$dir/${ind}
		--nosecondary

echo 'End of SLURM'


