#!/bin/sh

#SBATCH -J sra
#SBATCH --time=5-12:00:00
#SBATCH --nodes=10
#SBATCH --cpus-per-task=20
#SBATCH --mem=16gb

module load sratoolkit
#fastq-dump --outdir fastq/NA18502 --gzip --readids --read-filter pass --dumpbase --split-3 --clip SRR8551676
fastq-dump --outdir ~/GM18502_H3K27ac/ --gzip --readids --read-filter pass --dumpbase --split-3 --clip SRR1463109

