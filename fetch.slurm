#!/bin/sh

#SBATCH -J sra
#SBATCH --time=5-12:00:00
#SBATCH --nodes=1
#SBATCH --cpus-per-task=2
#SBATCH --mem-per-cpu=32gb

module load sratoolkit
fastq-dump --outdir fastq/NA18502 --gzip --readids --read-filter pass --dumpbase --split-3 --clip SRR10428402 &
fastq-dump --outdir fastq/NA12878 --gzip --readids --read-filter pass --dumpbase --split-3 --clip SRR8551677 &
wait
