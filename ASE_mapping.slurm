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

# add gene annotation
TagReadWithGeneExonFunction I=$dir/${ind}/outs/possorted_genome_bam.bam \
	O=$dir/${ind}/outs/annotated_genome_bam.bam \
	ANNOTATIONS_FILE=${refFlat} \
	TAG=GE

# pile up at snp
# need to change for vcf file 
samtools mpileup -q50 -f $HOME/refdata-cellranger-GRCh38-3.0.0/fasta/genome.fa \
	--position  ${vcf}  --output-extra QNAME,CB,UB,GX \
	--output $dir/${ind}/${ind}_het.mpileup \
	$dir/${ind}/outs/annotated_genome_bam.bam

perl $HOME/ASE/lib/pileup2base.pl $dir/${ind}/${ind}_het.mpileup \
	 33  $dir/${ind}/${ind}.parser

# get mtx
python $HOME/ASE/lib/csr_dense.py $dir/${ind}/outs/filtered_feature_bc_matrix \
               $HOME/refdata-cellranger-GRCh38-3.0.0/mitochondrial.gene \
               $dir/${ind}/${ind}

