#!/bin/sh

#SBATCH -J ASE_mapping
#SBATCH --time=3-12:00:00
#SBATCH -N 1
#SBATCH -n 1
#SBATCH --mem=32gb

ind=$1
dir=/mnt/pan/Data14/hPSC
vcf=${dir}/WTC_wgs_grch38_het.vcf

#refFlat=$HOME/ASE/refdata-cellranger-GRCh38-3.0.0/hg38.refFlat
#
## add gene annotation
#TagReadWithGeneExonFunction I=$dir/${ind}/outs/possorted_genome_bam.bam \
#	O=$dir/${ind}/outs/annotated_genome_bam.bam \
#	ANNOTATIONS_FILE=${refFlat} \
#	TAG=GE
#
## pile up at snp
## need to change for vcf file 
#samtools mpileup -q50 -f $HOME/ASE/refdata-cellranger-GRCh38-3.0.0/fasta/genome.fa \
#	--position  $dir/wgs_rosmap.vcf  --output-extra QNAME,CB,UB,GE \
#	--output $dir/${ind}/${ind}_het.mpileup \
#	$dir/${ind}/outs/annotated_genome_bam.bam
#
#perl $HOME/ASE/lib/pileup2base.pl $dir/${ind}/${ind}_het.mpileup \
#	 33  $dir/${ind}/${ind}.parser

pileup=${dir}/${ind}/${ind}.parser
python $HOME/ASE/lib/ase.py ${pileup} ${vcf} ${dir}/${ind}/${ind}

