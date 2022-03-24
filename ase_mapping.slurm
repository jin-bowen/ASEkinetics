#!/bin/sh

#SBATCH -J ASE_mapping
#SBATCH --time=1-12:00:00
#SBATCH -N 1
#SBATCH -n 5
#SBATCH --mem=30gb

module load bedtools
sample=$1
dir=/mnt/pan/Data14/1kgp/NA12878
#dir=/mnt/pan/Data14/1kgp/${sample}
vcf=${dir}/${sample}.GRCh38.phased.vcf.gz

#intersectBed -a ${vcf} -b ~/genome_ref/ref_GRCh38_genes.bed -wa -wb |  grep -e "0|1" -e "1|0" > ${dir}/${sample}.GRCh38.phased.het.orf.bed
#bgzip ${dir}/${sample}.GRCh38.phased.het.orf.bed
#zcat ${dir}/${sample}.GRCh38.phased.het.orf.bed | cut -f1,2 > ${dir}/${sample}.GRCh38.phased.het.pos

vcf_mapped=${dir}/${sample}.GRCh38.phased.het.orf.bed.gz
pos=${dir}/${sample}.GRCh38.phased.het.pos

# pile up at snp
# need to change for vcf file 
samtools mpileup -q50 -Q30 -f $HOME/refdata-cellranger-GRCh38-3.0.0/fasta/genome.fa \
	--position  ${pos}  --output-extra QNAME,CB,UB,GX \
	--output $dir/${sample}.mpileup \
	$dir/outs/possorted_genome_bam.bam

perl $HOME/ASE/lib/pileup2base.pl $dir/${sample}.mpileup 33 $dir/${sample}.parser

pileup=${dir}/${sample}.parser
python $HOME/ASE/lib/ase.py ${pileup} ${vcf_mapped} ${dir}/${sample}

# get mtx
#python $HOME/ASE/lib/csr_dense.py ${dir}/outs/filtered_feature_bc_matrix \
#		$HOME/refdata-cellranger-GRCh38-3.0.0/mitochondrial.gene \
#		$dir/${sample}


