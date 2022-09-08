#!/bin/sh

#SBATCH -J ASE_mapping
#SBATCH --time=1-12:00:00
#SBATCH -N 1
#SBATCH -n 5
#SBATCH --mem=30gb

#############
sample=$1
sc_dir=
ref_dir=
ASE_dir=
method=pb
##############

vcf=${sc_dir}/${sample}.GRCh38.phased.vcf.gz

intersectBed -a ${vcf} -b ${ref_dir}/ref_GRCh38_genes.bed -wa -wb | \
	 grep -e "0|1" -e "1|0" > ${sc_dir}/${sample}.GRCh38.phased.het.orf.bed
bgzip ${sc_dir}/${sample}.GRCh38.phased.het.orf.bed
zcat ${sc_dir}/${sample}.GRCh38.phased.het.orf.bed | cut -f1,2 > ${sc_dir}/${sample}.GRCh38.phased.het.pos

vcf_mapped=${sc_dir}/${sample}.GRCh38.phased.het.orf.bed.gz
pos=${sc_dir}/${sample}.GRCh38.phased.het.pos

# pile up at snp
# need to change for vcf file 
samtools mpileup -q50 -Q30 -f ${ref_dir}/refdata-cellranger-GRCh38-3.0.0/fasta/genome.fa \
	--position  ${pos}  --output-extra QNAME,CB,UB,GX \
	--output $dir/${sample}.mpileup \
	$dir/outs/possorted_genome_bam.bam

perl ${ASE_dir}/lib/pileup2base.pl $dir/${sample}.mpileup 33 $dir/${sample}.parser

pileup=${sc_dir}/${sample}.parser
python ${ASE_dir}/lib/aberrant_read.py ${pileup} ${vcf_mapped} ${sc_dir}/${sample}

# get mtx
python ${ASE_dir}/lib/csr_dense.py ${sc_dir}/outs/filtered_feature_bc_matrix \
		$HOME/refdata-cellranger-GRCh38-3.0.0/mitochondrial.gene \
		${sc_dir}/${sample}

############################################################################

reads=${sc_dir}/${sample}.mtx
ase=${sc_dir}/${sample}.ase
cb=${sc_dir}/${sample}.qc

python -W ignore ${ASE_dir}/lib/ase_bn.py ${reads} ${ase} ${cb} ${sc_dir}/${sample}  
python -W ignore ${ASE_dir}/lib/analysis/ase_infer_evaluation.py \
	${sc_dir}/${sample}.ase.record ${ref_dir}/ref_GRCh38_genes.bed ${sc_dir}/${sample}.qc 

ase_infer=${sc_dir}/${sample}.ase.infer
python ${ASE_dir}/lib/ase_reform_phased.py ${ase_infer} True ${sc_dir}/${sample}

##############################################################################

python ${ASE_dir}/lib/kpe.py ${sc_dir}/${sample} pb 

pb_est=${sc_dir}/${sample}_pb.est
ase_reform=${sc_dir}/${sample}.ase.reform

kpe=${sc_dir}/sub_est/${sample}_${method}.est
ase=${sc_dir}/${sample}.ase.reform

out_var=${sc_dir}/${sample}_${method}.var 
python ${ASE_dir}/lib/variance.py ${kpe} ${ase} ${out_var} 

out_eval=${sc_dir}/${sample}_${method}.eval 
python ${ASE_dir}/lib/evaluation.py ${kpe} ${method} ${ase} ${out_eval}

