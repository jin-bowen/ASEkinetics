from itertools import compress
import pandas as pd
import numpy as np
import sys 
import re

def ase_by_gene(sub_df,gene):

	ase_gene_df = pd.DataFrame()
	for irow, row in sub_df.iterrows():
		pile_df = pile_2_record(row)	
		ase_gene_df = ase_gene_df.append(pile_df,ignore_index=True)			

	ase_gene_df = ase_gene_df[ase_gene_df['gene']==gene]
	if ase_gene_df.empty: return pd.DataFrame()	

#	icb='AAGGCAGAGAGGTAGA-1'
#	igene='ENSG00000003402'

	ase_gene_grp = ase_gene_df.groupby(['cb','gene'])['ub_maternal','ub_paternal'].agg(set)
	ase_gene_grp.reset_index(inplace=True)

	ase_gene_grp['error_reads'] = ase_gene_grp.apply(lambda x: \
			x.ub_maternal.intersection(x.ub_paternal), axis=1)
	ase_gene_grp['error_reads'] = ase_gene_grp['error_reads'].apply(lambda x: \
				set([i for i in x if pd.notna(i)]))

	ase_gene_grp['ub_maternal'] = ase_gene_grp['ub_maternal'].apply(lambda x: \
				set([i for i in x if pd.notna(i)]))
	ase_gene_grp['ub_paternal'] = ase_gene_grp['ub_paternal'].apply(lambda x: \
				set([i for i in x if pd.notna(i)]))
	ase_gene_grp['ub_maternal_count'] = ase_gene_grp.apply(lambda x: \
			len(x.ub_maternal) - len(x.error_reads), axis=1)
	ase_gene_grp['ub_paternal_count'] = ase_gene_grp.apply(lambda x: \
			len(x.ub_paternal) - len(x.error_reads), axis=1)

	output_col = ['cb','gene','ub_maternal_count', 'ub_paternal_count']

	return ase_gene_grp[output_col]

def pile_2_record(row):

	chr = row['chr']
	loc = row['loc']
	ref = row['ref']
	alt = row['alt']
	genotype = row['genotype']

	ref_read = row[ref]
	alt_read = row[alt]

	cols = ['cb','gene','pos','allele_ref','ub_ref','allele_alt','ub_alt']
	df_bi = pd.DataFrame(columns=cols)

	if (ref_read is np.nan) and (alt_read is np.nan): return df_bi
	if ref_read is not np.nan :
		ref_read_mtx = np.array(re.split(',|:',ref_read)).reshape((-1,3))
		ref_read_df = pd.DataFrame(ref_read_mtx, columns=['cb','ub_ref','gene']).drop_duplicates()
		ref_read_df['gene'] = ref_read_df['gene'].str.split('[ ;]')
		ref_read_df = ref_read_df.explode('gene').reset_index(drop=True)
		ref_read_df['pos'] = str(chr) + ':'+ str(loc)
		ref_read_df['allele_ref'] = ref

	if alt_read is not np.nan :
		alt_read_mtx = np.array(re.split(',|:',alt_read)).reshape((-1,3))
		alt_read_df = pd.DataFrame(alt_read_mtx, columns=['cb','ub_alt','gene']).drop_duplicates()
		alt_read_df['gene'] = alt_read_df['gene'].str.split('[ ;]')
		alt_read_df = alt_read_df.explode('gene').reset_index(drop=True)
		alt_read_df['pos'] = str(chr) + ':' + str(loc)
		alt_read_df['allele_alt'] = alt

	if ref_read is np.nan:
		bi_read_df = alt_read_df.copy()
		bi_read_df['ub_ref'] = np.nan
	elif alt_read is np.nan:
		bi_read_df = ref_read_df.copy()
		bi_read_df['ub_alt'] = np.nan
	else: bi_read_df = pd.merge(ref_read_df,alt_read_df,how='outer',on=['cb','gene','pos'])
	bi_read_df['allele_ref'] = ref
	bi_read_df['allele_alt'] = alt

	if genotype == '0|1':
		bi_read_df['ub_maternal'] = bi_read_df['ub_ref']
		bi_read_df['ub_paternal'] = bi_read_df['ub_alt']
	elif genotype == '1|0':		
		bi_read_df['ub_maternal'] = bi_read_df['ub_alt']
		bi_read_df['ub_paternal'] = bi_read_df['ub_ref']
	else:
		bi_read_df['ub_maternal'] = np.nan
		bi_read_df['ub_paternal'] = np.nan
	return bi_read_df[cols+['ub_maternal','ub_paternal']]	

def main():

	pilefile = sys.argv[1]
	snpfile  = sys.argv[2]
	outfile  = sys.argv[3]

	pile_df = pd.read_csv(pilefile, sep='\t',header=0,
		names=['chr','loc','ref','A','T','C','G'],dtype={'chr':str,'loc':int})
	pile_df.dropna(thresh=4,inplace=True)

	genotype_df = pd.read_csv(snpfile, sep='\t',usecols=[0,1,3,4,9,13], header=None,\
		names=['chr','pos','ref','alt','genotype','gene'], dtype={'chr':str,'pos':int})

	df = pd.merge(left=pile_df, right=genotype_df, how='inner',\
			left_on=['chr','loc','ref'],right_on=['chr','pos','ref'])

	ase_df = pd.DataFrame()	
	df_grp = df.groupby('gene')
	for gene, sub_df in df_grp:
		if sub_df.empty: continue
		ase_gene_df = ase_by_gene(sub_df,gene)
		if ase_df.empty: ase_df = ase_gene_df.copy()
		else: ase_df.append(ase_gene_df,ignore_index=True)
	ase_df.to_csv('%s.ase'%outfile,index=False,mode='w')

if __name__ == '__main__':
	main()


