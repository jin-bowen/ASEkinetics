from itertools import compress
import pandas as pd
import numpy as np
import sys 
import re

def MergeRecord(df_subset,gene,outfile,phased=True):

	cols = ['cb','gene','pos','allele_ref','ub_ref','allele_alt','ub_alt']
	if phased: cols += ['ub_maternal','ub_paternal']
	read_df = pd.DataFrame(columns=cols)

	for irow, row in df_subset.iterrows():
		read_df_temp = PileToRecord(row, phased)
		read_df = read_df.append(read_df_temp, ignore_index=True)
	read_df_mapped = read_df.loc[read_df['gene']==gene]

	read_df_all = read_df_mapped.groupby(['cb','gene']).agg(
		ub_m_list=pd.NamedAgg(column='ub_maternal',aggfunc=list),
		ub_p_list=pd.NamedAgg(column='ub_paternal',aggfunc=list),
		ub_maternal_count=pd.NamedAgg(column='ub_maternal',aggfunc='nunique'),
		ub_paternal_count=pd.NamedAgg(column='ub_paternal',aggfunc='nunique'))
	read_df_all.reset_index(inplace=True)

	aberrant_read_list = []	
	for irow, row in read_df_all.iterrows():
		ub_m_list=row['ub_m_list']
		ub_p_list=row['ub_p_list']
		redundant = list(set(ub_m_list).intersection(ub_p_list))
		if np.nan in redundant: redundant.remove(np.nan)
	
		for ub in redundant:
			m_ub_count = ub_m_list.count(ub)
			p_ub_count = ub_p_list.count(ub)
			if m_ub_count!=0 and p_ub_count!=0: aberrant_read_list.append(irow) 

			if m_ub_count > p_ub_count: read_df_all.loc[irow,'ub_paternal_count'] -= 1
			elif m_ub_count < p_ub_count: read_df_all.loc[irow,'ub_paternal_count'] -= 1
			else: 
				read_df_all.loc[irow,'ub_maternal_count'] -= 1
				read_df_all.loc[irow,'ub_paternal_count'] -= 1

	if len(aberrant_read_list)>0:
		read_df_all.loc[aberrant_read_list].to_csv('%s_aberrant_read.csv'%outfile,mode='a')

	return read_df_all

def PileToRecord(row, phased):
	cols = ['cb','gene','pos','allele_ref','ub_ref','allele_alt','ub_alt']

	if phased: cols += ['ub_maternal','ub_paternal']
	read_df = pd.DataFrame(columns=cols)

	chr = row['chr']
	loc = row['loc']
	ref = row['ref']
	alt = row['alt']

	ref_read = row[ref]
	alt_read = row[alt]
	if (ref_read is np.nan) and (alt_read is np.nan): return read_df
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

	if alt_read is np.nan:
		read_df[['cb','gene','pos','allele_ref','ub_ref']] = ref_read_df[['cb','gene','pos','allele_ref','ub_ref']]
		read_df[['allele_alt','ub_alt']] = [alt,np.nan]
	elif ref_read is np.nan:
		read_df[['cb','gene','pos','allele_alt','ub_alt']] = alt_read_df[['cb','gene','pos','allele_alt','ub_alt']]
		read_df[['allele_ref','ub_ref']] = [ref,np.nan]
	else:
		read_df = pd.merge(left=ref_read_df, right=alt_read_df,how='outer',on=['cb','gene','pos'])
		read_df['allele_alt'].fillna(value=alt, inplace=True)
		read_df['allele_ref'].fillna(value=ref, inplace=True)

	if phased:
		genotype =  row['genotype']
		if genotype == '1|0': 
			read_df['ub_maternal'] = read_df['ub_alt']
			read_df['ub_paternal'] = read_df['ub_ref']
		elif genotype == '0|1':
			read_df['ub_maternal'] = read_df['ub_ref']
			read_df['ub_paternal'] = read_df['ub_alt']
		else: 
			read_df['ub_maternal'] = None
			read_df['ub_paternal'] = None
	return read_df		

def main():

	pilefile = sys.argv[1]
	snpfile  = sys.argv[2]
	outfile  = sys.argv[3]

	# header: chr;loc;ref;A;T;C;G
	pdf = pd.read_csv(pilefile, sep="\t",header=0,
		names=['chr','loc','ref','A','T','C','G'],dtype={'chr':str,'loc':int})
	pdf['ref'] =  pdf['ref'].astype('category')
	pdf.dropna(thresh=4,inplace=True)

	sdf = pd.read_csv(snpfile, sep="\t",usecols=list(range(6)) + [9,13], comment='#',header=None,
		names=['chr','loc','id','ref','alt','qual','genotype','gene'], dtype={'chr':str,'loc':int})
	sdf['ref'] =  sdf['ref'].astype('category')
	sdf['alt'] =  sdf['alt'].astype('category')
	
	df = pd.merge(left=pdf, right=sdf, how='inner',on=['chr','loc','ref'])
	ase_df = pd.DataFrame()	

	df_grp = df.groupby('gene')
	for gene, df_subset in df_grp:
		temp = MergeRecord(df_subset,gene,outfile,phased=True)
	
	ase_df_sum.reset_index(inplace=True)
	ase_df_sum.to_csv('%s.ase'%outfile,index=False,mode='w')

if __name__ == "__main__":
	main()

