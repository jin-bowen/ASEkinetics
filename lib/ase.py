import pandas as pd
import numpy as np
from itertools import compress
import sys 
import re

def pile_2_record(row):

	chr = row['chr']
	loc = row['loc']
	ref = row['ref']
	alt = row['alt']

	ref_read = row[ref]
	alt_read = row[alt]

	cols = ['cb','gene','pos','allele_ref','ub_ref','allele_alt','ub_alt']
	df_bi = pd.DataFrame(columns=cols)

	if (ref_read is np.nan) and (alt_read is np.nan): return df_bi
	if ref_read is not np.nan :
		ref_read_mtx = np.array(re.split(',|:',ref_read)).reshape((-1,3))
		ref_read_df = pd.DataFrame(ref_read_mtx, columns=['cb','ub','gene']).drop_duplicates()
		cref = ref_read_df.groupby(['cb','gene'])['ub'].nunique().reset_index().rename({"ub":"ub_ref"},axis=1)
		cref['pos'] = str(chr) + ':'+ str(loc)
		cref['allele_ref'] = ref

	if alt_read is not np.nan :
		alt_read_mtx = np.array(re.split(',|:',alt_read)).reshape((-1,3))
		alt_read_df = pd.DataFrame(alt_read_mtx, columns=['cb','ub','gene']).drop_duplicates()
		calt = alt_read_df.groupby(['cb','gene'])['ub'].nunique().reset_index().rename({"ub":"ub_alt"},axis=1)
		calt['pos'] = str(chr) + ':' + str(loc)
		calt['allele_alt'] = alt

	if alt_read is np.nan:
		df_bi[['cb','gene','pos','allele_ref','ub_ref']] = cref[['cb','gene','pos','allele_ref','ub_ref']]
		df_bi[['allele_alt','ub_alt']] = [alt,0]
	elif ref_read is np.nan:
		df_bi[['cb','gene','pos','allele_alt','ub_alt']] = calt[['cb','gene','pos','allele_alt','ub_alt']]
		df_bi[['allele_ref','ub_ref']] = [ref,0]
	else:
		df_bi = pd.merge(left=cref, right=calt,how='outer',left_on=['cb','gene','pos'],
		        right_on=['cb','gene','pos'])
		df_bi['allele_alt'].fillna(value=alt, inplace=True)
		df_bi['allele_ref'].fillna(value=ref, inplace=True)
		df_bi['ub_alt'].fillna(value=0, inplace=True)
		df_bi['ub_ref'].fillna(value=0, inplace=True)

	return df_bi[cols]	

def main():

	pilefile = sys.argv[1]
	snpfile  = sys.argv[2]
	outfile  = sys.argv[3]

	# header: chr;loc;ref;A;T;C;G
	pdf = pd.read_csv(pilefile, sep=";",header=0,
		names=['chr','loc','ref','A','T','C','G'],dtype={'chr':str,'loc':int})
	pdf['ref'] =  pdf['ref'].astype('category')
	pdf.dropna(thresh=4,inplace=True)

	sdf = pd.read_csv(snpfile, sep="\t",usecols=[0,1,3,4], comment='#',header=0,
		names=["chr","pos","ref","alt"], dtype={'chr':str,'pos':int})
	sdf['ref'] =  sdf['ref'].astype('category')
	sdf['alt'] =  sdf['alt'].astype('category')

	df = pd.merge(left=pdf, right=sdf, how='inner', left_on=['chr','loc','ref'],
		right_on=['chr','pos','ref'])

	ase_df = pd.DataFrame()	
	for idx, irow in df.iterrows():
		temp = pile_2_record(irow)
		if ase_df.empty: ase_df = temp
		else: ase_df = ase_df.append(temp,ignore_index=True)
	ase_df.to_csv('%s.ase'%outfile,index=False,mode='w')

if __name__ == "__main__":
	main()


