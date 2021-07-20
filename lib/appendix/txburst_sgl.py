from joblib import delayed,Parallel
from full_MCMC_est import *
from scipy import stats
import pandas as pd
import dask.dataframe as dd
import numpy as np
import sys

def process(df):
	df['ref_infer'] = np.ceil(df['ub_ref'] * df['umi']/ (df['ub_ref'] + df['ub_alt']))
	df['alt_infer'] = df['umi'] - df['ref_infer']

	df['allele_pos'] = df[['gene','pos','allele_ref','allele_alt']].agg('_'.join, axis=1)
	df_grp = df.groupby('allele_pos')['ref_infer','alt_infer']

	out = pd.DataFrame(columns=['gene','allele','list'])
	for key, group in df_grp:
		key_list = key.split('_')
		igene = key_list[0]
		ref_allele_id = key_list[1] + '_' + key_list[2]
		alt_allele_id = key_list[1] + '_' + key_list[3]

		ref_val = group['ref_infer'].values
		alt_val = group['alt_infer'].values

		ref_nonzero = ref_val[np.where(ref_val!=0)]
		alt_nonzero = alt_val[np.where(alt_val!=0)]

		out.loc[len(out.index)] = [igene, ref_allele_id, ref_nonzero]
		out.loc[len(out.index)] = [igene, alt_allele_id, alt_nonzero]
	return out

def estimate_by_row(row):
	
	gene = row['gene']
	allele = row['allele']
	mrna = row['list']
	if len(mrna) < 50: 
		return [gene,allele] + [None, None, None, None, None, None, None]

	mean = np.mean(mrna)
	var  = np.var(mrna)

	obj1  = mRNAkinetics(mrna)
	obj1.MaximumLikelihood()
	mrna_kpe = obj1.get_estimate()

	return [gene,allele] + mrna_kpe + pksyn + [len(mrna),mean,var]

def main():

	ase_infer = sys.argv[1]
	cb_w_gRNA = sys.argv[2]

	gene     = sys.argv[3]
	outdir   = sys.argv[4]

	inferred_allele_df = dd.read_csv(ase_infer,header=None,\
		names=['cb','gene','pos','allele_ref','ub_ref','allele_alt','ub_alt','umi'])

	select = inferred_allele_df['gene']==gene
	inferred_allele_df_subset = inferred_allele_df.loc[select,:].compute()

	cb_gRNA_df = pd.read_csv(cb_w_gRNA, sep=' ',header=None, names=['cb_w_exp','gRNA','gRNA-grp'])
	bool_var=cb_gRNA_df['gRNA-grp']=='NTC'
	cb_gRNA_df.loc[ bool_var,'grp'] = 'NTC'
	cb_gRNA_df.loc[~bool_var,'grp'] = cb_gRNA_df.loc[~bool_var,'gRNA']
	cb_gRNA_df.drop(['gRNA-grp','gRNA'], axis=1, inplace=True)
	cb_gRNA_df.drop_duplicates(inplace=True)
	suffix = ase_infer.split('.')[-1]

	df = pd.merge(inferred_allele_df_subset,cb_gRNA_df,left_on='cb',right_on='cb_w_exp')
	df_grp = df.groupby('grp')
	
	cols=['gene','allele','kon','koff','ksyn','pksyn','n','mean','var']
	out = pd.DataFrame(columns=['gene','gRNA-grp','allele','kon','koff','ksyn','pksyn','n','mean','var'])

	# txb estimation
	for gRNA_grp, idf_grp in df_grp:

		processed_df =  process(idf_grp)
		data = Parallel(n_jobs=10)(delayed(estimate_by_row)(row) for i,row in processed_df.iterrows())
		temp=pd.DataFrame(data,columns=cols)
		temp['gRNA-grp']=gRNA_grp
		out=out.append(temp,ignore_index=False)

	if not out.dropna().empty:
		outfile = '%s/%s.mle.%s'%(outdir,gene,suffix)
		out.dropna().to_csv(outfile,index=False)

if __name__ == "__main__":
	main()

