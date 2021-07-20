from joblib import delayed,Parallel
from est import full_MCMC_est
from est import pb_est
from est import gibbs_est
import pandas as pd
import dask.dataframe as dd
import numpy as np
import sys

def est_row(row,method='full'):
	gene   = row['gene']
	allele = row['allele']
	read   = row['list']

	n      = len(read)
	mean   = np.mean(read)
	var    = np.var(read)

	if len(read) < 50:
		return [gene,allele] + [None, None, None] + [n, mean, var]

	if method == 'full':
		obj1  = full_MCMC_est.mRNAkinetics(read)
		obj1.MaximumLikelihood()
		kpe = obj1.get_estimate()
		#kon, koff, ksyn, kon_error, koff_error, ksyn_error = obj1.metropolis_hastings()
		#record = [ kon, koff, ksyn, kon_error, koff_error, ksyn_error ]
		#kpe_df.loc[i] = record
		kon, koff, ksyn = kpe

	elif method == 'pb':
		obj1  = pb_est.mRNAkinetics(read)
		obj1.MaximumLikelihood()
		kpe = obj1.get_estimate()
		kon, koff, ksyn = kpe

	elif method == 'gibbs':
		params1, bioParams1 = gibbs_est.getParamsBayesian(read)
		kon  = params1.alpha.mean()
		koff = params1.beta.mean()
		ksyn = params1.gamma.mean()

	return [ gene, allele, kon, koff, ksyn, n, mean, var ]

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

def main():

	ase_infer  = sys.argv[1]
	method     = sys.argv[2]

	cb_w_cycle = sys.argv[3]
	outfile    = sys.argv[4]

	inferred_allele_df = dd.read_csv(ase_infer,header=0)
	cb_cycle_df = pd.read_csv(cb_w_cycle,index_col=0,header=0)
	df = dd.merge(inferred_allele_df,cb_cycle_df,left_on='cb',right_index=True)
	df_grp = df.groupby('cellPhase')
	
	cols = ['gene','allele','kon','koff','ksyn','n','mean','var','cycle']
	outf = open(outfile, "a+")
	outf.write(','.join(cols) + '\n')

	for igrp, idf_grp in df_grp:		
		processed_df = process(idf_grp.compute())
		res_kpe = Parallel(n_jobs=8)(delayed(est_row)(row,method) for i,row in processed_df.iterrows())
		res_kpe_df = pd.DataFrame(res_kpe,columns=cols[:-1])
		res_kpe_df['cycle'] = igrp	
		res_kpe_df[cols].to_csv(outf,index=False,header=False,float_format="%.5f")

if __name__ == "__main__":
	main()

