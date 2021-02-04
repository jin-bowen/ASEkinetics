from joblib import delayed,Parallel
from estimation import *
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
		return [gene,allele] + [None, None, None, None, None]

	obj1  = mRNAkinetics(mrna)
	obj1.MaximumLikelihood()
	mrna_kpe = obj1.get_estimate()

	return [gene,allele] + mrna_kpe + [len(mrna),np.mean(mrna)]
		
def main():

	ase_infer = sys.argv[1]
	name     = sys.argv[2]
	outdir   = sys.argv[3]

	inferred_allele_df = pd.read_csv(ase_infer)

	# txb estimation
	df= process(inferred_allele_df)

	data = Parallel(n_jobs=10)(delayed(estimate_by_row)(row) for i,row in df.iterrows())
	out = pd.DataFrame(data,columns=['gene','allele','kon','koff','ksyn','n','mean'])

	outfile = '%s/%s.mle.infer'%(outdir,name)
	out.dropna().to_csv(outfile, index=False,float_format='%.5f')

if __name__ == "__main__":
	main()

