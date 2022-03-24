from joblib import delayed,Parallel
from scipy import sparse
from est import full_est, pb_est, gibbs_est
from ase_reform import process_phased
import pandas as pd
import numpy as np
import sys

def est_vec(reads,method='full'):

	n      = len(reads)
	mean   = np.mean(reads)
	var    = np.var(reads)
	
	if mean == 0:
		return [None, None, None] + [n, mean, var]

	if method == 'full':
		obj1  = full_est.mRNAkinetics(reads)
		obj1.MaximumLikelihood()
		kpe = obj1.get_estimate()
		kon, koff, ksyn = kpe

	elif method == 'pb':
		obj1  = pb_est.mRNAkinetics(reads)
		obj1.MaximumLikelihood()
		kpe = obj1.get_estimate()
		kon, koff, ksyn = kpe

	elif method == 'gibbs':
		params1, bioParams1 = gibbs_est.getParamsBayesian(reads)
		kon  = params1.alpha.mean()
		koff = params1.beta.mean()
		ksyn = params1.gamma.mean()
	
	else: 
		print('no matched method')
		return 0
	return [ kon, koff, ksyn, n, mean, var ]

def main():

	ase_infer = sys.argv[1]
	method    = sys.argv[2]
	prefix    = sys.argv[3]

	inferred_allele_df = pd.read_csv(ase_infer,header=0)	
	
	processed_df = process_phased(inferred_allele_df)
#	processed_df.to_csv(prefix + '.reform')
#	return 0
	ase_mat = processed_df['cb_umi_list'].values
	out_index = processed_df['gene_allele']
	data = Parallel(n_jobs=8)(delayed(est_vec)(vec,method) for i,vec in enumerate(ase_mat))
	out_kpe_file = prefix + '_' + method + '.est'
	cols = ['kon','koff','ksyn','n','mean','var']
	res_df = pd.DataFrame(data,columns=cols,index=out_index)
	res_df.to_csv(out_kpe_file,float_format="%.5f")

if __name__ == "__main__":
	main()



