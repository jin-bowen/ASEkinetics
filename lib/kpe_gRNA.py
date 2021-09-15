from joblib import delayed,Parallel
from scipy import sparse
from est import full_est, pb_est, gibbs_est
from ase_reform import process
import dask.dataframe as dd
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

	ase_infer_in  = sys.argv[1]
	method   = sys.argv[2]
	cb_label_in   = sys.argv[3]
	gene     = sys.argv[4]
	out_dir  = sys.argv[5]

	ase_infer = dd.read_csv(ase_infer_in,header=0)
	cb_label = pd.read_csv(cb_label_in,sep='\s+|,',engine='python',header=None,names=['cb','label'])

	ase_infer_subset = dd.merge(ase_infer, cb_label, on='cb')		
	inferred_allele_df = ase_infer_subset[ase_infer_subset['gene']==gene].compute()
	sparse_mat, allele_list, cb_list = process(inferred_allele_df)
	cb_idx = pd.DataFrame(cb_list, columns=['cb'])
	cb_idx.reset_index(inplace=True)
	allele_idx = pd.DataFrame(allele_list, columns=['allele']) 
	allele_idx.reset_index(inplace=True)

	cb_idx_label = pd.merge(cb_idx, cb_label,on='cb')			
	cb_idx_grp = cb_idx_label.groupby('label')
	for label, igrp in cb_idx_grp:

		cb_idx_sub = igrp['index'].values
		ase_mat_sub = sparse_mat.toarray()[:,cb_idx_sub]

		org_allele = allele_idx['allele'].tolist() 
		out_index  = [ item + '-' + label for item in org_allele ]

		data = Parallel(n_jobs=8)(delayed(est_vec)(vec,method) for i,vec in enumerate(ase_mat_sub))
		out_kpe_file = out_dir + '/' + gene + '_' + label + '_' + method + '.est'
		cols = ['kon','koff','ksyn','n','mean','var']
		res_df = pd.DataFrame(data,columns=cols,index=out_index)
		res_df.dropna(inplace=True)
	
		if res_df.shape[0] < 1: return 0
		res_df.to_csv(out_kpe_file,float_format="%.5f")

		ase_out = pd.DataFrame(ase_mat_sub, index=out_index)
		out_ase_file = out_dir + '/' + gene + '_'  + label + '.ase.reform'
		ase_out.to_csv(out_ase_file)

if __name__ == "__main__":
	main()



