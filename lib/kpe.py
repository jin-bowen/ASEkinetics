from joblib import delayed,Parallel
from scipy import sparse
from lib.est import full_est, pb_est, gibbs_est
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

	prefix     = sys.argv[1]
	method     = sys.argv[2]

	cb_label_flag = False
	if len(sys.argv) > 4:
		cb_label_in   = sys.argv[4]
		cb_label_flag = True

	allele_list_in = prefix + '.allelei'
	allele_list = np.loadtxt(allele_list_in, dtype=str)
	
	cb_list_in = prefix + '.cbi'
	cb_list = np.loadtxt(cb_list_in, dtype=str)
	
	ase_sparse_mat_in = prefix + '.ase.npz'
	ase_sparse_mat = sparse.load_npz(ase_sparse_mat_in)

	cb_idx = pd.DataFrame(cb_list, columns=['cb'])
	cb_idx.reset_index(inplace=True)
	allele_idx = pd.DataFrame(allele_list, columns=['allele']) 
	allele_idx.reset_index(inplace=True)

	if cb_label_flag:
		cb_label = pd.read_csv(cb_label_in,header=0,names=['cb','label'])
		cb_idx_label = pd.merge(cb_idx, cb_label, how='left', on='cb')			

		cb_idx_grp = cb_idx_label.groupby('label')
		for label, igrp in cb_idx_grp:

			cb_idx_sub = igrp['index'].values
			ase_mat_sub = ase_sparse_mat.toarray()[:,cb_idx_sub]
			
			org_allele = allele_idx['allele'].tolist() 
			out_index  = [ item + '-' + label for item in org_allele ]

			data = Parallel(n_jobs=8)(delayed(est_vec)(vec,method) for i,vec in enumerate(ase_mat_sub))
			out_kpe_file = prefix + '_' + label + '_' + method + '.est'
			cols = ['kon','koff','ksyn','n','mean','var']
			res_df = pd.DataFrame(data,columns=cols,index=out_index)
			res_df.to_csv(out_kpe_file,float_format="%.5f")

			ase_out = pd.DataFrame(ase_mat_sub, index=out_index)
			out_ase_file = prefix + '_'  + label + '.ase.reform'
			ase_out.to_csv(out_ase_file)

	else:
		ase_mat = ase_sparse_mat.toarray()	
		out_index = allele_idx['allele'].tolist() 

		data = Parallel(n_jobs=8)(delayed(est_vec)(vec,method) for i,vec in enumerate(ase_mat))
		out_kpe_file = prefix + '_' + method + '.est'
		cols = ['kon','koff','ksyn','n','mean','var']
		res_df = pd.DataFrame(data,columns=cols,index=out_index)
		res_df.to_csv(out_kpe_file,float_format="%.5f")

		ase_out = pd.DataFrame(ase_mat, index=out_index)
		out_ase_file = prefix + '.ase.reform'
		ase_out.to_csv(out_ase_file)

if __name__ == "__main__":
	main()



