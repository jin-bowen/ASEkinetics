import matplotlib.pyplot as plt
import pandas as pd
import numpy.linalg as lin
from scipy.optimize import minimize
from scipy.special import gammaln
from scipy.stats import betabinom
import numpy as np
import sys

def conf_interval(theta, I_theta, alpha=0.05):

	lower_bound = theta - 1.96 * np.sqrt(I_theta)
	upper_bound = theta + 1.96 * np.sqrt(I_theta) 

	return lower_bound, upper_bound

def loglike_betabinom(params, *args):

	a, b = params[0], params[1]
	Nm = args[0]
	Nall  = args[1]

	log_pmf = gammaln(Nall+1) + gammaln(Nm+a) + gammaln(Nall-Nm+b) + gammaln(a+b) - \
	(gammaln(Nm+1) + gammaln(Nall-Nm+1) + gammaln(a) + gammaln(b) + gammaln(Nall+a+b))

	return -np.sum(log_pmf)

def fit_param_betabinom(ase):
	init_params = np.array([0.5, 0.5])
	exp_values  = ase[['H1_infer','umi']].values
	bnds = ((0.001,100),(0.001,100))
	res = minimize(loglike_betabinom,x0=init_params,bounds=bnds, \
		args=(ase['H1_infer'],ase['umi']),method='L-BFGS-B')	
	a_hat = res.x[0]
	b_hat = res.x[1]

	Ia = res.hess_inv.todense()[0,0]
	Ib = res.hess_inv.todense()[1,1]
	Ia_lb, Ia_ub = conf_interval(a_hat, Ia)
	Ib_lb, Ib_ub = conf_interval(b_hat, Ib)

	return a_hat, Ia_ub, Ia_lb, b_hat, Ib_ub, Ib_lb

def allele_infer(umi, ase, outfile, min_reads=1):

	ase_all = pd.merge(umi,ase,how='left',on=['cb','gene'])
	ase_intersect = ase_all.loc[~ase_all.isna().any(axis=1)]
	org_columns = ase_all.columns.tolist() 

	cols = ['cb','gene','H1_infer','H2_infer','umi']
	ase_intersect['H1_infer'] = ase_intersect['ub_H1'] * ase_intersect['umi']
	ase_intersect['H1_infer'] /= (ase_intersect['ub_H1'] + ase_intersect['ub_H2'])
	ase_intersect['H1_infer'] = np.ceil(ase_intersect['H1_infer'])
	ase_intersect['H2_infer'] = ase_intersect['umi'] - ase_intersect['H1_infer']	
	ase_intersect['ub_mp'] = ase_intersect.apply(lambda x: x.ub_H1 + x.ub_H2, axis=1)	
	ase_intersect['ub_mp_bool'] = ase_intersect.apply(lambda x: 1 if x.ub_mp >= x.umi else 0, axis=1)	
	
	select_bool = (ase_intersect['ub_H1']>=min_reads) | (ase_intersect['ub_H2']>=min_reads)
	
	ase_intersect_train = ase_intersect.loc[select_bool]
	ase_for_infer = ase_all.loc[ase_all.isna().any(axis=1)]
	ase_for_infer = ase_for_infer.append(ase_intersect.loc[~select_bool],ignore_index=True)

	ase_intersect_train.to_csv(outfile+'.org',index=False)

	ase_intersect_grp = ase_intersect_train.groupby('gene')
	ase_for_infer_grp = ase_for_infer.groupby('gene')

	record_list = []

	f = open(outfile+'.infer', 'w+')
	f.write(','.join(cols))
	f.write('\n')
	for gene, sub_intersect in ase_intersect_grp:
#		a_hat, Ia_ub, Ia_lb, b_hat, Ib_ub, Ib_lb = fit_param_betabinom(sub_intersect)
#		sub_for_infer = ase_for_infer_grp.get_group(gene)
		try: 
			a_hat, Ia_ub, Ia_lb, b_hat, Ib_ub, Ib_lb = fit_param_betabinom(sub_intersect)
			sub_for_infer = ase_for_infer_grp.get_group(gene)
		except: continue
		sub_for_infer['H1_infer'] = sub_for_infer['umi'].astype(int).apply(lambda x: betabinom.rvs(x,a_hat,b_hat))
		sub_for_infer['H2_infer'] = sub_for_infer['umi'] - sub_for_infer['H1_infer']

		ase_intersect_w_infer = pd.concat([sub_for_infer[cols], sub_intersect[cols]])
		ase_intersect_w_infer[cols].to_csv(f,header=False,index=False,mode='a')

		record = [gene, len(sub_intersect), len(ase_intersect_w_infer), len(sub_intersect)/len(ase_intersect_w_infer)]
		record += [a_hat, Ia_lb, Ia_ub, b_hat, Ib_lb, Ib_ub ]
		record_list.append(record)

	df_record_col = ['gene','nref','nall','percent','a_hat','Ia_lb','Ia_ub','b_hat','Ib_lb','Ib_ub']
	df_record = pd.DataFrame(data=record_list, columns=df_record_col)
	df_record.to_csv(outfile+'.record',index=False)
	f.close()

def ase_compare(ase_org, ase_infer):

	fig,ax = plt.subplots(ncols=2, nrows=2,sharex='all', sharey='row')
	ax[0,0].hist(ase_org['H1_infer'],density=True)
	ax[0,0].set_title('H1')
	ax[1,0].hist(ase_infer['H1_infer'],density=True,bins=20)
	ax[1,0].set_xlabel('umi')

	ax[0,1].hist(ase_org['H2_infer'],density=True)
	ax[0,1].set_title('H2')
	ax[1,1].hist(ase_infer['H2_infer'],density=True,bins=20)
	ax[1,1].set_xlabel('umi')
	plt.show()

def main():

	umi_file = sys.argv[1]
	ase_file = sys.argv[2]
	cb_file  = sys.argv[3]
	prefix   = sys.argv[4]

	if umi_file.endswith('gz'):
		umi_raw = dd.read_csv(umi_file,header=0,dtype={'cb':str,'gene':str,'umi':float}, compression='gzip')
	else: umi_raw = dd.read_csv(umi_file,header=0,dtype={'cb':str,'gene':str,'umi':float} )

	if ase_file.endswith('gz'):
		ase_raw = dd.read_csv(ase_file, header=0, compression='gzip')
	else: ase_raw = dd.read_csv(ase_file, header=0) 

	# filtered cells 
	import dask.dataframe as dd
	cb_raw = pd.read_csv(cb_file,header=0)
	cb_keep = cb_raw.loc[cb_raw['keep']==1,'cb']
	umi = dd.merge(umi_raw, cb_keep, on='cb',suffixes=('','_cb')).compute()
	ase = dd.merge(ase_raw, cb_keep, on='cb',suffixes=('','_cb')).compute()

	outfile = '%s.ase.class'%(prefix)
	ase_infer_outfile = '%s.ase'%(prefix)

	allele_infer(umi,ase,ase_infer_outfile)

if __name__ == "__main__":
	main()


