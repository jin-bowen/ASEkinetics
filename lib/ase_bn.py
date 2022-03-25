import matplotlib.pyplot as plt
import pandas as pd
import dask.dataframe as dd
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
	exp_values  = ase[['maternal_infer','umi']].values
	bnds = ((0.001,100),(0.001,100))
	res = minimize(loglike_betabinom,x0=init_params,bounds=bnds, \
		args=(ase['maternal_infer'],ase['umi']),method='L-BFGS-B')	
	a_hat = res.x[0]
	b_hat = res.x[1]

	Ia = res.hess_inv.todense()[0,0]
	Ib = res.hess_inv.todense()[1,1]
	Ia_lb, Ia_ub = conf_interval(a_hat, Ia)
	Ib_lb, Ib_ub = conf_interval(b_hat, Ib)

	return a_hat, Ia_ub, Ia_lb, b_hat, Ib_ub, Ib_lb

def allele_infer(umi, ase, outfile):

	ase_all = pd.merge(umi,ase,how='left',on=['cb','gene'])
	ase_intersect = ase_all.loc[~ase_all.isna().any(axis=1)]
	ase_for_infer = ase_all.loc[ase_all.isna().any(axis=1)]

	cols = ['cb','gene','maternal_infer','paternal_infer','umi']
	ase_intersect['maternal_infer'] = ase_intersect['ub_maternal'] * ase_intersect['umi']
	ase_intersect['maternal_infer'] /= (ase_intersect['ub_maternal'] + ase_intersect['ub_paternal'])
	ase_intersect['maternal_infer'] = np.ceil(ase_intersect['maternal_infer'])
	ase_intersect['paternal_infer'] = ase_intersect['umi'] - ase_intersect['maternal_infer']	
	ase_intersect.to_csv(outfile+'.org',index=False)

	ase_intersect_grp = ase_intersect.groupby('gene')
	ase_for_infer_grp = ase_for_infer.groupby('gene')

	record_list = []

	f = open(outfile+'.infer', 'w+')
	f.write(','.join(cols))
	f.write('\n')
	for gene, sub_intersect in ase_intersect_grp:
		try: 
			a_hat, Ia_ub, Ia_lb, b_hat, Ib_ub, Ib_lb = fit_param_betabinom(sub_intersect)
			sub_for_infer = ase_for_infer_grp.get_group(gene)
		except: continue
		sub_for_infer['maternal_infer'] = sub_for_infer['umi'].astype(int).apply(lambda x: betabinom.rvs(x,a_hat,b_hat))
		sub_for_infer['paternal_infer'] = sub_for_infer['umi'] - sub_for_infer['maternal_infer']

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
	ax[0,0].hist(ase_org['maternal_infer'],density=True)
	ax[0,0].set_title('maternal')
	ax[1,0].hist(ase_infer['maternal_infer'],density=True,bins=20)
	ax[1,0].set_xlabel('umi')

	ax[0,1].hist(ase_org['paternal_infer'],density=True)
	ax[0,1].set_title('paternal')
	ax[1,1].hist(ase_infer['paternal_infer'],density=True,bins=20)
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
	cb_raw = pd.read_csv(cb_file,header=0)
	cb_keep = cb_raw.loc[cb_raw['keep']==1,'cb']
	umi = dd.merge(umi_raw, cb_keep, on='cb',suffixes=('','_cb')).compute()
	ase = dd.merge(ase_raw, cb_keep, on='cb',suffixes=('','_cb')).compute()

	outfile = '%s.ase.class'%(prefix)
	ase_infer_outfile = '%s.ase'%(prefix)

	min_reads=2
	ase_filter = ase.loc[(ase['ub_maternal']>=min_reads) | (ase['ub_paternal']>=min_reads)]
	allele_infer(umi,ase_filter,ase_infer_outfile)

if __name__ == "__main__":
	main()


