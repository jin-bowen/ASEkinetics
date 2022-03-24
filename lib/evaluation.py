from joblib import delayed,Parallel
from est import full_est,pb_est,gibbs_est
from scipy.stats import poisson,beta,power_divergence,chi2,ks_2samp
import dask.dataframe as dd
import pandas as pd
import numpy as np
import scipy as sp
import sys

def simLikelihoodRatioTest(kp1, reads1, kp2, reads2):

	obj1  = pb_est.mRNAkinetics(reads1)
	obj2  = pb_est.mRNAkinetics(reads2)

	ll1_1 = obj1.LogLikelihood(kp1, reads1)
	ll2_2 = obj2.LogLikelihood(kp2, reads2)

	ll1_2 = obj1.LogLikelihood(kp2, reads1)
	ll2_1 = obj2.LogLikelihood(kp1, reads2)

	lr1 = 2 * (ll2_1 + ll1_1 - ll2_2 - ll1_1)
	lr1 = abs(lr1)
	pval1 = chi2.sf(lr1, 3)

	lr2 = 2 * (ll1_2 + ll2_2 - ll2_2 - ll1_1)
	lr2 = abs(lr2)
	pval2 = chi2.sf(lr2, 3)

	return min(pval1,pval2)


def LikelihoodRatioTest(kp1, reads1, kp2, reads2):

	obj1  = pb_est.mRNAkinetics(reads1)
	obj2  = pb_est.mRNAkinetics(reads2)

	ll1_1 = obj1.LogLikelihood(kp1, reads1)
	ll2_2 = obj2.LogLikelihood(kp2, reads2)

	reads = np.vstack([reads1, reads2])
	obj  = pb_est.mRNAkinetics(reads)
	obj.MaximumLikelihood()
	kpe = obj.get_estimate()
	ll12 = obj.LogLikelihood(kpe, reads)

	lr = 2 * (ll12 - ll1_1 - ll2_2)
	lr = abs(lr)
	pval = chi2.sf(lr, 3)

	return pval

def est_row(read,method='full'):

	if method == 'full':
		obj1  = full_est.mRNAkinetics(read)
		obj1.MaximumLikelihood()
		kpe = obj1.get_estimate()
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

	return [ kon, koff, ksyn ]

def pb_simulation(kpe_list, n):
	kon, koff, ksyn = kpe_list
	p = beta.rvs(kon, koff, size=int(n))
	mu = ksyn * p
	vals = np.array([ poisson.rvs(imu) for imu in mu])
	return list(vals)

def kstest(isim_df, iorg_df):
	try:
		return ks_2samp(isim_df, iorg_df)[1]

	except Exception as e:
		return np.nan

def GoF(isim_df, iorg_df):

	max_sim = np.max(isim_df)
	max_org = np.max(iorg_df)
	bond = np.max([max_sim, max_org])

	bins = np.max([10, int(bond/2)])

	isim_hist,bin_edge = np.histogram(isim_df,bins=bins,range=(0,bond))
	iorg_hist,bin_edge = np.histogram(iorg_df,bins=bins,range=(0,bond))

	selec = np.where((isim_hist>5) & (iorg_hist>5) )

	isim_hist_nozero = isim_hist[selec].astype(np.float32)
	iorg_hist_nozero = iorg_hist[selec].astype(np.float32)

	q = (isim_hist_nozero - iorg_hist_nozero)**2 
	q /= (isim_hist_nozero**2)
	q = np.sum(q)
	dof = len(isim_hist_nozero) - 4
	dof = np.max([0,dof])

	pval = sp.stats.chi2.sf(q, df=dof) 
	return q,pval

def main():
	
	# number of cell sampled
	kpe_profile   = sys.argv[1]
	method        = sys.argv[2]
	ase_reform_in = sys.argv[3]
	outfile       = sys.argv[4]

	# read kpe profile
	kpe = pd.read_csv(kpe_profile, header=0, index_col=0)
	kpe.dropna(inplace=True)
	n = kpe['n'].unique()

	ase_reform = pd.read_csv(ase_reform_in, header=None, index_col=0)

	# simulate the expression with pb
	cols = ['kon','koff','ksyn']
	simulated_cols = [ 'sim_' + x for x in cols ]
	simulated_data = Parallel(n_jobs=8)(delayed(pb_simulation)(row[cols],n) for i,row in kpe.iterrows())

	# re-estimate the kp from the simulated profile
	res_data = Parallel(n_jobs=8)(delayed(est_row)(row,method) for row in simulated_data)
	kpe[simulated_cols] = res_data

	# likelihood ratiotest	
	for i, (idx, row) in enumerate(kpe.iterrows()):
		sim_data = np.array(simulated_data[i])
		org_data = ase_reform.loc[idx].astype(np.float32).values

		sim_kp = row[simulated_cols].values
		org_kp = row[cols].values

		if np.sum(sim_data) == 0:
			kpe.loc[idx,'chisq_pval'] = np.nan
			kpe.loc[idx,'simlr_pval'] = np.nan
			kpe.loc[idx,'ks_pval'] = np.nan
			continue

		chisq, p = GoF(sim_data, org_data)
		ks_p = kstest(sim_data, org_data)
		simlr = simLikelihoodRatioTest(sim_kp, sim_data, org_kp, org_data)		

		kpe.loc[idx,'chisq_pval'] = p
		kpe.loc[idx,'simlr_pval'] = simlr
		kpe.loc[idx,'ks_pval']    = ks_p

	kpe.to_csv(outfile)

if __name__ == "__main__":
	main()



