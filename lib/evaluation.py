from joblib import delayed,Parallel
from est import full_MCMC_est
from est import pb_est
from est import gibbs_est
from scipy.stats import poisson
from scipy.stats import beta
import pandas as pd
import numpy as np
import scipy as sp
import sys

def est_row(read,method='full'):

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

	return [ kon, koff, ksyn ]


def pb_simulation(kpe_list, n):
	kon, koff, ksyn = kpe_list
	p = beta.rvs(kon, koff, size=n)
	mu = ksyn * p
	vals = np.array([ poisson.rvs(imu) for imu in mu])
	return list(vals)
	
def main():
	
	# number of cell sampled
	kpe_profile = sys.argv[1]
	method      = sys.argv[2]
	outfile     = sys.argv[3]
	n = 400

	cols = ['kon','koff','ksyn']
	simulation_cols = [ 'sim_' + x for x in cols ]
	kpe = pd.read_csv(kpe_profile, header=0)
	kpe.dropna(inplace=True)
	simulation_data = Parallel(n_jobs=8)(delayed(pb_simulation)(row[cols],n) for i,row in kpe.iterrows())

	res_data = Parallel(n_jobs=8)(delayed(est_row)(row,method) for row in simulation_data)
	kpe[simulation_cols] = res_data
	kpe.to_csv(outfile,index=False,float_format="%.5f")

if __name__ == "__main__":
	main()



