from scipy.optimize import minimize
from scipy.special import j_roots,gamma,hyp1f1
from scipy.special import beta as beta_fun
from scipy import stats
from joblib import delayed,Parallel
from os import cpu_count
import pandas as pd
import numpy as np
import math
import sys

class MCMC(object):

	def __init__(self, vals, estimation):
		self.vals = vals	
		self.estimate = estimation
	
	def fun(self,at, m):
		if (max(m) < 1e6): return(stats.poisson.pmf(at,m))
		else: return(norm.pdf(at,loc=m,scale=sqrt(m)))

	def dBP(self,at, alpha, bet, lam):
		at.shape = (len(at), 1)
		np.repeat(at, 50, axis = 1)
		x,w = j_roots(50,alpha = bet - 1, beta = alpha - 1)
		gs = np.sum(w*self.fun(at, m = lam*(1+x)/2), axis=1)
		prob = 1/beta_fun(alpha, bet)*2**(-alpha-bet+1)*gs
		return(prob)

	def LogLikelihood(self,x,value):
		kon,koff,ksyn = x
		return(-np.sum(np.log( self.dBP(value,kon,koff,ksyn) + 1e-10)))
	
	def get_estimate(self):
		return self.estimate

	def prior(self, params):
		kon, koff, ksyn = params
		return 1.0/(kon * koff * ksyn)
	
	def acceptance(self, likelihood, likelihood_new):
		if likelihood_new>likelihood:
			return True
		else:
			accept=np.random.uniform(0,1)
			return (accept < (np.exp(likelihood_new-likelihood)))	

	def metropolis_hastings(self,iterations=10000):

		transition_model = lambda x: np.random.normal(x,0.1*x) 
		data = np.copy(self.vals)
		kon_init, koff_init, ksyn_init = self.estimate

		kon  = kon_init
		koff = koff_init
		ksyn = ksyn_init
		kon_sampled  = np.zeros(iterations)
		koff_sampled = np.zeros(iterations)
		ksyn_sampled = np.zeros(iterations)
		for i in range(iterations):
			# update kon
			kon_new     =  transition_model(kon) 
			kon_lik     = -self.LogLikelihood([kon, koff_init, ksyn_init], data)
			kon_new_lik = -self.LogLikelihood([kon_new, koff_init, ksyn_init], data)	

			LogPosterior = kon_lik	+ np.log(self.prior([kon, koff_init, ksyn_init]))
			LogPosterior_new = kon_new_lik + np.log(self.prior([kon_new, koff_init, ksyn_init]))

			if self.acceptance(LogPosterior, LogPosterior_new):
				kon = kon_new
				kon_sampled[i]=kon_new

			# update ksyn
			koff_new     =  transition_model(koff)    
			koff_lik     = -self.LogLikelihood([kon_init, koff, ksyn_init], data)
			koff_new_lik = -self.LogLikelihood([kon_init, koff_new, ksyn_init], data)	

			LogPosterior = koff_lik	+ np.log(self.prior([kon_init, koff, ksyn_init]))
			LogPosterior_new = koff_new_lik + np.log(self.prior([kon_init, koff_new, ksyn_init]))

			if self.acceptance(LogPosterior, LogPosterior_new):
				koff = koff_new
				koff_sampled[i]=koff_new

			# update ksyn
			ksyn_new     =  transition_model(ksyn)    
			ksyn_lik     = -self.LogLikelihood([kon_init, koff_init, ksyn], data)
			ksyn_new_lik = -self.LogLikelihood([kon_init, koff_init, ksyn_new], data)	

			LogPosterior = ksyn_lik	+ np.log(self.prior([kon_init, koff_init, ksyn]))
			LogPosterior_new = ksyn_new_lik + np.log(self.prior([kon_init, koff_init, ksyn_new]))

			if self.acceptance(LogPosterior, LogPosterior_new):
				ksyn = ksyn_new
				ksyn_sampled[i]=ksyn_new

		num_burnin = int(iterations/2)
		kon_burnin  = kon_sampled[num_burnin:]
		koff_burnin = koff_sampled[num_burnin:]
		ksyn_burnin = ksyn_sampled[num_burnin:]

		kon_sampled_nozero  = kon_burnin[kon_burnin!=0]
		koff_sampled_nozero = koff_burnin[koff_burnin!=0]
		ksyn_sampled_nozero = ksyn_burnin[ksyn_burnin!=0]

		if len(kon_sampled_nozero) < 1: 
			kon_median = np.nan
			kon_ci_low = np.nan
			kon_ci_up  = np.nan
		else: 
			kon_w = np.ones(len(kon_sampled_nozero))/len(kon_sampled_nozero)
			kon_rv  = stats.rv_discrete(values=( kon_sampled_nozero, kon_w ))
			kon_median = kon_rv.median()
			kon_ci_low, kon_ci_up = kon_rv.interval(0.95)	

		if len(koff_sampled_nozero) < 1: 
			koff_median = np.nan
			koff_ci_low = np.nan
			koff_ci_up  = np.nan
		else: 
			koff_w = np.ones(len(koff_sampled_nozero))/len(koff_sampled_nozero)
			koff_rv  = stats.rv_discrete(values=( koff_sampled_nozero, koff_w ))
			koff_median = koff_rv.median()
			koff_ci_low, koff_ci_up = koff_rv.interval(0.95)	

		if len(ksyn_sampled_nozero) < 1: 
			ksyn_median = np.nan
			ksyn_ci_low = np.nan
			ksyn_ci_up  = np.nan
		else: 
			ksyn_w = np.ones(len(ksyn_sampled_nozero))/len(ksyn_sampled_nozero)
			ksyn_rv  = stats.rv_discrete(values=( ksyn_sampled_nozero, ksyn_w ))
			ksyn_median = ksyn_rv.median()
			ksyn_ci_low, ksyn_ci_up = ksyn_rv.interval(0.95)	

		return  kon_median,koff_median,ksyn_median,\
			kon_ci_low, kon_ci_up,koff_ci_low, koff_ci_up,ksyn_ci_low, ksyn_ci_up

def fun(org_data,org_kp):
	obj1  = MCMC(org_data, org_kp)
	kon,koff,ksyn,kon_ci_low,kon_ci_up,koff_ci_low,koff_ci_up,ksyn_ci_low,ksyn_ci_up = obj1.metropolis_hastings()
	record=[kon,koff,ksyn,kon_ci_low,kon_ci_up,koff_ci_low,koff_ci_up,ksyn_ci_low,ksyn_ci_up]
	return record

def main():

	# number of cell sampled
	kpe_profile   = sys.argv[1]
	ase_reform_in = sys.argv[2]
	outfile       = sys.argv[3]

	# read kpe profile
	kpe = pd.read_csv(kpe_profile, header=0, index_col=0)
	kpe.dropna(inplace=True)

	ase_reform = pd.read_csv(ase_reform_in, header=0, index_col=0)

	# simulate the expression with pb
	cols = ['kon','koff','ksyn']
	data = Parallel(n_jobs=cpu_count())(\
		delayed(fun)(ase_reform.loc[idx].values, row[cols].values) for idx,row in kpe.iterrows())

	add_cols = ['kon_mean','koff_mean','ksyn_mean',\
			'kon_low','kon_upper',\
			'koff_low','koff_upper',\
			'ksyn_low','ksyn_upper']

	kpe[add_cols] = data
	kpe.to_csv(outfile,float_format="%.5f")	

if __name__ == "__main__":
	main()

