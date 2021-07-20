from scipy.optimize import minimize
from scipy.stats import poisson
from scipy.special import gamma,hyp1f1
from scipy import stats
import pandas as pd
import numpy as np
import math
import sys

class mRNAkinetics(object):

	def __init__(self, vals):
		self.vals = vals	
		self.estimate = None
	
	def pdf(self, param, m):
		kon,koff,ksyn = param
		P_m = poisson.pmf(m, ksyn) 
		#P_m *= np.power(kon/(kon+koff),m)
		P_m *= gamma(kon+m) / gamma(kon)
		P_m *= gamma(kon+koff) / gamma(kon+koff+m)
		#print(kon,koff,np.power(kon/(kon+koff),m),hyp1f1(koff, kon+koff+m, ksyn))
		P_m *= hyp1f1(koff, kon+koff+m, ksyn)
		return P_m
	
	def LogLikelihood(self,param,value):
		P_M = list(map(lambda m: self.pdf(param, m), value))
		return(-np.sum(np.log( np.array(P_M) + 1e-10)))
	
	def MomentInference(self,value):
		m1 = float(np.mean(value))
		m2 = float(sum(value*(value - 1))/len(value))
		m3 = float(sum(value*(value - 1)*(value - 2))/len(value))
	
		# sanity check on input (e.g. need at least on expression level)
		if sum(value) == 0: return np.nan
		if m1 == 0: return np.nan
		if m2 == 0: return np.nan
	
		r1=m1
		r2=m2/m1
		r3=m3/m2
	
		if (r1*r2-2*r1*r3 + r2*r3) == 0: return np.nan
		if ((r1*r2 - 2*r1*r3 + r2*r3)*(r1-2*r2+r3)) == 0: return np.nan
		if (r1 - 2*r2 + r3) == 0: return np.nan
	
		lambda_est = (2*r1*(r3-r2))/(r1*r2-2*r1*r3 + r2*r3)
		mu_est = (2*(r3-r2)*(r1-r3)*(r2-r1))/((r1*r2 - 2*r1*r3 + r2*r3)*(r1-2*r2+r3))
		v_est = (2*r1*r3 - r1*r2 - r2*r3)/(r1 - 2*r2 + r3)
	
		return np.array([lambda_est, mu_est, v_est])


	def MaximumLikelihood(self, method = 'L-BFGS-B'):

		if len(self.vals) == 0:
		        self.estimate = [np.nan, np.nan, np.nan]
		        return 0
		vals_ = np.copy(self.vals) # Otherwise the structure is violated.
		init = self.MomentInference(vals_)
		if np.isnan(init).any() or any(init < 0):init = np.array([10,10,10])

		bnds = ((1e-3,1e3),(1e-3,1e3), (1, 1e4))
		try:
		        ll = minimize(self.LogLikelihood,init,args = (vals_),method=method,bounds=bnds)
		        self.estimate = list(ll.x)
		except:
		        self.estimate = [np.nan,np.nan,np.nan]

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
	
	def metropolis_hastings(self,iterations=1000):

		transition_model = lambda x: np.random.normal(x,1)
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

		kon_sampled_nozero  = kon_sampled[kon_sampled!=0]
		koff_sampled_nozero = koff_sampled[koff_sampled!=0]
		ksyn_sampled_noxero = ksyn_sampled[ksyn_sampled!=0]

		kon_rv  = stats.rv_discrete(values=( kon_sampled_nozero,  np.ones(len(kon_sampled))/len(kon_sampled) ))
		koff_rv = stats.rv_discrete(values=( koff_sampled_nozero, np.ones(len(koff_sampled))/len(koff_sampled) ))
		ksyn_rv = stats.rv_discrete(values=( ksyn_sampled_nozero, np.ones(len(ksyn_sampled))/len(ksyn_sampled) )) 

		#self.visualize(kon_sampled)

		return   kon_rv.median(),    koff_rv.median(),    ksyn_rv.median(),\
			 kon_rv.interval(0.9),koff_rv.interval(0.9),ksyn_rv.interval(0.9) 

	def visualize(self, rvs):	
		fig, ax = plt.subplots()
		ax.hist(rvs)
		plt.show()

def main():
	mtx = sys.argv[1]
	read = np.loadtxt(mtx)
	obj1  = mRNAkinetics(read)
	obj1.MaximumLikelihood()
	kpe = obj1.get_estimate()
	#kon, koff, ksyn, kon_error, koff_error, ksyn_error = obj1.metropolis_hastings()
	#record = [ kon, koff, ksyn, kon_error, koff_error, ksyn_error ]
	#kpe_df.loc[i] = record
	kon, koff, ksyn = kpe
	record = [ kon, koff, ksyn ]	
	print(record)

if __name__ == "__main__":
	main()

