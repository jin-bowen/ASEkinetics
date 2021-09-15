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

def main():
	mtx = sys.argv[1]
	read = np.loadtxt(mtx)
	
	obj1  = mRNAkinetics(read)
	obj1.MaximumLikelihood()
	kpe = obj1.get_estimate()
	kon, koff, ksyn = kpe
	record = [ kon, koff, ksyn ]	
	print(record)

	read_nozero = read[np.where(read!=0)]
	obj2 = mRNAkinetics(read_nozero)
	obj2.MaximumLikelihood()
	kpe2 = obj2.get_estimate()
	print(kpe2)

if __name__ == "__main__":
	main()

