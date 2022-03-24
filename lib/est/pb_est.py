from scipy.optimize import minimize
from scipy.stats import poisson,norm
from scipy.special import j_roots
from scipy.special import beta as beta_fun
from scipy import stats
import pandas as pd
import numpy as np
import math
import sys

class proteinkinetics(object):

	def __init__(self, mrna_vals, mrna_kpe, prot_vals):
		self.m_vals = mrna_vals
		self.p_vals = prot_vals
		self.k1,self.k0,self.v0 = mrna_kpe
		self.v1 = None
		self.d1 = None

	def solvequad(self,a,b,c):
		d = b**2-4*a*c # discriminant
		if d < 0:
			x = np.nan
			return x
		elif d == 0:
			x = (-b+math.sqrt(b**2-4*a*c))/2*a
			return x
		else:
			x1 = (-b+math.sqrt((b**2)-(4*(a*c))))/(2*a)
			x2 = (-b-math.sqrt((b**2)-(4*(a*c))))/(2*a)
			
			if x1 > 0:return x1
			elif x2 > 0:return x2

	def MomentInference(self):
		m = np.mean(self.m_vals)
		n = np.mean(self.p_vals)
		var_n = np.var(self.p_vals)
		ita_sq = var_n / (n*n)
	
		b = self.k0 + self.k1 + m*self.k1/self.k0 - m*(ita_sq-1/n)
		c = m*(1/n - ita_sq) *(self.k0 + self.k1)
		gamma_inverse = self.solvequad(1,b,c)
		self.v1 = (self.k0 + self.k1)*n*gamma_inverse/(self.v0*self.k0)
		self.d1 = gamma_inverse

	def get_estimate(self):
		return [ self.v1, self.d1 ]
	
class mRNAkinetics(object):

	def __init__(self, vals):
		self.vals = vals	
		self.estimate = None
	
	def fun(self,at, m):
		if np.max(m) < 1e6: return(poisson.pmf(at,m))
		else:return(norm.pdf(at,loc=m,scale=np.sqrt(m)))
	
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

class mRNAkineticsPoisson(object):

	def __init__(self, vals):
		self.vals = vals	
		self.estimate = None
	
	def Poisson(self,value,ms):
		prob = stats.poisson.pmf(value, ms)
		return(prob)
	
	def LogLikelihood(self,x,value):
		ksyn = x
		return(-np.sum(np.log(self.Poisson(value,ksyn) + 1e-10)))
	
	def MomentInference(self,value):
		m1 = float(np.mean(value))
		m2 = float(np.var(value))	
		return m2

	def MaximumLikelihood(self, method = 'L-BFGS-B'):

		if len(self.vals) == 0: 
			self.estimate = [np.nan]
			return 0
		vals_ = np.copy(self.vals) # Otherwise the structure is violated.
		init = self.MomentInference(vals_)
		try:
			ll = minimize(self.LogLikelihood,init,args = (vals_),method=method)
			self.estimate = list(ll.x)
		except:
			self.estimate = [np.nan]

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
