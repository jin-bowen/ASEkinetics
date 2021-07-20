from scipy.interpolate import interp1d
from scipy.optimize import minimize
from scipy import special
from scipy import stats
from scipy.sparse import spdiags,csc_matrix
from scipy.sparse.linalg import onenormest

from expv import expv

import pandas as pd
import numpy as np
import math
import sys

def GenericFSP(ti,tf,xi,A,pi,ptimes=2):

	tvec = np.linspace(0,tf-ti,ptimes)
	tol=1e-9

	soln = np.zeros((len(pi),len(tvec)))
	# what is m
	m=30
	n = int(len(pi))

	soln[:,0] = pi
	pv = pi
	anorm = onenormest(A)
	wsp = np.zeros(7+n*(m+2)+5*(m+2)*(m+2),dtype=np.float64)
	iwsp = np.zeros(m+2,dtype=np.int32)
	for i in range(1,len(tvec)):
		pv,m,v = expv(tvec[i]-tvec[i-1],A,pv,tol = tol,m=30)
		soln[:,i] = pv 
	return soln

def getA(params,N):
	kon,koff,kr= params
	g = 1
	d0 = np.tile(np.array([0,koff]),N)
	d1 = np.tile(np.array([kon,0]),N)
	d2 = g*np.repeat(np.arange(N),2)
	d3 = np.tile([0,kr],N)
	dm = -d1-d2-d3-d0
	return spdiags([d0,d1,d2,d3,dm],[1,-1,2,-2,0],2*N,2*N)
    
def solve(params,ti=0,tf=400,N=100):
	ptimes=10
	Nt = 2*N
	pi = np.zeros((Nt))
	pi[0] = 1
	N_pars = len(params)
	xi = np.zeros((N_pars+1)*Nt)
	xi[0] = 1

	A = csc_matrix(getA(params,N))
	soln = GenericFSP(ti,tf,xi,A,pi,ptimes=ptimes)
	marginal = np.zeros((N,ptimes))
	for i in range(N):
		marginal[i,:] =  np.sum( soln[2*i:2*i+2,:], axis=0)
	return marginal 

class mRNAkinetics(object):

	def __init__(self, vals):
		self.vals = vals	
		self.estimate = None
		self.init = None

	def LogLikelihood(self,params,values):
		marginal = solve(params)	
		last_p = marginal[:,-1]
		N = 100
		hist, bin_edges = np.histogram(values, range=(0,N),bins=N)
		return np.dot(last_p,hist)

	def MomentInference(self, value):
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

	def MaximumLikelihood(self,method = 'L-BFGS-B'):

		if len(self.vals) == 0: 
			self.estimate = [np.nan, np.nan, np.nan]
			return 0
		vals_ = np.copy(self.vals)
		self.init_param = self.MomentInference(vals_)
		init = self.MomentInference(vals_)
		if np.isnan(init).any() or any(init < 0):
			init = np.array([1,10,10])
			self.init_param = np.array([1,10,10])
		bnds = ((1e-4,1e4),(1e-4,1e4), (1e-4, 1e4))
		try:
			ll = minimize(self.LogLikelihood,init,args = (vals_),method=method,bounds=bnds)
			self.estimate = list(ll.x)
		except:
			self.estimate = [np.nan,np.nan,np.nan]
		print(init)
	def get_estimate(self):
		return self.estimate

def main():
	mtx = sys.argv[1]
	df = pd.read_csv(mtx, index_col=0)

	gene_index = df.index.tolist()
	mrna_kpe_df = pd.DataFrame(index=gene_index, columns=['kon','koff','ksyn'])
	for i, record in df.iterrows():
		reads = record.values
		obj1  = mRNAkinetics(reads)
		obj1.MaximumLikelihood()
		mrna_kpe = obj1.get_estimate()
		mrna_kpe_df.loc[i,['kon','koff','ksyn']] = mrna_kpe
		print(mrna_kpe)
	mrna_kpe_df.to_csv('simulation_kpe.txt')
	
if __name__ == "__main__":
	main()
