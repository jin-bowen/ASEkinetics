from __future__ import division
from scipy.special import kv, gammaln, gamma, hyp1f1, factorial,j_roots
from scipy.special import beta as beta_fun
from scipy.stats import gmean, ks_2samp, anderson_ksamp, chi2, vonmises
from scipy.stats import poisson as poissonF
from decimal import Decimal, getcontext
from collections import namedtuple
from numpy import log, array, zeros, median, rint, power, hstack
from numpy import hsplit, seterr, mean, isnan, floor, divide, exp, round, where
from numpy.random import beta, poisson, random
import matplotlib.pyplot as plt
import numpy as np
import mpmath as mp
import sys

seterr(all='ignore')
mp.mp.dps = 30 
mp.mp.pretty = True

Params = namedtuple("Params", ["alpha", "beta", "gamma", "c"])
BioParams = namedtuple("BioParams", ["size", "freq", "duty"])

class RVar:

	def __init__(self, value=0):
		self.value = value
		self.leftLimit = 0
		self.rightLimit = float('Inf')
		self.sample = []

	def mean(self):
		return mean(self.sample)

	def setSampleFunction(self, function):
		self.sampleFunction = function

	def draw(self, maxSteps=1000, saveToSample = False):

		x0 = self.value
		w = abs(self.value / 2)

		f = self.sampleFunction;

		logPx = f(x0)
		logSlice  = log(random()) + logPx

		xLeft = x0 - random() * w
		xRight = xLeft + w

		if xLeft < self.leftLimit: xLeft = self.leftLimit
		if xRight > self.rightLimit: xRight = self.rightLimit

		v = random()
		j = floor(maxSteps*v)
		k = maxSteps-1 - j
		
		while j > 0 and logSlice < f(xLeft) and xLeft - w > self.leftLimit:
			j = j-1
			xLeft = xLeft - w

		while k > 0 and logSlice < f(xRight) and xRight + w < self.rightLimit:
			k = k - 1
			xRight = xRight + w

		n = 10000
		while 1:
			n = n - 1
			if n < 0 :
				print("Warning: Can't find a new value.")
				return x0

			x1 = (xRight - xLeft) * random() + xLeft
			
			if logSlice <= f(x1): break
			if x1 < x0: xLeft = x1
			else: xRight = x1

		self.value = x1
		if saveToSample: self.sample.append(x1)
		return x1

# Estimation of the parameters from the moments
def getParamsMoments(p):

	try:
		rm1 = sum(p) / len(p)
		rm2 = sum( [pow(x,2) for x in p] ) / len(p)
		rm3 = sum( [pow(x,3) for x in p] ) / len(p) 

		fm1 = rm1
		fm2 = rm2 - rm1
		fm3 = rm3 - 3*rm2 + 2*rm1

		r1 = fm1
		r2 = fm2 / fm1
		r3 = fm3 / fm2

		alpha = 2*r1 * (r3 - r2) / (r1*r2 - 2*r1*r3 + r2*r3)
		beta = 2 * (r2 - r1) * (r1 - r3) * (r3 - r2) / ((r1*r2 - 2*r1*r3 + r2*r3) *(r1 - 2*r2 + r3))
		gamma = (-r1*r2 + 2*r1*r3 - r2*r3) / (r1 - 2*r2 +r3)
	except:
		return Params(-1,-1,-1,-1)

	return Params(alpha, beta, gamma, 0)

# Esimation of the parameters of a sample drawn from a Poisson-beta distribution using bayesian inference method
def getParamsBayesian(p, iterN=1000):

	HyperParams = namedtuple("HyperParams", ["k_alpha", "theta_alpha", "k_beta", "theta_beta", "k_gamma", "theta_gamma"])
	parFit = getParamsMoments(p)
	hyperParams = HyperParams(k_alpha = 1, theta_alpha = 100, k_beta = 1, theta_beta = 100, k_gamma = 1, theta_gamma = max(p) )
	
	if parFit.alpha > 0 and parFit.beta > 0 and parFit.gamma > 0:
		params = Params(alpha = RVar(parFit.alpha), beta = RVar(parFit.beta), gamma = RVar(parFit.gamma), c = [ RVar(0.5) for i in range(len(p)) ] )	
	else:
		params = Params(alpha = RVar(0.5), beta = RVar(0.5), gamma = RVar(mean(p)+1e6), c = [ RVar(0.5) for i in range(len(p)) ] )

	bioParams = BioParams(size = RVar(), freq = RVar(), duty = RVar() )
	log_ll_list = []
	save = False
	for i in range(iterN):

		if i > iterN / 2:
			save = True

			alpha = params.alpha.value
			beta = params.beta.value
			gamma = params.gamma.value

			bioParams.size.sample.append( gamma / beta )
			bioParams.freq.sample.append( alpha*beta / (alpha + beta) )
			bioParams.duty.sample.append( alpha / (alpha + beta)  )

		for c,pi in zip(params.c,p):
			c.setSampleFunction(lambda x: (params.alpha.value - 1 ) * log(x) + \
					(params.beta.value - 1) * log(1-x) + pi * log(x) - \
					params.gamma.value * x)
			c.draw(saveToSample = save)

		params.gamma.setSampleFunction(lambda x: (hyperParams.k_gamma-1)*log(x) - \
					x / hyperParams.theta_gamma + \
					log(x) * sum(p) - \
					x * sum( [c.value for c in params.c] ) )
		params.gamma.draw(saveToSample = save)
		
		params.alpha.setSampleFunction(lambda x: (hyperParams.k_alpha-1)*log(x) - \
					x / hyperParams.theta_alpha + \
					len(p)*(gammaln(x + params.beta.value) - \
					gammaln(x)) + \
					(x-1) * sum([ log(c.value) for c in params.c ]) )
		params.alpha.draw(saveToSample = save)

		params.beta.setSampleFunction(lambda x: (hyperParams.k_beta-1)*log(x) - \
					x / hyperParams.theta_beta + \
					len(p)*(gammaln(x + params.alpha.value) - \
					gammaln(x)) + \
					(x-1) * sum([ log(1-c.value) for c in params.c ]) )
		params.beta.draw(saveToSample = save)
		if i < np.floor(0.1*iterN):continue
#		log_ll = logLikelihood(p, params, i)

		log_ll = LogLikelihood(params,p)
		log_ll_list += [log_ll]
	
	return params, bioParams,log_ll_list

def randPoissonBeta(params,n):

	x = beta( params.alpha, params.beta, n)
	p = poisson( x * params.gamma )

	return p

# Find log of likelihood for sample 'p' with parameters 'params', doing poisson-beta sampling 'n' times 
def empirical_logLikelihood(p, params):

	try:
		alpha = params.alpha.mean()
		beta_ = params.beta.mean()
		gamma =  int(round(params.gamma.mean()))+1
	except:
		alpha = params.alpha
		beta_ = params.beta
		gamma =  int(round(params.gamma))+1

	if n * gamma > 1e9:
		n = int( round(1e9 / gamma) )
		logStatus( Status(1, idx, "Reduced Poisson-Beta sample to " + str(n) + ".") )

	pVal = []
	for item in p:
		x = beta(alpha, beta_, n)
		pTemp = 0

		for i in range(n):
			pTemp += poissonF.pmf(item, gamma*x[i])
		pVal.append(pTemp / n)

	pVal = array(pVal)
	pVal[where(pVal==0)] = 1/n

	return sum(log(pVal))

def fun(at, m):
        if (max(m) < 1e6):return(poissonF.pmf(at,m))
        else:return(norm.pdf(at,loc=m,scale=sqrt(m)))

def dBP(at, alpha, bet, lam):
	at.shape = (len(at), 1)
	np.repeat(at, 50, axis = 1)
	x,w = j_roots(50,alpha = bet - 1, beta = alpha - 1)
	gs = np.sum(w*fun(at, m = lam*(1+x)/2), axis=1)
	prob = 1/beta_fun(alpha, bet)*2**(-alpha-bet+1)*gs
	return(prob)

def LogLikelihood(params,p):
	alpha = params.alpha.value
	beta_ = params.beta.value
	gamma =  int(round(params.gamma.value))+1

	return(-np.sum(np.log(dBP(p, alpha, beta_, gamma) + 1e-10)))

def main():
	mtx = sys.argv[1]
	reads = np.loadtxt(mtx)
	params1, bioParams1,log_ll = getParamsBayesian(reads, iterN=1000)	
	print(params1.alpha.mean(), params1.beta.mean(), params1.gamma.mean())

	fig, ax = plt.subplots()
	n = len(log_ll)
	ax.scatter(list(range(n)), log_ll, s=1)
	plt.savefig('log_ll_monitor.png')

if __name__ == "__main__":
	main()



