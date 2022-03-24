import csv
import numpy as np
import scipy as sp
import scipy.stats as st
from scipy.stats import poisson
from scipy.stats import beta
import matplotlib as mpl
import matplotlib.pyplot as plt
mpl.rcParams['font.family'] = 'Arial'
plt.rcParams['font.size'] = 11
plt.rcParams['axes.linewidth'] = 2
mpl.use('Agg')

def main():

	param = [3,6,10,1]

	fig, axs = plt.subplots()
	kp = param[0]
	kn = param[1]
	r =  param[2]
	d = param[3]

	x = np.linspace(0,r/d,100)

	betaab = sp.special.beta(kp/d, kn/d)
	C = d / ( np.power(r,(kp+kn)/d ) * betaab )
	p0 = C * np.power(d * x, kp/d - 1) * np.power(r - d * x, kn/d)
	p1 = C * np.power(d * x, kp/d ) * np.power(r - d * x, kn/d - 1)
	p = p0 + p1

	n=10000
	p1 = beta.rvs(kp, kn, size=n)
	mu = r * p1
	vals = np.array([ poisson.rvs(imu) for imu in mu])

	axs.hist(vals, density=True,label='experimental observation', edgecolor="white")
	axs.plot(x, p, '-',label='k+=%s;k-=%s;r=%s,d=%s'%(kp, kn, r, d),c='k')
	axs.legend()
	axs.set_ylabel('probability density')
	axs.set_xlabel('# mRNA')
	plt.savefig('pb.pdf')
	plt.show()

if __name__ == "__main__":
	main()

