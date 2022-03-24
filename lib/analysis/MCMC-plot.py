import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.lines import Line2D
import matplotlib as mpl
plt.rcParams['font.size'] = 10
plt.rcParams['axes.linewidth'] = 2
import scipy.stats as st
import seaborn as sns
import pandas as pd
import numpy as np
import sys

def main():

	mcmc_data_in = sys.argv[1]
	mcmc_data = pd.read_csv(mcmc_data_in,header=0)
	fig, ax = plt.subplots(nrows=4,ncols=2,constrained_layout=True,\
	                        gridspec_kw={'width_ratios': [3, 1]})
	
	x = mcmc_data.index.tolist()
	
	cols = ['kon','koff','ksyn','ll']
	burnin=int(len(mcmc_data)/2)
	
	for ikp, kp in enumerate(cols):
	    ax[ikp,0].set_xlabel('iterations')
	    ax[ikp,0].set_ylabel(kp)
	    kp_data = mcmc_data[kp].values
	    kp_data_nozero = kp_data[kp_data!=0]
	    
	    lower = np.min(kp_data_nozero) * 0.95
	    upper = np.max(kp_data_nozero) * 1.05
	    ax[ikp,0].set_ylim([lower, upper])    
	    
	    ax[ikp,0].scatter(x[:burnin],kp_data[:burnin],s=2, c='gray')
	    ax[ikp,0].scatter(x[burnin:],kp_data[burnin:],s=2, c='k')
	    
	    kp_data_nozero = kp_data[burnin:]
	    kp_data_nozero = kp_data_nozero[kp_data_nozero!=0]
	    ax[ikp,1].hist(kp_data_nozero,color='k')
	    ax[ikp,1].set_xlabel(kp)
	
	ax[ikp,0].set_ylabel('log(likelihood)')
	ax[ikp,1].set_xlabel('log(likelihood)')

	plt.savefig(mcmc_data_in.split('.')[0]+'.pdf')

if __name__ == "__main__":
	main()


	
	
