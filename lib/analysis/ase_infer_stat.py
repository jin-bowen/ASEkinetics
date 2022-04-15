import seaborn as sns
import matplotlib as mpl
import matplotlib.pyplot as plt
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = ['Arial']
plt.rcParams['font.size'] = 10
plt.rcParams['axes.linewidth'] = 2
mpl.use('Agg')

import dask.dataframe as dd
import pandas as pd
import numpy as np
import scipy as sp
import sys

def main():
	
	# number of cell sampled
	ase_profile  = sys.argv[1]

	ase = dd.read_csv(ase_profile, header=0)
	select = (ase['ub_mp'] ==1) 
	ase = ase.loc[select]
	
	print(ase['umi'].compute().mean())
	print(ase['umi'].compute().median())
	print(ase['umi'].compute().max())

	plt.hist(ase['umi'].compute())
	plt.yscale('log', nonposy='clip')
	plt.savefig('ase_dist.png')
	plt.show()

if __name__ == "__main__":
	main()



