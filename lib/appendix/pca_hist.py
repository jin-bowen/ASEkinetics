import matplotlib as mpl
import matplotlib.pyplot as plt
mpl.rcParams['font.family'] = 'Avenir'
plt.rcParams['font.size'] = 12
plt.rcParams['axes.linewidth'] = 2
from sklearn.cluster import AgglomerativeClustering
from sklearn.decomposition import PCA
from matplotlib.colors import ListedColormap
import matplotlib.patches as mpatches
import scipy.stats as st
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import sys

def poisson(kpe_df1):

	fig, ax = plt.subplots(constrained_layout=True)
	xs1 =  kpe_df1['mean']
	ys1 =  kpe_df1['var']
	ratio = kpe_df1['var'] / kpe_df1['mean']
	print(ratio)
	print(kpe_df1[ratio<0.9])
	ax.hist(ratio,range(0,2))
	
	plt.show()
	
def pca_compare(kpe_df1,kpe_df2):

	fig, ax = plt.subplots(constrained_layout=True)#,ncols=2,sharex=True,sharey=True)
	xs1 = kpe_df1['pc1']
	ys1 = kpe_df1['pc2']

	xs2 = kpe_df2['pc1']
	ys2 = kpe_df2['pc2']
	colors = {'G1':'gray', 'S':'green', 'G2M':'yellow'}
	var_color = [ colors[x] for x in kpe_df2['cycle'].values ]

	ax.scatter(xs1, ys1, s=2, c='silver', label='NA12878')
	ax.scatter(xs2, ys2, s=2, c=var_color, label='simulation')
	
	ax.set_xlabel('pc1')
	ax.set_ylabel('pc2')

	ax.xaxis.set_tick_params(which='major', size=10, width=2, direction='in')
	ax.yaxis.set_tick_params(which='major', size=10, width=2, direction='in')
	
	plt.legend(frameon=False)
#	plt.savefig('NA12878_K562.tiff', dpi=300,bbox_inches ='tight')
	plt.show()

def main():
	kpe_in1 = sys.argv[1]
	kpe_in2 = sys.argv[2]
	#count_file = sys.argv[3]

	kpe_df1_raw = pd.read_csv(kpe_in1)
	#kpe_df1_raw.set_index('gene',inplace=True)
	kpe_df1  = kpe_df1_raw.dropna()
	#kpe_df1 = kpe_df1_raw[kpe_df1_raw['n']>50].dropna()

	kpe_df1['burst_size'] = kpe_df1['ksyn']/kpe_df1['koff']
	kpe_df1['burst_freq'] = kpe_df1['kon'] * kpe_df1['koff'] / (kpe_df1['kon'] + kpe_df1['koff'])
	kpe_df1['mean_occu'] = kpe_df1['kon']/(kpe_df1['kon']+kpe_df1['koff'])
	kpe_df1['tau']       = 1/(kpe_df1['kon']+kpe_df1['koff'])

	kpe_df2_raw = pd.read_csv(kpe_in2,index_col=0,header=0)
	#kpe_df2 = kpe_df2_raw[['kon_est','koff_est','r_est']]
	kpe_df2 = kpe_df2_raw
	#kpe_df2.columns = ['kon','koff','ksyn']

	kpe_df2['burst_size'] = kpe_df2['ksyn']/kpe_df2['koff']
	kpe_df2['burst_freq'] = kpe_df2['kon'] * kpe_df2['koff'] / (kpe_df2['kon'] + kpe_df2['koff'])
	kpe_df2['mean_occu'] = kpe_df2['kon']/(kpe_df2['kon']+kpe_df2['koff'])
	kpe_df2['tau']       = 1/(kpe_df2['kon']+kpe_df2['koff'])

	#count = pd.read_csv(count_file,index_col=0,header=0)
	#kpe_df2['mean'] = count.mean(axis=1)
	#kpe_df2['var' ] = count.var(axis=1)

	ori_cols = ['kon','koff','ksyn']
	log_cols = ['log(kon)','log(koff)','log(ksyn)']

	kpe_df1[log_cols] = kpe_df1[ori_cols].transform(np.log10)
	kpe_df1['ind'] = 'NA12878'
	kpe_df1 = kpe_df1.dropna()
	pca = PCA(n_components=2)
	pca.fit(kpe_df1[log_cols])
	kpe_df1_pca = pca.transform(kpe_df1[log_cols])
	kpe_df1[['pc1','pc2']] = kpe_df1_pca

	kpe_df2[log_cols] = kpe_df2[ori_cols].transform(np.log10)
	kpe_df2 = kpe_df2.dropna()
	kpe_df_pca = pca.transform(kpe_df2[log_cols])
	kpe_df2[['pc1','pc2']] = kpe_df_pca
	pca_compare(kpe_df1, kpe_df2)

#	poisson(kpe_df2)

if __name__ == "__main__":
	main()


