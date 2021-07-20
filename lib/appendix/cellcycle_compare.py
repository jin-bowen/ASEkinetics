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

def fun(x,y):
	logx = np.log10(x)
	logy = np.log10(y)
	if logy > logx - 0.3 and logy < logx + 0.3: return 1
	else: return 0
	
def pca_compare(kpe_df,kpe_df2,kpe_df3):

	fig, ax = plt.subplots(constrained_layout=True,ncols=3)

	var_dict = {'pc':('pc1','pc2'),'burst':('burst_size','burst_freq'),\
		'time':('mean_occu','tau')}

	for i, (key, var) in enumerate(var_dict.items()):

		var1,var2 = var
		xs1 = kpe_df[var1]
		ys1 = kpe_df[var2]
		ax[i].scatter(xs1, ys1, s=2, c='gray', label='original')
	
		bool_select = kpe_df3['col'] == 1
		xs2 = kpe_df2.loc[bool_select,var1]
		ys2 = kpe_df2.loc[bool_select,var2]
		ax[i].scatter(xs2, ys2, s=2, c='dodgerblue', label='simulation',alpha=0.5)
		
		ax[i].set_xlabel(var1)
		ax[i].set_ylabel(var2)
		ax[i].set_title(key, loc='right')

	num_gene = kpe_df3.shape[0]
	num_gene_filter = np.sum(kpe_df3['col'] == 1)
	plt.suptitle("filtered/all:%s/%s"%(num_gene_filter,num_gene))
	plt.legend(frameon=False)
	plt.savefig('NA12878_G1_sim_org_comparison.png', dpi=300,bbox_inches ='tight')
	plt.show()

def main():
	kpe_in_cycle = sys.argv[1]
	kpe_in_all   = sys.argv[2]

	kpe_cycle_raw = pd.read_csv(kpe_in_cycle,header=0)
	kpe_cycle_raw.set_index(['gene','allele'],inplace=True)
	kpe_cycle  = kpe_cycle_raw.dropna()
	kpe_G1 = kpe_cycle[kpe_cycle['cycle']=='G1']

	kpe_all_raw = pd.read_csv(kpe_in_all,header=0)
	kpe_all_raw.set_index(['gene','allele'],inplace=True)
	kpe_all  = kpe_all_raw.dropna()

	ori_cols = ['kon','koff','ksyn']
	sim_cols = [ x + '_G1' for x in ori_cols ]
	log_cols = ['log(kon)','log(koff)','log(ksyn)']

	kpe_df = pd.merge(kpe_G1, kpe_all,left_index=True,right_index=True,suffixes=('_G1',''))

	kpe_df['col1'] = kpe_df.apply(lambda row: fun(row['kon_G1'],  row['kon']),  axis=1)
	kpe_df['col2'] = kpe_df.apply(lambda row: fun(row['koff_G1'], row['koff']), axis=1)
	kpe_df['col3'] = kpe_df.apply(lambda row: fun(row['ksyn_G1'], row['ksyn']), axis=1)
	kpe_df['col']  = kpe_df.apply(lambda row: row['col1']*row['col2']*row['col3'], axis=1)

	kpe_ori = kpe_df[ori_cols]
	kpe_sim = kpe_df[sim_cols]
	kpe_consist = kpe_df[kpe_df['col']!=1]

	kpe_ori[log_cols] = kpe_ori[ori_cols].transform(np.log10)
	pca = PCA(n_components=2)
	pca.fit(kpe_ori[log_cols])
	kpe_ori_pca = pca.transform(kpe_ori[log_cols])
	kpe_ori[['pc1','pc2']] = kpe_ori_pca
	kpe_ori['burst_size'] = kpe_ori['ksyn']/kpe_ori['koff']
	kpe_ori['burst_freq'] = kpe_ori['kon'] * kpe_ori['koff'] / (kpe_ori['kon'] + kpe_ori['koff'])
	kpe_ori['mean_occu'] = kpe_ori['kon']/(kpe_ori['kon']+kpe_ori['koff'])
	kpe_ori['tau']       = 1/(kpe_ori['kon']+kpe_ori['koff'])

	kpe_sim[ori_cols] = kpe_sim[sim_cols]
	kpe_sim[log_cols] = kpe_sim[sim_cols].transform(np.log10)
	kpe_df_pca = pca.transform(kpe_sim[log_cols])
	kpe_sim[['pc1','pc2']] = kpe_df_pca
	kpe_sim['burst_size'] = kpe_sim['ksyn']/kpe_sim['koff']
	kpe_sim['burst_freq'] = kpe_sim['kon'] * kpe_sim['koff'] / (kpe_sim['kon'] + kpe_sim['koff'])
	kpe_sim['mean_occu'] = kpe_sim['kon']/(kpe_sim['kon']+kpe_sim['koff'])
	kpe_sim['tau']       = 1/(kpe_sim['kon']+kpe_sim['koff'])

	pca_compare(kpe_ori, kpe_sim, kpe_df)
	kpe_consist.to_csv('NA12878.mle.inconsist')

if __name__ == "__main__":
	main()


