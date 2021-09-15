from matplotlib.colors import ListedColormap
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import matplotlib as mpl
#mpl.use('Agg')
mpl.rcParams['font.family'] = 'Arial'
plt.rcParams['font.size'] = 12
plt.rcParams['axes.linewidth'] = 1
from sklearn.decomposition import PCA
import scipy.stats as st
import seaborn as sns
import pandas as pd
import numpy as np
import sys

def fun(x,y):
	logx = np.log10(x)
	logy = np.log10(y)
	if logy > logx - 0.3 and logy < logx + 0.3: return 1
	else: return 0
	
def pca_compare(kpe_df, out):

	fig, ax = plt.subplots(constrained_layout=True,ncols=3,figsize=(9, 3))

	var_dict = {'Burst':('burst size','burst frequency'),\
		'Time':('n','τ')}
	
	bool_select = kpe_df['consist'] == 1
	for i, (key, var) in enumerate(var_dict.items()):
		var1,var2 = var
		xs1 = kpe_df.loc[~bool_select, var1]
		ys1 = kpe_df.loc[~bool_select, var2]
		ax[i].scatter(xs1, ys1, s=1, c='gray', label='original')
	
		xs2 = kpe_df.loc[ bool_select, var1]
		ys2 = kpe_df.loc[ bool_select, var2]
		ax[i].scatter(xs2, ys2, s=1, c='dodgerblue', label='filtered')
		
		ax[i].set_xlabel(var1)
		ax[i].set_ylabel(var2)
		ax[i].set_title(key, loc='right')

		if key == 'Burst':
			ax[i].set_xscale('log')
			ax[i].set_yscale('log')

	color = {0:'gray', 1:'dodgerblue'}
	kpe_df_reform = pd.melt(kpe_df, id_vars=['consist'], \
			value_vars=['log(kon)','log(koff)','log(ksyn)'], var_name='kp',value_name='kp_val')

	g = sns.violinplot(x='kp', y='kp_val', hue='consist', data=kpe_df_reform, ax=ax[2], \
			palette=color, linewidth=0)
	ax[2].set_ylabel([])
	ax[2].set_xticklabels(['log(k+)','log(k-)','log(r)'],rotation=90)
	ax[2].set_title('Distribution', loc='right')
	g.set(xlabel=None)
	g.set(ylabel='value')
	g.legend(loc='center left', frameon=False, bbox_to_anchor=(1, 0.5))
	legend = g.axes.get_legend()
	legend.set_title('')

#	stat_df_reform = pd.melt(kpe_df, id_vars=['consist'], \
#			value_vars=['col','chisq_pval','simlr_pval'], var_name='stat',value_name='stat_val')
#	stat_df_reform['log(stat_val)'] = stat_df_reform['stat_val'].transform(np.log10) 
#
#	stat_df_reform.replace({-np.inf:-100, np.inf:100}, inplace=True)
#	print(stat_df_reform.loc[(stat_df_reform['consist']==False) & (stat_df_reform['stat']=='col')])
#
#	g1 = sns.violinplot(x='stat', y='log(stat_val)', hue='consist', data=stat_df_reform, ax=ax[3], \
#			palette=color, linewidth=0)
#	ax[3].set_yscale('log')
#	ax[3].set_xticklabels(['log(k+)','log(k-)','log(r)'],rotation=90)
#	ax[3].set_title('Distribution', loc='right')

	# replace labels
	new_labels = ['inconsistant', 'consistant']
	for t, l in zip(legend.texts, new_labels): t.set_text(l)

	num_gene = kpe_df.shape[0]
	num_gene_filter = np.sum(kpe_df['consist'] == 1)
	num_gene_rest = num_gene - num_gene_filter
	plt.suptitle(out)
	plt.figtext(0.8, 0.01,"consistant/inconsistant:%s/%s"%(num_gene_filter,num_gene_rest),\
			size='x-small')
	plt.savefig('%s.pdf'%out, dpi=300, bbox_inches ='tight')
#	plt.show()

def main():

	kpe_in = sys.argv[1]
	prefix = sys.argv[2]

	kpe_df = pd.read_csv(kpe_in,header=0, index_col=0)
	input_col = kpe_df.columns.tolist()

	ori_cols = ['kon','koff','ksyn']
	sim_cols = ['sim_' + x for x in ori_cols ]
	log_cols = ['log(kon)','log(koff)','log(ksyn)']

#	# method comparison only
#	pb_cols = [ x + '_pb'   for x in ori_cols ]
#	full_cols = [ x + '_full' for x in ori_cols ]
#	for i,(ikp, jkp) in enumerate(zip(pb_cols, ori_cols)):
#		kpe_df.rename({ikp:jkp}, axis=1, inplace=True)
#
#	for i,(ikp, jkp) in enumerate(zip(full_cols, sim_cols)):
#		kpe_df.rename({ikp:jkp}, axis=1, inplace=True)
#
	for i, (kp1,kp2) in enumerate(zip(ori_cols, sim_cols)):
		new_col = 'col' + str(i)
		kpe_df[new_col] = kpe_df.apply(lambda row: fun(row[kp1],  row[kp2]),  axis=1)
	kpe_df['col']  = kpe_df.apply(lambda row: row['col0']*row['col1']*row['col2'], axis=1)
	kpe_df['consist']  = (kpe_df['col']==1)& ((kpe_df['chisq_pval'] > 0.05)|(kpe_df['simlr_pval'] > 0.05))

	kpe_df[log_cols] = kpe_df[ori_cols].transform(np.log10)	
	pca = PCA(n_components=2)
	pca.fit(kpe_df[log_cols])

	kpe_df_pca = pca.transform(kpe_df[log_cols])
	kpe_df[['PC1','PC2']] = kpe_df_pca
	kpe_df['burst size']  = kpe_df['ksyn']/kpe_df['koff']
	kpe_df['burst frequency']  = kpe_df['kon'] * kpe_df['koff'] /(kpe_df['kon'] + kpe_df['koff'])
	kpe_df['n'] = kpe_df['kon']/(kpe_df['kon']+kpe_df['koff'])
	kpe_df['τ']   = 1/(kpe_df['kon']+kpe_df['koff'])

	pca_compare(kpe_df, prefix)
	kpe_consist = kpe_df[kpe_df['consist']==1]
	kpe_consist[input_col].to_csv(prefix+'.est.consist', float_format="%.5f")

if __name__ == "__main__":
	main()


