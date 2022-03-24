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
import re
import sys

def fun(x,y):
	logx = np.log10(x)
	logy = np.log10(y)
	if logy > logx - 0.3 and logy < logx + 0.3: return 1
	else: return 0

def ci_plot(kpe):

	fig, ax = plt.subplots(constrained_layout=True,ncols=3,figsize=(9, 3))

	kp_list = ['kon','koff','ksyn']
	for i, var in enumerate(kp_list):
		var_mean = var + '_mean'		
		var_ci = var + '_ci'

		xs = kpe[var_mean]
		ys = kpe[var_ci] 

		ax[i].scatter(xs, ys, s=2)
		ax[i].set_xlabel('kp')
		ax[i].set_ylabel('log2(ci)')
		ax[i].set_title(var, loc='right')
	plt.show()
#	plt.savefig('%s.pdf'%out, dpi=300, bbox_inches ='tight')
	
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

def main():

	kpe_var_in  = sys.argv[1]
	kpe_eval_in = sys.argv[2]
	kpe_est_in  = sys.argv[3]
	prefix      = sys.argv[4]
	
	kpe_var  = pd.read_csv(kpe_var_in, header=0, index_col=0)
	kpe_eval_all = pd.read_csv(kpe_eval_in, header=0, index_col=0)

	# process evaluation file
	ori_cols = ['kon','koff','ksyn']
	sim_cols = ['sim_' + x for x in ori_cols ]
	log_cols = ['log(kon)','log(koff)','log(ksyn)']

	for i, (kp1,kp2) in enumerate(zip(ori_cols, sim_cols)):
		new_col = 'col' + str(i)
		kpe_eval_all[new_col] = kpe_eval_all.apply(lambda row: fun(row[kp1],  row[kp2]),  axis=1)
	kpe_eval_all['col']  = kpe_eval_all.apply(lambda row: row['col0']*row['col1']*row['col2'], axis=1)
	kpe_eval_all['consist']  = (kpe_eval_all['col']==1)& ((kpe_eval_all['ks_pval'] > 0.05)|(kpe_eval_all['simlr_pval'] > 0.05))
	kpe_eval_all['consist1']  = (kpe_eval_all['col']==1)& ((kpe_eval_all['chisq_pval'] > 0.05)|(kpe_eval_all['simlr_pval'] > 0.05))

	print(kpe_eval_all['consist'].sum())
	print(kpe_eval_all['consist1'].sum())
	print(np.sum(kpe_eval_all[['consist','consist1']].sum(axis=1)==2))

	kpe_eval_all[log_cols] = kpe_eval_all[ori_cols].transform(np.log10)	
	pca = PCA(n_components=2)
	pca.fit(kpe_eval_all[log_cols])

	kpe_eval_all_pca = pca.transform(kpe_eval_all[log_cols])
	kpe_eval_all[['PC1','PC2']] = kpe_eval_all_pca
	kpe_eval_all['burst size']  = kpe_eval_all['ksyn']/kpe_eval_all['koff']
	kpe_eval_all['burst frequency']  = kpe_eval_all['kon'] * kpe_eval_all['koff'] /(kpe_eval_all['kon'] + kpe_eval_all['koff'])
	kpe_eval_all['n'] = kpe_eval_all['kon']/(kpe_eval_all['kon']+kpe_eval_all['koff'])
	kpe_eval_all['τ']   = 1/(kpe_eval_all['kon']+kpe_eval_all['koff'])


	kpe_eval = kpe_eval_all[kpe_eval_all['consist']==1]
	# filter variance profile
	kp_cols = ['kon','koff','ksyn']	
	for kp in kp_cols:
		mean = kp + '_mean'
		lower = kp + '_low'
		upper = kp + '_upper'
		ci = kp + '_ci' 

		select = (kpe_var[mean] > kpe_var[lower]) & (kpe_var[mean] < kpe_var[upper])
		kpe_var.loc[select,ci] = ( kpe_var.loc[select,upper] - kpe_var.loc[select,lower] )/ kpe_var.loc[select,mean]

	kpe_var[ci] = kpe_var[ci].transform(np.log2)
	ci_cols = [ x + '_ci' for x in kp_cols ]
	select = (kpe_var[ci_cols] <= 1) 
	stat = np.sum(select, axis=1)	
	kpe_var_filter = kpe_var[stat > 2]
	
	# consist between variance and evaluation profile
	kpe_union = pd.merge(kpe_eval, kpe_var_filter, how='outer', \
			suffixes=('_eval',''), left_index=True, right_index=True)
	for kp in ori_cols:
		kpe_union[kp] = kpe_union.apply(lambda x: x['%s_eval'%kp] if x['%s'%kp] is np.nan \
							else x[kp], axis=1)
	print("# var", len(kpe_var))
	print("% var< 2", len(kpe_var[stat > 2])/len(kpe_var))
	print("# eval", len(kpe_eval))
	print("# union", len(kpe_union))

	output_cols = kp_cols + ['n','mean','var']
	kpe_est = pd.read_csv(kpe_est_in, header=0, index_col=0)
	kpe_est['pass'] = 0
	kpe_est.loc[kpe_union.index.tolist(),'pass'] = 1

	kpe_union.to_csv('%s.vda'%(prefix),float_format="%.5f")
	kpe_est.to_csv('%s.est.consist'%(prefix),float_format="%.5f")

if __name__ == "__main__":
	main()

