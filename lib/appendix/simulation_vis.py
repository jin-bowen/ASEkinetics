import sys
import numpy as np
import pandas as pd
import scipy.stats as st
import matplotlib.pyplot as plt
import pickle

def fun(x,y):
	logx = np.log10(x)
	logy = np.log10(y)
	if logy > logx - 1 and logy < logx + 1: return 1
	else: return 0

def main():

	est_file   = sys.argv[1]
	param_file = sys.argv[2]
	count_file = sys.argv[3]

	count        = pd.read_csv(count_file,index_col=0,header=0)
	filter = count[count>0].sum(axis=1) > 200
	estimations_raw = pd.read_csv(est_file,  index_col=0,header=0)
	params_raw      = pd.read_csv(param_file,index_col=0,header=0)
	
	estimations = estimations_raw.astype('float')
	params      = params_raw.astype('float')

	estimations['n']   = estimations['kon']/(estimations['kon']+estimations['koff'])
	estimations['tau'] = 1/(estimations['kon']+estimations['koff'])
	estimations['bs']  = estimations['r']/estimations['koff']

	params['n']   = params['kon']/(params['kon']+params['koff'])
	params['tau'] = 1/(params['kon']+params['koff'])
	params['bs']  = params['r']/params['koff']

	kp_list = ['kon','koff','r']
	kp_dev  = ['n','tau']
		
	fig, axs = plt.subplots(ncols=len(kp_list),nrows=2,constrained_layout=True)
	fig.suptitle('correlation between estimated and true parameters')

	for ikp, kp in enumerate(kp_list):
		estimation = estimations[kp].fillna(0).values
		param      = params[kp].fillna(0).values

		corr = st.pearsonr(estimation, param)[0].round(decimals=3)
		x = np.linspace(1e-2, 1e3, 1000)
		axs[0,ikp].set_xscale('log')
		axs[0,ikp].set_yscale('log')
		axs[0,ikp].scatter(param, estimation, c='k',s=0.5)
		axs[0,ikp].set_title('corr=%s'%(str(corr)))
		axs[0,ikp].plot(x, x, ls="--", c="k",lw=0.1)
		axs[0,ikp].set_ylabel('estimated %s'%kp)
		axs[0,ikp].set_xlabel('true %s'%kp)
		axs[0,ikp].axis('square')

	for ikp, kp in enumerate(kp_dev):
		estimation = estimations[kp].fillna(0).values
		param      = params[kp].fillna(0).values
		corr = st.pearsonr(estimation, param)[0].round(decimals=3)
	
		axs[1,ikp].scatter(param, estimation, c='k',s=1)
		axs[1,ikp].set_title('corr=%s'%(str(corr)))
		axs[1,ikp].set_ylabel('estimated %s'%kp)
		axs[1,ikp].set_xlabel('true %s'%kp)
		axs[1,ikp].axis('square')

	df = pd.merge(estimations,params,left_index=True,right_index=True,suffixes=('_est',''))
	
	bool_filter = []
	df['col1'] = df.apply(lambda row: fun(row['kon_est'], row['kon']), axis=1)
	df['col2'] = df.apply(lambda row: fun(row['koff_est'], row['koff']), axis=1)
	df['col3'] = df.apply(lambda row: fun(row['r_est'], row['r']), axis=1)	
	df['col']  = df.apply(lambda row: row['col1']*row['col2']*row['col3'], axis=1)

	df['mean'] = count.mean(axis=1)
	df['var' ] = count.var(axis=1)
	df['ratio'] =  df['var' ] / df['mean']

	cols = ['kon_est','koff_est','r_est','kon','koff','r']
	df.loc[df['col']==1,cols].to_csv('high_conf_est')
#	plt.show()
	plt.savefig(est_file+'.png')
	
if __name__ == "__main__":
	main()

