from sklearn.ensemble import ExtraTreesClassifier
from sklearn.model_selection import cross_validate
from sklearn.svm import LinearSVC
from sklearn.feature_selection import mutual_info_classif

import matplotlib.pyplot as plt
import matplotlib as mpl
import matplotlib.ticker as ticker
from adjustText import adjust_text
mpl.rcParams['font.family'] = 'Avenir'
plt.rcParams['font.size'] = 12
plt.rcParams['axes.linewidth'] = 1
import pandas as pd
import seaborn as sns
import dask.dataframe as dd
import pickle
import numpy as np
import sys

def svm(X,y,kpe_gRNA):
	binary_var = 'type'

	clf_coef  = mutual_info_classif(X,y,random_state=0) 
	clf_coef = np.array(clf_coef)
	variable = X.columns.tolist()
	t10 = np.argsort(clf_coef)[-20:]
	top_var = list(map(lambda i: variable[i], t10[::-1]))

	kpe_gRNA[binary_var] = kpe_gRNA[binary_var].astype(int)

	num_in_grp = kpe_gRNA.groupby(binary_var)[top_var].count()
	kp_var_grp = kpe_gRNA.groupby(binary_var)[top_var].sum()

	print(num_in_grp)
	print(kp_var_grp)

	kp_var_grp /= num_in_grp
	print(kp_var_grp)
	
	kp_var_reform = kp_var_grp.T
	act_bool = kp_var_reform[0] <  kp_var_reform[1] 
	rep_bool = kp_var_reform[0] >  kp_var_reform[1] 

	fig, ax = plt.subplots(constrained_layout=True)
	np.savetxt('act.list',kp_var_reform[act_bool].index.tolist(),fmt='%s')
	np.savetxt('rep.list',kp_var_reform[rep_bool].index.tolist(),fmt='%s')

	g = sns.heatmap(kp_var_reform, xticklabels=True,cmap='RdBu_r',\
		vmin=0,annot=True,annot_kws={"size":10})
	g.set_xticklabels(g.get_xticklabels())
	plt.savefig('gRNA_burst.png',dpi=300,bbox_inches ='tight')

def tree(X,y, variable, kpe_gRNA):
	binary_var = 'type'
	clf = ExtraTreesClassifier(max_depth=3)
	clf = clf.fit(X,y)
	clf_coef = clf.feature_importances_
	score = clf.score(X,y)	
	score = round(score, 3)
	
	t10 = np.argsort(-clf_coef)[:20]
	top_var = list(map(lambda i: variable[i], t10))
	
	kpe_gRNA[binary_var] = kpe_gRNA[binary_var].astype(int)

	print(len(variable))
	print(kpe_gRNA['type'].value_counts())
	
	num_in_grp = kpe_gRNA.groupby(binary_var)[top_var].count().values
	kp_var_grp = kpe_gRNA.groupby(binary_var)[top_var].sum()
	kp_var_grp /= num_in_grp
	
	kp_var_reform = kp_var_grp.T
	act_bool = kp_var_reform[0] <  kp_var_reform[1] 
	rep_bool = kp_var_reform[0] >  kp_var_reform[1] 

	fig, ax = plt.subplots(constrained_layout=True)
	np.savetxt('act.list',kp_var_reform[act_bool].index.tolist(),fmt='%s')
	np.savetxt('rep.list',kp_var_reform[rep_bool].index.tolist(),fmt='%s')

	ax.set_title('score=%s'%score, loc='right')
	g = sns.heatmap(kp_var_reform, xticklabels=True,cmap='RdBu_r',\
		vmin=0,annot=True,annot_kws={"size":10})
	g.set_xticklabels(g.get_xticklabels())
	plt.savefig('gRNA_burst.png',dpi=300,bbox_inches ='tight')


def main():

	gRNA_target_in = sys.argv[1]
	kpe_in         = sys.argv[2] 
	gene_infor_in  = sys.argv[3]

	gRNA_target = pd.read_csv(gRNA_target_in, sep='\t|,',engine='python',\
		names=['chr','start','end','regulator','score','strand',\
			'gchr','gstart','gend','gRNA','infor'])
	fct_list = gRNA_target['regulator'].unique()

	gRNA_target['occ'] = 1
	gRNA_target_sub = gRNA_target[['gRNA','regulator','occ']]
	gRNA_target_sub.drop_duplicates(inplace=True)
	gRNA_target_reform = gRNA_target_sub.pivot(index='gRNA',columns='regulator',values='occ')	
	gRNA_target_reform.fillna(0, inplace=True)
	
	gene_infor = pd.read_csv(gene_infor_in,header=None,names=['gene','allele'])
	kpe = pd.read_csv(kpe_in, header=0)
	kpe['bs']  = kpe['ksyn']/kpe['koff']
	kpe['bf']  = kpe['kon'] * kpe['koff'] 
	kpe['bf'] /= kpe['kon'] + kpe['koff']
	kpe['n']  = kpe['kon']/(kpe['kon']+kpe['koff'])
	kpe['τ'] = 1/(kpe['kon']+kpe['koff'])

	for i, row in gene_infor.iterrows():
		gene   = row['gene']
		allele = row['allele']
		ikpe = kpe.loc[(kpe['gene']==gene) & (kpe['allele']==allele)]
		ikpe_ctrl = ikpe[ikpe['gRNA']=='NTC']

		for kp in ['bs','bf']:
			kpe.loc[i,kp] = kpe.loc[i,kp]/ikpe_ctrl[kp].values[0]
			kpe.loc[i,kp] = np.log2(kpe.loc[i,kp])
	
	bool_de = (abs(kpe['bs']) > 1) | (abs(kpe['bf']) > 1)
	bool_act = (kpe['bs'] < 0) & (kpe['bf'] > 0) & bool_de 
	bool_rep = (kpe['bs'] > 0) & (kpe['bf'] < 0) & bool_de
	kpe.loc[bool_rep,'type'] = 0
	kpe.loc[bool_act,'type'] = 1
	
	kpe.dropna(inplace=True)
	kpe_gRNA = pd.merge(kpe, gRNA_target_reform, left_on='gRNA', right_index=True)

	variable = gRNA_target_reform.columns.tolist() 
	X = kpe_gRNA[variable]
	y = kpe_gRNA['type']

	svm(X,y,kpe_gRNA)

if __name__ == "__main__":
	main()

