import seaborn as sns
import matplotlib as mpl
import matplotlib.pyplot as plt
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = ['Arial']
plt.rcParams['font.size'] = 10
plt.rcParams['axes.linewidth'] = 2
mpl.use('Agg')

import pandas as pd
import numpy as np
import scipy as sp
import sys

def figure_by_chr(ase_eval,outfile):

	fig, ax = plt.subplots(figsize=(10,4))
	ase_eval_keep = ase_eval[ase_eval['qc']==True]
	ase_eval_keep['m_prob_mean'] = ase_eval_keep['m_prob_mean'].apply(lambda x: 1 if x >1 else x)
	order = [str(x) for x  in range(1,23) ]
	order +=  ['X']
	ax.xaxis.grid(True, linestyle='--')
	ax.yaxis.grid(False)
	ax.axhline(y=0.5, c='k', ls='--')
	sns.violinplot(x="chr", y="m_prob_mean",data=ase_eval_keep,order=order,\
		height=6, aspect=0.2, linewidth=1,ax=ax)

	ax.set_ylabel('ASE probability of maternal allele')
	ax.set_xlabel('chromosome')
	plt.savefig(outfile+'.bychrom.pdf')

def figure(ase_eval,outfile):

	cols=['nall','percent','m_prob_mean','m_prob_var']	
	fig, ax = plt.subplots(ncols=len(cols),constrained_layout=True,\
			sharex=True, figsize=(len(cols)*2,2))
	for i, icol  in enumerate(cols):
		strata = ase_eval['qc']==True
		ax[i].violinplot([ase_eval.loc[~strata,icol], ase_eval.loc[strata,icol]], [0,1],\
			points=60, widths=0.7, showmeans=True,showextrema=True,bw_method=0.5)
		ax[i].set_title(icol)
		ax[i].set_ylabel('value')
		ax[i].set_xticks([0,1])
		ax[i].set_xticklabels(['not_pass','pass'])
	plt.savefig(outfile+'.stat.pdf')

def main():
	
	# number of cell sampled
	ase_profile  = sys.argv[1]
	ref_gene_tab = sys.argv[2]
	outfile      = sys.argv[3]

	ase_eval = pd.read_csv(ase_profile, header=None, \
		names=['gene','nref','nall','percent','a_hat','Ia_lb','Ia_ub','b_hat','Ib_lb','Ib_ub'])

	ase_eval['a_qc'] = ase_eval.apply(lambda x: (x.Ia_lb > 0) & (x.Ia_ub > 0) ,axis=1)
	ase_eval['b_qc'] = ase_eval.apply(lambda x: (x.Ib_lb > 0) & (x.Ib_ub > 0),axis=1)
	ase_eval['qc']   = ase_eval.apply(lambda x: x.a_qc & x.b_qc,axis=1)

	ase_eval['m_prob_mean'] = ase_eval.apply(lambda x: x.a_hat/(x.a_hat+x.b_hat),axis=1)
	ase_eval['m_prob_var'] = ase_eval.apply(lambda x: 1.0/(x.a_hat+x.b_hat+1.0),axis=1) 

	print(ase_eval)
	print(ase_eval['qc'].sum(), ase_eval['qc'].count())
	figure(ase_eval,outfile)

	ref_gene = pd.read_csv(ref_gene_tab, header=0, sep='\t', names=['chr','start','end','gene','score','strand'])
	ase_eval_chr = pd.merge(ase_eval, ref_gene, on='gene')
	figure_by_chr(ase_eval_chr,outfile)
			

if __name__ == "__main__":
	main()



