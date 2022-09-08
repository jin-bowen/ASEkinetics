import seaborn as sns
import matplotlib as mpl
import matplotlib.pyplot as plt
plt.rcParams['font.serif'] = ['Arial']
plt.rcParams['font.size'] = 8
plt.rcParams['axes.linewidth'] = 2

from scipy import sparse, stats
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
	yticks = [ round(x,2) for x in np.arange(0,1.1,0.1) ]
	ax.set_yticks(yticks)
	ax.set_yticklabels(yticks)

	ax.set_ylabel('ASE probability of Haplotype1')
	ax.set_xlabel('Chromosome')
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

def scatter(ase_eval, outfile):

	ase_eval = ase_eval.loc[ase_eval['qc']==1]
	ase_eval1 = ase_eval.loc[ase_eval['a_hat']<0.1].sample(1,random_state=1)
	ase_eval2 = ase_eval.loc[ase_eval['a_hat']>10].sample(1,random_state=1)

	xs = ase_eval['a_hat']
	ys = ase_eval['b_hat']


	x1_area = np.array([1e-3, 1])
	x2_area = np.array([1, 1e3])

	ax1 = plt.subplot2grid((2, 2), (0, 0), rowspan=2)
	ax1.fill_between(x1_area, y1=1e-3, y2=1,color='#d62728', alpha=0.4)
	ax1.fill_between(x2_area, y1=1, y2=1e3, color='#1f77b4', alpha=0.4)
	ax1.scatter(xs, ys, s=3, c='k')
	ax1.set_xscale('log')
	ax1.set_yscale('log')
	ax1.axvline(x=1,ls='--', lw=1, c='k')
	ax1.axhline(y=1,ls='--', lw=1, c='k')
	ax1.set_xlabel('α')
	ax1.set_ylabel('β')
	ax1.set_aspect('equal', adjustable='box')

	x = np.linspace(0, 1.0, 100)
	a = ase_eval1['a_hat']
	b = ase_eval1['b_hat']
	r = stats.beta.pdf(x,a, b)
	ax1.scatter(a,b,c='#d62728', s=24, marker='*')
	
	a2 = ase_eval2['a_hat']
	b2 = ase_eval2['b_hat']
	r2 = stats.beta.pdf(x,a2, b2)
	ax1.scatter(a2,b2,c='#1f77b4', s=24, marker='*')

	ax2 = plt.subplot2grid((2, 2), (1, 1), colspan=1)
	ax3 = plt.subplot2grid((2, 2), (0, 1), colspan=1, sharex=ax2)

	ax2.plot(x,r, c='#d62728', lw=2)
	ax3.plot(x,r2,c='#1f77b4', lw=2)
	ax2.set_xlabel('Haplotype1 allele ratio')
	ax2.get_yaxis().set_visible(False)
	ax3.get_yaxis().set_visible(False)
	plt.savefig(outfile+'.paramdist.pdf')
	plt.show()

def main():
	
	# number of cell sampled
	ase_profile  = sys.argv[1]
	ref_gene_tab = sys.argv[2]
	outfile      = sys.argv[3]

	ase_eval = pd.read_csv(ase_profile, header=0)
	ase_eval['a_qc'] = ase_eval.apply(lambda x: (x.Ia_lb > 0) & (x.Ia_ub > 0) ,axis=1)
	ase_eval['b_qc'] = ase_eval.apply(lambda x: (x.Ib_lb > 0) & (x.Ib_ub > 0),axis=1)
	ase_eval['qc']   = ase_eval.apply(lambda x: x.a_qc & x.b_qc,axis=1)
	ase_eval['qc']   = ase_eval.apply(lambda x: x.a_qc | x.b_qc,axis=1)

	ase_eval['m_prob_mean'] = ase_eval.apply(lambda x: x.a_hat/(x.a_hat+x.b_hat),axis=1)
	ase_eval['m_prob_var']  = ase_eval.apply(lambda x: 1.0/(x.a_hat+x.b_hat+1.0),axis=1) 

	ase_eval['m_prob_mean'] = ase_eval['m_prob_mean'].apply(lambda x: max(min(x,1),0))
	ase_eval.to_csv(outfile+'.stats',index=False)
#	figure(ase_eval,outfile)

	ref_gene = pd.read_csv(ref_gene_tab, header=0, sep='\t', names=['chr','start','end','gene','score','strand'])
	ase_eval_chr = pd.merge(ase_eval, ref_gene, on='gene')

	figure_by_chr(ase_eval_chr,outfile)
	scatter(ase_eval, outfile)

#	print(ase_eval_chr.loc[ase_eval_chr['chr']=='X'])
#	scatter(ase_eval_chr.loc[ase_eval_chr['chr']=='X'], outfile+'_chrX_')

if __name__ == "__main__":
	main()



