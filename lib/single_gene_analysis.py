import matplotlib.pyplot as plt
import matplotlib as mpl
import matplotlib.ticker as ticker
from adjustText import adjust_text
from scipy.stats import chi2
mpl.use('Agg')
mpl.rcParams['font.family'] = 'Avenir'
plt.rcParams['font.size'] = 6
plt.rcParams['axes.linewidth'] = 1
from est import pb_est
from evaluation import simLikelihoodRatioTest, GoF
import pandas as pd
import seaborn as sns
import dask.dataframe as dd
import pickle
import numpy as np
import scipy as sp
import sys

def plot_coordinate(gRNA_target,gene_coordinate,gRNA_coordinate,process_df_filter,kpe,out):

	gRNA_target_grp = gRNA_target.groupby('gRNA')
	ngRNA = gRNA_coordinate['gRNA'].nunique()
	ngRNA_filter = gRNA_target_grp.ngroups

	num_reg = gRNA_target_grp['regulator'].nunique()
	num_reg = np.max(num_reg.values)
	
	ratio = np.ceil(num_reg/16 + 1)

	fig, ax = plt.subplots(nrows=4,gridspec_kw={'height_ratios': [1.5,ratio,1,2]},
				constrained_layout=True,figsize=(4,6))

	color_list = sns.color_palette("husl", ngRNA)
	color_dict = dict(zip(gRNA_coordinate['gRNA'].values, color_list))
	color_dict['gNTC'] = 'gray'

	yticks = []
	ytickslabel = []
	# plot gene
	gene = gene_coordinate['gene']
	gene_x = gene_coordinate[['start','end']].values
	gene_y = [0,0]
	yticks += [0]
	ytickslabel += [gene]

	if gene_coordinate['strand'] == '+':
		ax[0].plot(gene_x,gene_y,'>-k',linewidth=4, markevery=[0])
	else:
		ax[0].plot(gene_x,gene_y,'<-k',linewidth=4, markevery=[-1])

	texts = []
	for irow, row in gRNA_coordinate.iterrows():
		
		igRNA = row['gRNA']
		icolor = color_dict[igRNA]
		igRNA_x = row[['start','end']].values
		igRNA_y = irow + 1
		igRNA_y_coord = [igRNA_y,igRNA_y]
		if abs(row['start'] - row['end']) < 10e4:  
			ax[0].scatter(igRNA_x[0],igRNA_y,marker='|',c=icolor)
		else:
			ax[0].plot(igRNA_x,igRNA_y_coord,ls='-',linewidth=4, c=icolor)
	
		yticks += [igRNA_y]
		ytickslabel += [igRNA]
	
		if igRNA not in gRNA_target_grp.groups.keys(): continue
		grp = gRNA_target_grp.get_group(igRNA).drop_duplicates(subset=['regulator'], keep='last')

		j = 0
		ax[1].plot([igRNA_y-0.25, igRNA_y+0.25], [0,0],ls='-',c=icolor,linewidth=4)
		text_interval = ax[1].get_xticks()

		gRNA_len = row['start'] - row['end']
		gRNA_len /= 0.5

		if len(grp) > 16: fs = 2
		else: fs = 2
#		else: fs = 2

		#within each group
		for irow,record in grp.iterrows():
			regulator = record['regulator']
			reg_x_abs = igRNA_y - 0.25 + ( - record[['start','end']].values + row['start'] )/ gRNA_len 

			reg_x1 = np.max([reg_x_abs[0], igRNA_y-0.25])
			reg_x2 = np.min([reg_x_abs[1], igRNA_y+0.25])

			reg_x_coord = [reg_x1,reg_x2]

			reg_y = j+1
			reg_y_coord = [reg_y,reg_y]

			ax[1].plot(reg_x_coord, reg_y_coord,'k-',linewidth=0.5)
			ax[1].text(igRNA_y+0.25, reg_y, regulator, fontsize=fs)
			j += 1
		
		ax[1].set_yticks([])
		ax[1].spines['right'].set_visible(False)
		ax[1].spines['top'].set_visible(False)
		ax[1].spines['left'].set_visible(False)

	ax[0].set_yticklabels(ytickslabel, fontsize=6)
	ax[0].spines['right'].set_visible(False)
	ax[0].spines['top'].set_visible(False)
	ax[0].spines['left'].set_visible(False)

	ax[0].set_yticks(yticks) 
	ax[0].set_yticklabels(ytickslabel, fontsize=6)
	ax[0].set_ylim([-2,ngRNA+1])
	ax[0].set_xlabel('genomic coordinate')

	g = sns.violinplot(y="list", x="gRNA", palette=color_dict, \
			ax=ax[2], data=process_df_filter,linewidth=0.01, \
			order = ['gNTC']+ytickslabel[1:])
	sns.despine(ax=ax[2], top=False, right=False, left=False)
	ax[1].sharex(ax[2])
	ax[1].axes.xaxis.set_visible(False)
	ax[1].spines['bottom'].set_visible(False)

	ax[2].spines['right'].set_visible(False)
	ax[2].spines['top'].set_visible(False)
	ax[2].spines['left'].set_visible(False)

	ax[2].set(ylabel='UMI',xlabel='')
	g.set_xticklabels(['gNTC']+ytickslabel[1:], fontsize=6, rotation=30)
	g1 = sns.barplot(x='variable',y='value', hue='gRNA',data=kpe,ax=ax[3],
			palette=color_dict, ci=None, hue_order=['gNTC']+ytickslabel[1:])
	ax[3].set(xlabel='',ylabel='log2(fold change)')
	ax[3].axhline(y=0, c='k',linewidth=0.5, ls='--', label='0')
	ax[3].get_legend().remove()
	ax[3].spines['top'].set_visible(False)
	ax[3].spines['right'].set_visible(False)
	ax[3].spines['left'].set_visible(False)
	ax[3].set_yscale('symlog')
	ax[3].yaxis.set_major_locator(ticker.MultipleLocator())
	ax[3].yaxis.set_minor_locator(ticker.MultipleLocator())
	plt.setp(ax[1].get_xticklabels(), visible=False)
	plt.savefig(out+'_coord.png',dpi=300,bbox_inches ='tight')
	plt.show()

def main():

	gene   = sys.argv[1]
	allele = sys.argv[2]
	kpe_in = sys.argv[3] 

	ase_dir   = sys.argv[4]

	tss_in    = sys.argv[5]
	gRNA_in   = sys.argv[6]
	gRNA_target_in = sys.argv[7]

	tss_df = pd.read_csv(tss_in, names=['chr','start','end','gene','score','strand'], \
		sep='\t|,',engine='python')
	gRNA_target = pd.read_csv(gRNA_target_in,sep='\t',
		names=['gchr','gstart','gend','gRNA','chr','start','end','regulator'])

	gRNA_df = pd.read_csv(gRNA_in, sep='\t',header=0, names=['chr','start','end','gRNA'])

	kpe_raw = pd.read_csv(kpe_in, header=0, index_col=0)
	kpe_raw[['gene','allele','gRNA']] = kpe_raw.index.to_series().str.split('-', n=2, expand=True)
	kpe = kpe_raw.loc[(kpe_raw['gene']==gene) & (kpe_raw['allele']==allele)]

	process_df = pd.DataFrame(columns=['gRNA','list'])
	for i,gRNA in enumerate(kpe['gRNA'].unique()): 
		ase_in = ase_dir + '/' + gene + '_' + gRNA + '.ase.reform'
		ase = pd.read_csv(ase_in, header=0, index_col=0)
		index = gene + '-' + allele + '-' + gRNA
		ase_numeric = ase.loc[index].values.reshape(-1)

		process_df.loc[i,'gRNA'] = gRNA
		process_df.loc[i,'list'] = ase_numeric

	kpe['bs']  = kpe['ksyn']/kpe['koff']
	kpe['bf']  = kpe['kon'] * kpe['koff'] 
	kpe['bf'] /= kpe['kon'] + kpe['koff']
	kpe['n']  = kpe['kon']/(kpe['kon']+kpe['koff'])
	kpe['τ'] = 1/(kpe['kon']+kpe['koff'])

	# DE test
	kpe_ctrl = kpe[kpe['gRNA']=='gNTC']
	ctrl_kp = kpe_ctrl[['kon','koff','ksyn']].values[0]
	ctrl_reads = process_df.loc[process_df['gRNA']=='gNTC','list'].values[0]
	for gRNA in kpe['gRNA'].tolist():
		kp_bool = kpe['gRNA']==gRNA
		gRNA_kp = kpe.loc[kp_bool,['kon','koff','ksyn']].values[0]
		read_bool = process_df['gRNA']==gRNA
		gRNA_reads = process_df.loc[read_bool,'list'].values[0]
		lr_pval = simLikelihoodRatioTest(gRNA_kp, gRNA_reads, ctrl_kp, ctrl_reads)		
		kpe.loc[kp_bool,'lr_pval'] = lr_pval
		q, chisq_pval = GoF(gRNA_reads,ctrl_reads)
		kpe.loc[kp_bool,'chisq_pval'] = chisq_pval

	# change abosulate value to fold
	for kp in ['kon','koff','ksyn','bs','bf','n','τ','mean']:
		kpe[kp] = kpe[kp]/kpe_ctrl[kp].values[0]
		kpe[kp] = kpe[kp].transform(np.log2)

	fold_thres = np.log2(2)
#	# check if the gNTC and NTC are different
#	kpe_gNTC = kpe[kpe['gRNA']=='gNTC'] 
#	bool_change = (abs(kpe_gNTC['kon']) > fold_thres) | \
#		(abs(kpe_gNTC['koff']) > fold_thres) | \
#		(abs(kpe_gNTC['ksyn']) > fold_thres)
#
#	if bool_change.values[0]: 
#		print(gene, allele, 'gNTC is not consistent with NTC')
#		return 0
#
#	# filter by kpe fold change
	bool_filter_fc = (abs(kpe['kon']) > fold_thres) | \
		(abs(kpe['koff']) > fold_thres) | \
		(abs(kpe['ksyn']) > fold_thres)
#	gRNA_filter = kpe.loc[bool_filter,'gRNA'].tolist()
#	# filter by burst kp fold change
#	bool_filter = (abs(kpe['bs']) > fold_thres) | \
#		(abs(kpe['bf']) > fold_thres) 
#	gRNA_filter = kpe.loc[bool_filter,'gRNA'].tolist()
	
	bt = 0.01
	bool_filter_bt = (kpe['lr_pval'] < bt) | (kpe['chisq_pval'] < bt)
	bool_filter = bool_filter_fc & bool_filter_bt
	gRNA_filter = kpe.loc[bool_filter,'gRNA'].tolist()
	gRNA_filter = set(gRNA_filter) - set(['NTC'])
	gRNA_filter = list(gRNA_filter)
	if len(gRNA_filter) < 2: return 0

	kpe_filter = kpe[bool_filter]
	kpe_df_reform = pd.melt(kpe_filter, id_vars=['gene','allele','gRNA'],\
			value_vars=['kon','koff','ksyn','bs','bf','n','τ','mean'])

	# create violin plot format
	process_df_filter = process_df[process_df['gRNA'].isin(gRNA_filter+['gNTC'])]
	process_df_filter = process_df_filter.explode('list',ignore_index=True)
	process_df_filter = process_df_filter.explode('list',ignore_index=True)
	process_df_filter['list'] = process_df_filter['list'].astype(np.float32)

	gene_coordinate = tss_df[tss_df['gene']==gene].squeeze()
	gRNA_coordinate = gRNA_df[gRNA_df['gRNA'].isin(gRNA_filter)]
	gRNA_target_filter = gRNA_target[gRNA_target['gRNA'].isin(gRNA_filter)]

	gRNA_coordinate.reset_index(inplace=True, drop=True)
	gRNA_coordinate.drop_duplicates(inplace=True)

	plot_coordinate(gRNA_target_filter, gene_coordinate, gRNA_coordinate, process_df_filter, kpe_df_reform, gene+'_'+allele)

if __name__ == "__main__":
	main()

