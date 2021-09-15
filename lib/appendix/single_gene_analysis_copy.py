import matplotlib.pyplot as plt
import matplotlib as mpl
import matplotlib.ticker as ticker
from adjustText import adjust_text
mpl.use('Agg')
mpl.rcParams['font.family'] = 'Avenir'
plt.rcParams['font.size'] = 6
plt.rcParams['axes.linewidth'] = 1
import pandas as pd
import seaborn as sns
import dask.dataframe as dd
import pickle
import numpy as np
import sys

def process(df):

	df['ref_infer'] = np.ceil(df['ub_ref'] * df['umi']/ (df['ub_ref'] + df['ub_alt']))
	df['alt_infer'] = df['umi'] - df['ref_infer']

	df['allele_pos'] = df[['gene','pos','allele_ref','allele_alt']].agg('_'.join, axis=1)
	df_grp = df.groupby('allele_pos')['ref_infer','alt_infer']

	out = pd.DataFrame(columns=['gene','allele','list'])
	for key, group in df_grp:
		key_list = key.split('_')
		igene = key_list[0]
		ref_allele_id = key_list[1] + '_' + key_list[2]
		alt_allele_id = key_list[1] + '_' + key_list[3]

		ref_val = group['ref_infer'].values
		alt_val = group['alt_infer'].values

		ref_nonzero = ref_val[np.where(ref_val!=0)]
		alt_nonzero = alt_val[np.where(alt_val!=0)]

		out.loc[len(out.index)] = [igene, ref_allele_id, ref_nonzero]
		out.loc[len(out.index)] = [igene, alt_allele_id, alt_nonzero]
	return out

def plot_coordinate(gRNA_target,gene_coordinate,gRNA_coordinate,process_df_filter,kpe,out):

	gRNA_target_grp = gRNA_target.groupby('gRNA')
	ngRNA = gRNA_coordinate['gRNA'].nunique()
	ngRNA_filter = gRNA_target_grp.ngroups

	num_reg = gRNA_target_grp['regulator'].nunique()
	num_reg = np.max(num_reg.values)
	
	ratio = np.ceil(num_reg/16 + 1)

	fig, ax = plt.subplots(nrows=4,gridspec_kw={'height_ratios': [1,ratio,1,2]},
				constrained_layout=True,figsize=(4,6))

	color_list = sns.color_palette("husl", ngRNA)
	color_dict = dict(zip(gRNA_coordinate['gRNA'].values, color_list))
	color_dict['NTC'] = 'gray'

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
		grp = gRNA_target_grp.get_group(igRNA)

		j = 0
		ax[1].plot([igRNA_y-0.25, igRNA_y+0.25], [0,0],ls='-',c=icolor,linewidth=4)
		text_interval = ax[1].get_xticks()

		gRNA_len = row['start'] - row['end']
		gRNA_len /= 0.5

		if len(grp) > 16: fs = 2
		else: fs = 4

		#within each group
		for irow,record in grp.iterrows():
			regulator = record['regulator']
			reg_x = igRNA_y - 0.25 + ( - record[['start','end']].values + row['start'] )/ gRNA_len 
			reg_y = j+1
			reg_y_coord = [reg_y,reg_y]

			ax[1].plot(reg_x, reg_y_coord,'k-',linewidth=0.5)
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
			order = ['NTC']+ytickslabel[1:])
	sns.despine(ax=ax[2], top=False, right=False, left=False)
	ax[1].sharex(ax[2])
	ax[1].axes.xaxis.set_visible(False)
	ax[1].spines['bottom'].set_visible(False)

	ax[2].spines['right'].set_visible(False)
	ax[2].spines['top'].set_visible(False)
	ax[2].spines['left'].set_visible(False)

	ax[2].set(ylabel='UMI',xlabel='')
	g.set_xticklabels(['NTC']+ytickslabel[1:], fontsize=6, rotation=15)
	g1 = sns.barplot(x='variable',y='value', hue='gRNA',data=kpe,ax=ax[3],
			palette=color_dict, ci=None, hue_order=['NTC']+ytickslabel[1:])
	ax[3].set(xlabel='',ylabel='fold change')
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

	ase_infer = sys.argv[1]
	cb_w_gRNA = sys.argv[2]
	gRNA_in   = sys.argv[3]

	tss_in         = sys.argv[4]
	gRNA_target_in = sys.argv[5]

	gene   = sys.argv[6]
	allele = sys.argv[7]

	de_in  = sys.argv[8]
	kpe_in = sys.argv[9] 

	tss_df = pd.read_csv(tss_in, names=['chr','start','end','gene','score','strand'], \
		sep='\t|,',engine='python')
	gRNA_target = pd.read_csv(gRNA_target_in, sep='\t|,',engine='python',\
		names=['chr','start','end','regulator','score','strand',\
			'gchr','gstart','gend','gRNA','infor'])
	gRNA_df = pd.read_csv(gRNA_in, sep='\s+|\t|,',engine='python',\
		usecols=range(4),names=['chr','start','end','gRNA'])
	gRNA_df.drop_duplicates(inplace=True)

	inferred_allele_df = dd.read_csv(ase_infer,header=0)
	select = inferred_allele_df['gene']==gene 
	inferred_allele_df_subset = inferred_allele_df.loc[select,:].compute()

	cb_gRNA_df = pd.read_csv(cb_w_gRNA, sep=' ',header=None, names=['cb_w_exp','gRNA','gRNA-grp'])
	bool_var = cb_gRNA_df['gRNA-grp']=='NTC'
	cb_gRNA_df.loc[ bool_var,'grp'] = 'NTC'
	cb_gRNA_df.loc[~bool_var,'grp'] = cb_gRNA_df.loc[~bool_var,'gRNA']
	cb_gRNA_df.drop(['gRNA', 'gRNA-grp'], axis=1, inplace=True)
	cb_gRNA_df.drop_duplicates(inplace=True)
	#select by gene and allele
	df = pd.merge(inferred_allele_df_subset,cb_gRNA_df,left_on='cb',right_on='cb_w_exp')
	# extract gRNA and NTC group
	df_grp = df.groupby('grp')

	de_raw  = dd.read_csv(de_in, header=0)
	kpe_raw = dd.read_csv(kpe_in, header=0)

	de  = de_raw.loc[(de_raw['gene']==gene) & (de_raw['allele']==allele)].compute()
	kpe = kpe_raw.loc[(kpe_raw['gene']==gene) & (kpe_raw['allele']==allele)].compute()

	process_df = pd.DataFrame(columns=['gene','gRNA','allele','list'])
	for target, igrp in df_grp:
		target_grp = df_grp.get_group(target) 
		process_df_temp =  process(target_grp)
		process_df_temp = process_df_temp[process_df_temp['allele']==allele]
		process_df_temp['gRNA'] = target
		process_df = process_df.append(process_df_temp,ignore_index=True)

	sig_filter = de[['ks','mwu','ad']] < 0.05
	sig_filter_bool = sig_filter.any(axis='columns')	
	gRNA_filter = de.loc[sig_filter_bool,'gRNA'].tolist()

	if len(gRNA_filter) < 1: return 0

	# create violin plot format
	process_df_filter = process_df[process_df['gRNA'].isin(gRNA_filter+['NTC'])]
	process_df_filter = process_df_filter.explode('list',ignore_index=True)
	process_df_filter["list"] = pd.to_numeric(process_df_filter["list"])

	#modify kpe profile
	kpe_filter = kpe[kpe['gRNA'].isin(gRNA_filter+['NTC'])]
	if len(kpe_filter) < 2: return 0
	
	kpe_filter['bs']  = kpe_filter['ksyn']/kpe_filter['koff']
	kpe_filter['bf']  = kpe_filter['kon'] * kpe_filter['koff'] 
	kpe_filter['bf'] /= kpe_filter['kon'] + kpe_filter['koff']
	kpe_filter['mu']  = kpe_filter['kon']/(kpe_filter['kon']+kpe_filter['koff'])
	kpe_filter['tau'] = 1/(kpe_filter['kon']+kpe_filter['koff'])
	kpe_ctrl = kpe_filter[kpe_filter['gRNA']=='NTC']
	for kp in ['kon','koff','ksyn','bs','bf','mu','tau','mean']:
		kpe_filter[kp] = kpe_filter[kp]/kpe_ctrl[kp].values[0] - 1

	kpe_df_reform = pd.melt(kpe_filter, id_vars=['gene','allele','gRNA'],\
			value_vars=['kon','koff','ksyn','bs','bf','mu','tau','mean'])

	
	gene_coordinate = tss_df[tss_df['gene']==gene].squeeze()
	gRNA_coordinate = gRNA_df[gRNA_df['gRNA'].isin(gRNA_filter)]
	gRNA_target_filter = gRNA_target[gRNA_target['gRNA'].isin(gRNA_filter)]
	gRNA_coordinate.reset_index(inplace=True, drop=True)
	gRNA_coordinate.drop_duplicates(inplace=True)
	plot_coordinate(gRNA_target_filter, gene_coordinate, gRNA_coordinate,process_df_filter, kpe_df_reform,gene+'_'+allele)

if __name__ == "__main__":
	main()

