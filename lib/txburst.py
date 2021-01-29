from joblib import delayed,Parallel
from estimation import *
import pandas as pd
import dask.dataframe as dd
import numpy as np
import sys

def allele_class(ase, outfile=None, save=False):

	df = ase.copy().compute()
	df['alt_ratio'] = df['ub_alt']/(df['ub_ref'] + df['ub_alt'])	
	df['ref_ratio'] = df['ub_ref']/(df['ub_ref'] + df['ub_alt'])	
	df['all_ratio'] = df[['alt_ratio','ref_ratio']].min(axis=1)

	df_grp = df.groupby(['gene','pos','allele_ref','allele_alt']).agg(
		cb_count=pd.NamedAgg(column='cb',aggfunc='count'),
		ref_ratio_mean=pd.NamedAgg(column='ref_ratio',aggfunc='mean'),
		alt_ratio_list=pd.NamedAgg(column='alt_ratio',aggfunc=list),
		all_ratio_mean=pd.NamedAgg(column='all_ratio',aggfunc='mean'))

	df_grp_uniq = df_grp.reset_index().\
		sort_values(['cb_count'],ascending=False).\
		drop_duplicates('gene')

	df_grp_uniq['grp'] = 0
	df_grp_uniq.loc[(abs(df_grp_uniq['ref_ratio_mean']-0.5)==0.5) & \
			(df_grp_uniq['all_ratio_mean']==0), 'grp'] = 1

	df_grp_uniq.loc[(abs(df_grp_uniq['ref_ratio_mean']-0.5)<0.5) & \
			(df_grp_uniq['all_ratio_mean']==0), 'grp'] = 2

	df_grp_uniq.loc[(abs(df_grp_uniq['ref_ratio_mean']-0.5)>=0.4) & \
			(df_grp_uniq['all_ratio_mean']!=0) & \
			(df_grp_uniq['all_ratio_mean']<=0.1), 'grp'] = 3

	df_grp_uniq.loc[(abs(df_grp_uniq['ref_ratio_mean']-0.5)<0.4) & \
			(df_grp_uniq['all_ratio_mean']!=0) & \
			(df_grp_uniq['all_ratio_mean']<=0.1), 'grp'] = 4

	df_grp_uniq.loc[(abs(df_grp_uniq['ref_ratio_mean']-0.5)<0.5) & \
			(df_grp_uniq['all_ratio_mean']>0.1), 'grp'] = 5
	if save:
		cols  = ['gene','pos','allele_ref','allele_alt']
		cols += ['cb_count','ref_ratio_mean','all_ratio_mean','grp']
		df_grp_uniq[cols].to_csv(outfile,index=False,float_format='%.5f')
	return df_grp_uniq

def allele_infer(umi, ase, ase_class, save=False, outfile=None):
	
	# umi : dask datafram
	# ase : dask datafram
	# ase_class: pandas dataframe
	ase_class.set_index('gene', inplace=True)
	columns = ['cb','gene','pos','allele_ref','ub_ref','allele_alt','ub_alt']

	ase_tab = dd.merge(umi,ase,how='left',on=['cb','gene'])
	ase_tab_gene = ase_tab['gene'].unique().compute()
	ase_tab_intersect = dd.merge(umi,ase,on=['cb','gene'])

	ase_tab_grp = ase_tab.groupby('gene')
	
	gene_class_tab = set(ase_class.index.tolist())
	gene_class_tab = gene_class_tab.intersection(ase_tab_gene)
	gene_class_tab = list(gene_class_tab)
	
	for gene in gene_class_tab:

		ase_tab_igrp = ase_tab_grp.get_group(gene).compute()
	
		bool_row = ase_tab_igrp['pos'].isna()
		if bool_row.sum() == 0 :continue
		if gene not in ase_class.index.tolist(): continue
		if ase_class.loc[gene, 'cb_count'] < 10: continue

		ase_tab_igrp.loc[bool_row,'pos'] 	   = ase_class.loc[gene,'pos']
		ase_tab_igrp.loc[bool_row,'allele_ref'] = ase_class.loc[gene,'allele_ref']
		ase_tab_igrp.loc[bool_row,'allele_alt'] = ase_class.loc[gene,'allele_alt']

		dst = ase_class.loc[gene,'alt_ratio_list']
	
		n_infer = np.sum(bool_row)	
		hist, bin_edges = np.histogram(list(dst), bins=10, range=(0,1))
		prob = hist/hist.sum()
		m_ratio_bin = bin_edges[bin_edges < 1] + 0.05

		infer_alt_ratio = np.random.choice(m_ratio_bin,size=n_infer,p=prob)
	
		ase_tab_igrp.loc[bool_row,'ub_ref'] = 1 - infer_alt_ratio
		ase_tab_igrp.loc[bool_row,'ub_alt'] = infer_alt_ratio		

		ase_tab_intersect = dd.concat([ase_tab_intersect,ase_tab_igrp.loc[bool_row]])
	if save:
		ase_tab_intersect.to_csv(outfile,index=False, single_file = True)
	
	return ase_tab_intersect

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

def estimate_by_row(row):
	
	gene = row['gene']
	allele = row['allele']
	mrna = row['list']
	if len(mrna) < 50: 
		return [gene,allele] + [None, None, None, None, None]

	obj1  = mRNAkinetics(mrna)
	obj1.MaximumLikelihood()
	mrna_kpe = obj1.get_estimate()

	return [gene,allele] + mrna_kpe + [len(mrna),np.mean(mrna)]
		
def main():

	umi_file = sys.argv[1]
	ase_file = sys.argv[2]
	cb_file  = sys.argv[3]
	name     = sys.argv[4]
	outdir   = sys.argv[5]

	umi_raw = dd.read_csv(umi_file,header=0,\
		dtype={'cb':str,'gene':str,'umi':float}).set_index('cb')

	ase_raw = dd.read_csv(ase_file,usecols=range(7),
		dtype={'allele_ref':str,'ub_ref':float,\
			'allele_alt':str,'ub_alt':float}).set_index('cb')
	ase_raw['ub_alt'] = ase_raw['ub_alt'].fillna(0)	
	ase_raw['ub_ref'] = ase_raw['ub_ref'].fillna(0)	

	# filtered cells 
	cb_raw = np.loadtxt(cb_file,dtype='str')
	cb_umi = set(cb_raw).intersection(umi_raw.index)
	umi = umi_raw.loc[list(cb_umi)]
	umi = umi.reset_index()

	cb_ase = set(cb_raw).intersection(ase_raw.index)
	ase = ase_raw.loc[list(cb_ase)]
	ase = ase.reset_index()	

	# infer class
	outfile = '%s/%s.ase.class'%(outdir,name)
	ase_class = allele_class(ase,save=True,outfile=outfile)

	# infer ase
	outfile = '%s/%s.ase.infer'%(outdir,name)
	inferred_allele_df = allele_infer(umi,ase,ase_class,save=True,outfile=outfile)

	# txb estimation
	df= process(inferred_allele_df.compute())

	data = Parallel(n_jobs=10)(delayed(estimate_by_row)(row) for i,row in df.iterrows())
	out = pd.DataFrame(data,columns=['gene','allele','kon','koff','ksyn','n','mean'])

	outfile = '%s/%s.mle.infer'%(outdir,name)
	out.dropna().to_csv(outfile, index=False,float_format='%.5f')

if __name__ == "__main__":
	main()

