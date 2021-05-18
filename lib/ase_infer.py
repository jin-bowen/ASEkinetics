from joblib import delayed,Parallel
from estimation import *
import pandas as pd
import dask.dataframe as dd
import numpy as np
import sys

def allele_class(ase, outfile=None, save=False):

	df = ase.copy()
	df['alt_ratio'] = df['ub_alt']/(df['ub_ref'] + df['ub_alt'])	
	df['ref_ratio'] = df['ub_ref']/(df['ub_ref'] + df['ub_alt'])	
	df['all_ratio'] = df[['alt_ratio','ref_ratio']].min(axis=1)

	df_grp = df.groupby(['gene','pos','allele_ref','allele_alt']).agg(
		cb_count=pd.NamedAgg(column='cb',aggfunc='count'),
		ref_ratio_mean=pd.NamedAgg(column='ref_ratio',aggfunc='mean'),
		alt_ratio_list=pd.NamedAgg(column='alt_ratio',aggfunc=list),
		all_ratio_mean=pd.NamedAgg(column='all_ratio',aggfunc='mean'))

	df_grp_uniq = df_grp.reset_index().\
		sort_values(['cb_count'],ascending=False).drop_duplicates('gene')

	df_grp_uniq['grp'] = None
	df_grp_uniq.loc[(abs(df_grp_uniq['ref_ratio_mean']-0.5)==0.5) & \
			(df_grp_uniq['all_ratio_mean']==0), 'grp'] = 'fm'

	df_grp_uniq.loc[(abs(df_grp_uniq['ref_ratio_mean']-0.5)<0.5) & \
			(df_grp_uniq['all_ratio_mean']==0), 'grp'] = 'rm'

	df_grp_uniq.loc[(abs(df_grp_uniq['ref_ratio_mean']-0.5)>=0.4) & \
			(df_grp_uniq['all_ratio_mean']!=0) & \
			(df_grp_uniq['all_ratio_mean']<=0.1), 'grp'] = 'ifb'

	df_grp_uniq.loc[(abs(df_grp_uniq['ref_ratio_mean']-0.5)<0.4) & \
			(df_grp_uniq['all_ratio_mean']!=0) & \
			(df_grp_uniq['all_ratio_mean']<=0.1), 'grp'] = 'irb'

	df_grp_uniq.loc[(abs(df_grp_uniq['ref_ratio_mean']-0.5)<0.5) & \
			(df_grp_uniq['all_ratio_mean']>0.1), 'grp'] = 'bb'
	if save:
		cols  = ['gene','pos','allele_ref','allele_alt']
		cols += ['cb_count','ref_ratio_mean','all_ratio_mean','grp']
		df_grp_uniq[cols].to_csv(outfile,index=False,float_format='%.5f')
	return df_grp_uniq

def allele_infer(umi, ase, ase_class_raw, outfile, min_cb=10):
	
	cols = ['cb','gene','pos','allele_ref','ub_ref','allele_alt','ub_alt','umi']

	ase_tab_intersect = pd.merge(umi,ase,on=['cb','gene'])
	ase_tab_intersect[cols].to_csv(outfile+'.org',index=False)

	ase_tab_full = pd.merge(umi,ase,how='left',on=['cb','gene'])
	ase_tab      = ase_tab_full[ase_tab_full['pos'].isna()]

	ase_class = ase_class_raw[ase_class_raw['cb_count'] > min_cb]
	ase_tab_class = pd.merge(ase_tab, ase_class, on='gene',suffixes=('_tab', ''))

	f = open(outfile+'.infer', 'a')
	f.write(','.join(cols) )
	f.write('\n')
	for i, row in ase_tab_class.iterrows():

		dst = row['alt_ratio_list']	
		hist, bin_edges = np.histogram(list(dst), bins=10, range=(0,1))
		prob = hist/hist.sum()
		m_ratio_bin = bin_edges[bin_edges < 1] + 0.01
	
		infer_alt_ratio = np.random.choice(m_ratio_bin,p=prob)	
		row['ub_ref'] = 1 - infer_alt_ratio
		row['ub_alt'] = infer_alt_ratio		

		row[cols].to_frame().T.to_csv(f,header=False,\
			index=False,float_format='%.5f')
	f.close()

def main():

	umi_file = sys.argv[1]
	ase_file = sys.argv[2]
	cb_file  = sys.argv[3]
	name     = sys.argv[4]
	outdir   = sys.argv[5]

	umi_raw = pd.read_csv(umi_file,header=0,\
		dtype={'cb':str,'gene':str,'umi':float})

	ase_raw = pd.read_csv(ase_file,usecols=range(7),
		dtype={'allele_ref':str,'ub_ref':float,\
			'allele_alt':str,'ub_alt':float})
	ase_raw['ub_alt'] = ase_raw['ub_alt'].fillna(0)	
	ase_raw['ub_ref'] = ase_raw['ub_ref'].fillna(0)	

	# filtered cells 
	cb_raw = pd.read_csv(cb_file,dtype='str',header=None,names=['cb'])
	umi = pd.merge(umi_raw, cb_raw, on='cb')
	ase = pd.merge(ase_raw, cb_raw, on='cb')

	# infer class
	outfile = '%s/%s.ase.class'%(outdir,name)
	ase_class = allele_class(ase,save=True,outfile=outfile)

	# infer ase
	ase_infer_outfile = '%s/%s.ase'%(outdir,name)
	allele_infer(umi,ase,ase_class,ase_infer_outfile)

if __name__ == "__main__":
	main()

