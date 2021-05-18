from joblib import delayed,Parallel
from estimation import *
from scipy.stats import ks_2samp, mannwhitneyu, anderson_ksamp
import pandas as pd
import dask.dataframe as dd
import numpy as np
import sys

def DGE_test(process_df, process_NTC_df):

	full_tab = pd.merge(process_df,process_NTC_df,how='left',on=['gene','allele'],suffixes=('','_NTC'))
	for i, record in full_tab.iterrows():
		if record['list_NTC'] is None:
			full_tab.loc[i,'ks'] = None
			full_tab.loc[i,'mwu'] = None
			full_tab.loc[i,'ad'] = None
		else:
			rvs1 = record['list']
			rvs2 = record['list_NTC']
			rvs  = record[['list','list_NTC']]

			ks_stat, ks_pval   = ks_2samp(rvs1,rvs2)
			mwu_stat, mwu_pval = mannwhitneyu(rvs1, rvs2)
			stat, critical_stat, ad_level = anderson_ksamp(rvs)
			full_tab.loc[i,'ks'] = ks_pval
			full_tab.loc[i,'mwu'] = mwu_pval
			full_tab.loc[i,'ad'] = ad_level
	return full_tab

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

def main():

	ase_infer = sys.argv[1]
	cb_w_gRNA = sys.argv[2]

	gene     = sys.argv[3]
	outdir   = sys.argv[4]

	inferred_allele_df = dd.read_csv(ase_infer,header=None,\
		names=['cb','gene','pos','allele_ref','ub_ref','allele_alt','ub_alt','umi'])

	select = inferred_allele_df['gene']==gene
	inferred_allele_df_subset = inferred_allele_df.loc[select,:].compute()
	cb_gRNA_df = pd.read_csv(cb_w_gRNA, sep=' ',header=None, names=['cb_w_exp','gRNA','gRNA-grp'])

	bool_var=cb_gRNA_df['gRNA-grp']=='NTC'
	cb_gRNA_df.loc[ bool_var,'grp'] = 'NTC'
	cb_gRNA_df.loc[~bool_var,'grp'] = cb_gRNA_df.loc[~bool_var,'gRNA']

	suffix = ase_infer.split('.')[-1]

	df = pd.merge(inferred_allele_df_subset,cb_gRNA_df,left_on='cb',right_on='cb_w_exp')
	df_grp = df.groupby('grp')
	
	out = pd.DataFrame(columns=['gene','gRNA','allele','ks','mwu','ad'])

	NCT_grp = df_grp.get_group('NTC') 
	processed_NTC_df =  process(NCT_grp)

	for gRNA_grp, idf_grp in df_grp:
		processed_df =  process(idf_grp)
		res = DGE_test(processed_df, processed_NTC_df)
		res['gRNA'] = gRNA_grp
		out=out.append(res[out.columns.tolist()],ignore_index=False)

	if not out.dropna().empty:
		outfile = '%s/%s.de'%(outdir,gene)
		out.dropna().to_csv(outfile,index=False)

if __name__ == "__main__":
	main()

