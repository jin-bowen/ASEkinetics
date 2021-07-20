from joblib import delayed,Parallel
from est import full_MCMC_est
from est import pb_est
from est import gibbs_est
from scipy.stats import ks_2samp, mannwhitneyu, anderson_ksamp
from scipy.stats import poisson, beta
import pandas as pd
import numpy as np
import sys

def est_row(row,method='full'):
	gene   = row['gene']
	allele = row['allele']
	read   = row['list']

	n      = len(read)
	mean   = np.mean(read)
	var    = np.var(read)

	if len(read) < 50:
		return [gene,allele] + [None, None, None] + [n, mean, var]

	if method == 'full':
		obj1  = full_MCMC_est.mRNAkinetics(read)
		obj1.MaximumLikelihood()
		kpe = obj1.get_estimate()
		#kon, koff, ksyn, kon_error, koff_error, ksyn_error = obj1.metropolis_hastings()
		#record = [ kon, koff, ksyn, kon_error, koff_error, ksyn_error ]
		#kpe_df.loc[i] = record
		kon, koff, ksyn = kpe

	elif method == 'pb':
		obj1  = pb_est.mRNAkinetics(read)
		obj1.MaximumLikelihood()
		kpe = obj1.get_estimate()
		kon, koff, ksyn = kpe

	elif method == 'gibbs':
		params1, bioParams1 = gibbs_est.getParamsBayesian(read)
		kon  = params1.alpha.mean()
		koff = params1.beta.mean()
		ksyn = params1.gamma.mean()

	return [ gene, allele, kon, koff, ksyn, n, mean, var ]


def DGE_test(process_df, process_NTC_df):

	full_tab = pd.merge(process_df,process_NTC_df,how='left',on=['gene','allele'],suffixes=('','_NTC'))
	for i, record in full_tab.iterrows():
		if len(record['list_NTC']) < 1:
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
	method    = sys.argv[2]

	cb_w_gRNA = sys.argv[3]
	gene     = sys.argv[4]

	outdir   = sys.argv[5]

	inferred_allele_df = pd.read_csv(ase_infer,header=0)
	select = inferred_allele_df['gene']==gene
	inferred_allele_df_subset = inferred_allele_df.loc[select,:]

	cb_gRNA_df = pd.read_csv(cb_w_gRNA, sep=' ',header=None, names=['cb_w_exp','gRNA','gRNA-grp'])
	bool_var=cb_gRNA_df['gRNA-grp']=='NTC'
	cb_gRNA_df.loc[ bool_var,'grp'] = 'NTC'
	cb_gRNA_df.loc[~bool_var,'grp'] = cb_gRNA_df.loc[~bool_var,'gRNA']
	cb_gRNA_df.drop(['gRNA-grp','gRNA'], axis=1, inplace=True)
	cb_gRNA_df.drop_duplicates(inplace=True)

	df = pd.merge(inferred_allele_df_subset,cb_gRNA_df,left_on='cb',right_on='cb_w_exp')
	df_grp = df.groupby('grp')
	
	NTC_grp = df_grp.get_group('NTC') 
	processed_NTC_df =  process(NTC_grp)

	outfile_kpe_df = '%s/%s.mle'%(outdir,gene)
	outfile_kpe_cols = ['gene','allele','kon','koff','ksyn','n','mean','var','gRNA']
	outfile_kpe = open(outfile_kpe_df, "a+")
	outfile_kpe.write(','.join(outfile_kpe_cols) + '\n')

	outfile_de_df  = '%s/%s.de'%(outdir,gene)
	outfile_de_cols = ['gene','allele','ks','mwu','ad','gRNA']
	outfile_de = open(outfile_de_df, "a+")
	outfile_de.write(','.join(outfile_de_cols) + '\n')

	for gRNA_grp, idf_grp in df_grp:
		processed_df =  process(idf_grp)
		res_de = DGE_test(processed_df, processed_NTC_df)
		res_de['gRNA'] = gRNA_grp
		res_de.dropna(inplace=True)
		res_de[outfile_de_cols].to_csv(outfile_de,header=False,index=False,float_format="%.5f")

		res_kpe = Parallel(n_jobs=8)(delayed(est_row)(row,method) for i,row in processed_df.iterrows())
		res_kpe_df = pd.DataFrame(res_kpe,columns=outfile_kpe_cols[:-1])
		res_kpe_df['gRNA'] = gRNA_grp
		res_kpe_df.dropna(inplace=True)
		res_kpe_df[outfile_kpe_cols].to_csv(outfile_kpe,header=False,index=False,float_format="%.5f")

if __name__ == "__main__":
	main()

