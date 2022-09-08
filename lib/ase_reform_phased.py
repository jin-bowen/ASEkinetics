from scipy.sparse import csr_matrix, save_npz
import pandas as pd
import numpy as np
import sys

def process_unphased(df):

	df['ref_infer'] = np.ceil(df['ub_ref'] * df['umi']/ (df['ub_ref'] + df['ub_alt']))
	df['alt_infer'] = df['umi'] - df['ref_infer']

	df['allele_pos'] = df[['gene','pos','allele_ref','allele_alt']].agg('_'.join, axis=1)
	df_grp = df.groupby(['allele_pos'])[['ref_infer','alt_infer']]

	ncb     = df['cb'].nunique() 
	cb_list = df['cb'].unique()
	cb_idx  = list(range(ncb))
	cb_tab  = dict(zip(cb_list, cb_idx))

	processed_df = pd.DataFrame(columns=['gene','allele','cb_umi_list'])
	for key, group in df_grp:
		key_list = key.split('_')
		igene = key_list[0]
		ref_allele_id = key_list[1] + '_' + key_list[2]
		alt_allele_id = key_list[1] + '_' + key_list[3]

		ref_val = group['ref_infer'].tolist()
		alt_val = group['alt_infer'].tolist()

		gcb = group['cb'].tolist()
		gcb_idx = [ cb_tab[igcb] for igcb in gcb ]

		ref_cb_umi = list(zip(gcb_idx, ref_val))	
		alt_cb_umi = list(zip(gcb_idx, alt_val))	

		processed_df.loc[len(processed_df.index)] = [igene, ref_allele_id, ref_cb_umi] 
		processed_df.loc[len(processed_df.index)] = [igene, alt_allele_id, alt_cb_umi]

	processed_df['gene_allele'] = processed_df[['gene','allele']].agg('-'.join, axis=1)
	processed_df.drop(['gene','allele'],axis=1, inplace=True)
	allele_list = processed_df['gene_allele'].values
	ngene    = processed_df.shape[0]

	sparse_df = processed_df.explode('cb_umi_list')	
	sparse_df[['cb','allele_umi']] = pd.DataFrame([*sparse_df['cb_umi_list']], sparse_df.index)
	data = sparse_df['allele_umi'].tolist()
	row  = sparse_df.index.tolist()
	col  = sparse_df['cb'].tolist()

	sparse_mat = csr_matrix((data, (row,col)),dtype=np.float32,shape=(ngene,ncb))
	return sparse_mat, allele_list, cb_list

def process_phased(df):

	df['H1_infer'] = np.ceil(df['ub_H1'] * df['umi']/ (df['ub_H1'] + df['ub_H2']))
	df['H2_infer'] = df['umi'] - df['H1_infer']

	df_grp = df.groupby('gene')[['H1_infer','H2_infer']]

	ncb     = df['cb'].nunique() 
	cb_list = df['cb'].unique()
	cb_idx  = list(range(ncb))
	cb_tab  = dict(zip(cb_list, cb_idx))

	processed_df = pd.DataFrame(columns=['gene','allele','cb_umi_list'])
	for key, group in df_grp:
		igene = key
		H1_val = group['H1_infer'].tolist()
		H2_val = group['H2_infer'].tolist()

		gcb = group['cb'].tolist()
		gcb_idx = [ cb_tab[igcb] for igcb in gcb ]

		H1_cb_umi = list(zip(gcb_idx, H1_val))	
		H2_cb_umi = list(zip(gcb_idx, H2_val))	

		processed_df.loc[len(processed_df.index)] = [igene, 'H1_allele', H1_cb_umi] 
		processed_df.loc[len(processed_df.index)] = [igene, 'H2_allele', H2_cb_umi]

	processed_df['gene_allele'] = processed_df[['gene','allele']].agg('-'.join, axis=1)
	processed_df.drop(['gene','allele'],axis=1, inplace=True)
	allele_list = processed_df['gene_allele'].values
	ngene    = processed_df.shape[0]

	sparse_df = processed_df.explode('cb_umi_list')	
	sparse_df[['cb','allele_umi']] = pd.DataFrame([*sparse_df['cb_umi_list']], sparse_df.index)
	data = sparse_df['allele_umi'].tolist()
	row  = sparse_df.index.tolist()
	col  = sparse_df['cb'].tolist()

	sparse_mat = csr_matrix((data, (row,col)),dtype=np.float32,shape=(ngene,ncb))
	return sparse_mat, allele_list, cb_list

def main():

	ase_infer  = sys.argv[1]
	phased     = sys.argv[2]
	out_prefix = sys.argv[3]

	inferred_allele_df = pd.read_csv(ase_infer,header=0,names=['cb','gene','ub_H1','ub_H2','umi'])

	if phased: sparse_mat, allele_list, cb_list = process_phased(inferred_allele_df)
	else: sparse_mat, allele_list, cb_list = process_unphased(inferred_allele_df)

	out_allele = out_prefix + '.allelei'
	np.savetxt(out_allele, allele_list, fmt='%s')

	out_cb = out_prefix + '.cbi'
	np.savetxt(out_cb, cb_list, fmt='%s')

	out_mat = out_prefix + '.ase.npz'
	save_npz(out_mat,sparse_mat)

if __name__ == "__main__":
	main()

