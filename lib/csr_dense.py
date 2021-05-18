import dask.dataframe as dd
import pandas as pd
import numpy as np
import scipy.io
import csv
import gzip
import os
import sys

def main():
	matrix_dir     = sys.argv[1]
	mito_gene_list = sys.argv[2]
	outfile        = sys.argv[3]

	# load mitochondrial genes 
	mito_gene = np.loadtxt(mito_gene_list, dtype='str')
	mito_gene_df = pd.DataFrame(data=np.ones(len(mito_gene)), index=mito_gene, columns=['mito'])

	mat_path = os.path.join(matrix_dir, "matrix.mtx.gz")	
	mat      = dd.read_csv(mat_path, skiprows=[0,1,2], names=['igene','icb','umi'], \
		sep=' ', compression='gzip')

	features_path = os.path.join(matrix_dir, "features.tsv.gz")
	feature_raw   = dd.read_csv(features_path,names=['gene','gene_name','type'],\
		sep='\t',compression='gzip')

	feature = dd.merge(feature_raw,mito_gene_df,left_on='gene',right_index=True,how='left')
	feature['mito'] = feature['mito'].fillna(0)

	barcodes_path = os.path.join(matrix_dir, "barcodes.tsv.gz")
	barcodes      = dd.read_csv(barcodes_path,names=['cb'], sep="\t", compression='gzip')

	n_feature = feature.shape[0].compute()
	feature['igene'] = feature.index + 1
	df_12 = dd.merge(feature, mat, on='igene')

	n_barcode = barcodes.shape[0].compute()	
	barcodes['icb'] = barcodes.index + 1
	df = dd.merge(df_12, barcodes, on='icb')

	# dense matrix
	header=['cb','gene','umi']
	df_uniq = df[header].drop_duplicates(subset=['cb', 'gene'], keep='last')
	df_uniq.to_csv('%s.mtx'%outfile,header=True,index=False,single_file = True)

	# filter valid barcode
	df['mito_umi'] = df['umi'] * df['mito']
	df_stat = df.compute().groupby(['cb']).agg(
		tot_expr = pd.NamedAgg(column='umi',aggfunc='sum'),
		mito_expr = pd.NamedAgg(column='mito_umi',aggfunc='sum'))

	df_stat['mito_ratio'] = df_stat['mito_expr']/df_stat['tot_expr']
	filter_bool = df_stat['mito_ratio'] < 0.1
	cbfile = '%s.cb'%outfile
	np.savetxt(cbfile, df_stat.loc[filter_bool].index, fmt='%s')

if __name__ == "__main__":
	main()

