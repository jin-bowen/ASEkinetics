import matplotlib.pyplot as plt
import dask.dataframe as dd
from scipy import sparse
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
	df['mito_umi'] = df['mito'] * df['umi']

	cb_stat = df.groupby('cb').agg({'igene': ['count'],\
					'umi':['sum'],\
					'mito_umi':['sum']}).compute()

	cb_stat.columns = ['num_gene','num_umi','mito_umi']
	cb_stat['mito_percent'] = cb_stat['mito_umi'] / cb_stat['num_umi']
	num_gene_mean = np.mean(cb_stat['num_gene'])
	num_gene_std  = np.std(cb_stat['num_gene'])
	num_umi_mean  = np.mean(cb_stat['num_umi'])
	num_umi_std  = np.std(cb_stat['num_umi'])

	select1 = abs(cb_stat['num_gene'] - num_gene_mean) < num_gene_std
	select2 = abs(cb_stat['num_umi'] - num_umi_mean) < num_umi_std
	select3 = cb_stat['mito_percent'] < 0.2
	cb_stat['keep'] = 0
	cb_stat.loc[select1 & select2 & select3,'keep'] = 1

	cb_stat[['num_gene','num_umi','mito_percent','keep']].to_csv(outfile+'.qc')
	
	# dense matrix
	header=['cb','gene','umi']
	df[header].to_csv('%s.mtx'%outfile,header=True,index=False,single_file = True)

if __name__ == "__main__":
	main()

