import csv
import gzip
import os
import scipy.io
import sys
import pandas as pd
import dask.dataframe as dd

def main():
	matrix_dir = sys.argv[1]
	outfile    = sys.argv[2]

	mat_path = os.path.join(matrix_dir, "matrix.mtx.gz")	
	mat      = dd.read_csv(mat_path, skiprows=[0,1,2], names=['igene','icb','umi'], \
		sep=' ', compression='gzip')

	features_path = os.path.join(matrix_dir, "features.tsv.gz")
	feature       = dd.read_csv(features_path,names=['id','gene','type'],\
		sep='\t',compression='gzip')

	barcodes_path = os.path.join(matrix_dir, "barcodes.tsv.gz")
	barcodes      = dd.read_csv(barcodes_path,names=['cb'], sep="\t", compression='gzip')

	n_feature = feature.shape[0].compute()
	feature['igene'] = feature.index + 1
	df_12 = dd.merge(feature, mat, on='igene')

	n_barcode = barcodes.shape[0].compute()	
	barcodes['icb'] = barcodes.index + 1
	df = dd.merge(df_12, barcodes, on='icb')

	header=['cb','gene','umi']
	df_uniq = df[header].drop_duplicates(subset=['cb', 'gene'], keep='last')
	df_uniq.to_csv('%s.mtx'%outfile,header=True,index=False,single_file = True)

if __name__ == "__main__":
	main()

