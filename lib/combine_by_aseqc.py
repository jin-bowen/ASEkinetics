from scipy import stats
import pandas as pd
import numpy as np
import sys

def main():

	kpe_in1 = sys.argv[1]
	ase_qc_in = sys.argv[2]
	prefix = kpe_in1.split('.')[0]

	kpe_df1 = pd.read_csv(kpe_in1,header=0, index_col=0)
	kpe_df1.dropna(inplace=True)
	
	kpe_df1[['gene','allele']] = kpe_df1.index.to_series().str.split('-', n=1, expand=True)	
	ase_qc = pd.read_csv(ase_qc_in, header=0)

	filter_gene = ase_qc.loc[ase_qc['qc']==True,'gene']
	kpe_df1_filter = kpe_df1.loc[kpe_df1['gene'].isin(filter_gene)]

	kpe_df1_filter.to_csv(prefix +  '_qc.est')

if __name__ == "__main__":
	main()

