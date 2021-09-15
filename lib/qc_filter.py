import scipy.stats as st
import pandas as pd
import numpy as np
import sys

def main():

	cell_qc_in = sys.argv[1]
	cell_ct_in = sys.argv[2]
	prefix     = sys.argv[3]
	
	cell_qc = pd.read_csv(cell_qc_in, header=0)
	cell_ct = pd.read_csv(cell_ct_in, sep='\s+|,',engine='python',header=None,names=['cb','ct'])

	cell_comm = pd.merge(cell_ct, cell_qc, on=['cb'])
	keep = cell_comm['keep']==1
	cell_comm.loc[keep, ['cb','ct']].to_csv(prefix, index=False)

if __name__ == "__main__":
	main()


