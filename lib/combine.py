import scipy.stats as st
import pandas as pd
import numpy as np
import sys

def main():

	kpe_pb_in   = sys.argv[1]
	kpe_full_in = sys.argv[2]
	
	kpe_pb   = pd.read_csv(kpe_pb_in,   header=0)
	kpe_full = pd.read_csv(kpe_full_in, header=0)

	kpe_pb['index']   = kpe_pb[['gene','allele']].agg('+'.join, axis=1).values
	kpe_full['index'] = kpe_full[['gene','allele']].agg('+'.join, axis=1).values
#	kpe_pb['index']   = kpe_pb[['gene','allele','gRNA']].agg('+'.join, axis=1).values
#	kpe_full['index'] = kpe_full[['gene','allele','gRNA']].agg('+'.join, axis=1).values

	filtered_pb_index = set(kpe_pb['index']) - set(kpe_full['index'])

	kpe_pb.set_index(['index'], inplace=True)
	kpe_pb_filter = kpe_pb.loc[filtered_pb_index].reset_index()

	kpe_final = pd.concat([kpe_pb_filter, kpe_full], ignore_index=True)
	prefix = kpe_pb_in.split('_')[0]
	kpe_final[kpe_pb.columns].to_csv(prefix + '.est.final', index=False,float_format='%.5f')

	kpe_comm = pd.merge(kpe_pb, kpe_full, on=['gene','allele'],suffixes=('_pb','_full'))
	kpe_comm.to_csv(prefix + '.est.comm', index=False,float_format='%.5f')

if __name__ == "__main__":
	main()


