from scipy import stats
import pandas as pd
import numpy as np
import sys

def fun(x,y):

	if np.isnan(x) and np.isnan(y): return 1

	logx = np.log10(x)
	logy = np.log10(y)
	fc   = np.log10(2)
	if logy > logx - fc and logy < logx + fc: return 1
	else: return 0

def compare(kpe_df1, kpe_df2, label1, label2):

	kpe_pair_df = pd.merge(kpe_df1,kpe_df2,left_index=True, right_index=True,\
			suffixes = ('_%s'%label1,'_%s'%label2))
	kp_cols = ['kon','koff','ksyn']	
#	kp_cols = ['burst size','burst frequency']
	kp_df1_cols = [ x + '_%s'%label1 for x in kp_cols ]
	kp_df2_cols = [ x + '_%s'%label2 for x in kp_cols ]

	display_cols = kp_df1_cols+kp_df2_cols
	display_cols += [ 'mean_%s'%label1, 'mean_%s'%label2]
	display_cols += ['col']

	compare_col = []
	# method comparison only
	for i,(ikp, jkp) in enumerate(zip(kp_df1_cols, kp_df2_cols)):
		new_col = 'col' + str(i)
		kpe_pair_df[new_col] = kpe_pair_df.apply(lambda row: fun(row[ikp],  row[jkp]),  axis=1)
		compare_col.append(new_col)

	kpe_pair_df['col'] = kpe_pair_df[compare_col].sum(axis=1)
	print(kpe_pair_df['col'].value_counts(ascending=True)/len(kpe_pair_df))
	print(kpe_pair_df.groupby('col')['n_%s'%label2].mean())
	
def main():

	kpe_in1 = sys.argv[1]
	label1 = sys.argv[2]
	kpe_in2 = sys.argv[3]
	label2 = sys.argv[4]
	prefix = sys.argv[5]	

	kpe_df1 = pd.read_csv(kpe_in1,header=0, index_col=0)
	kpe_df2 = pd.read_csv(kpe_in2,header=0, index_col=0)	
	compare(kpe_df1, kpe_df2, label1, label2)

#	kpe_df1 = kpe_df1[kpe_df1['pass']==1]
#	kpe_df2 = kpe_df2[kpe_df2['pass']==1]

	filtered_index = set(kpe_df2.index.tolist()) - set(kpe_df1.index.tolist())
	kpe_df2_filter = kpe_df2.loc[filtered_index]
	
	print(len(kpe_df1.index.tolist()))
	print(len(kpe_df2.index.tolist()))
	print(len(kpe_df2_filter))

	kpe_final = pd.concat([kpe_df1, kpe_df2_filter])
	kpe_final.to_csv(prefix + '.est.final',float_format='%.5f')

if __name__ == "__main__":
	main()


