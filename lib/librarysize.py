import seaborn as sns
import pandas as pd
import sys
import matplotlib.pyplot as plt

def main():

	mtx_in = sys.argv[1]
	mtx_df = pd.read_csv(mtx_in, sep='\t')
		
	mtx_grp = mtx_df.groupby('cb')['umi'].sum().reset_index()
	mtx_grp.set_index('cb', inplace=True)
	ax = sns.distplot(mtx_grp)
	mean = mtx_grp.mean().values[0]
	median = mtx_grp.median().values[0]
	ax.set_title('mean=%s \n median=%s'%(mean,median))
	plt.show()

	mtx_grp = mtx_df.groupby('gene')['umi'].sum().reset_index()
	mtx_grp.set_index('gene', inplace=True)
	ax = sns.distplot(mtx_grp)
	mean = mtx_grp.mean().values[0]
	median = mtx_grp.median().values[0]
	ax.set_title('mean=%s \n median=%s'%(mean,median))
	plt.show()


if __name__ == "__main__":
	main()



