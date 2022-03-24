import seaborn as sns
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
import pandas as pd
import numpy  as np
import sys

def hist(df):
	
	data_stack = []
	for key, grp in df.groupby('grp'):
		data=grp['cb_count']
		hist, bins = np.histogram(np.log10(data), bins='auto')
		#data_stack.append(data)
		plt.bar(bins, hist)
	plt.gca().set_xscale("log")
	plt.show()

def main():
	
	infile = sys.argv[1]
	df = pd.read_csv(infile,header=0, index_col=0)

	new_labels = {'irb':'Imbalanced random biallelic', 'bb':'Balanced biallelic','ifb':'Imbalanced fixed biallelic',\
			'fm':'Fixed monoallelic','rm':'Random monoallelic'}

	df['group'] = df['grp'].apply(lambda x: new_labels[x])
	
	sns.set_theme(style="ticks")
	
	f, ax = plt.subplots(figsize=(7, 5))
	sns.despine(f)
	
	sns.histplot(
	    df,x="cb_count", hue="group",
	    multiple="stack",
	    palette="husl",
	    edgecolor=".3",
	    linewidth=0.01,
	    hue_order = new_labels.values(),
	    log_scale=True)

	ax.set_ylabel('# genes')
	ax.set_xlabel('expressed in # cells')
	ax.xaxis.set_major_formatter(mpl.ticker.ScalarFormatter())
	plt.savefig(infile+'.pdf')

if __name__ == "__main__":
	main()


