import numpy as np
import pandas as pd
import scipy.stats as st
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
mpl.rcParams['font.family'] = 'Avenir'
plt.rcParams['font.size'] = 12
plt.rcParams['axes.linewidth'] = 2
import sys

def sample(m_param, n=100):
	
	m_kon, m_koff, m_syn  = m_param
	m_array_p = np.ones(n)
	m_array_x = np.ones(n)

	m_array_p = np.random.beta(m_kon,m_koff,n)
	m_array_x = np.random.poisson(m_syn*m_array_p)
	return m_array_x

def hazard(params, pop):

	if len(params) < 6:
		kon, koff, s0 = params
		s1 = d1 = 0
		d0 = 1
	else:kon, koff, s0, s1, d0, d1 = params

	gon, m, p = pop
	return np.array([ gon * koff,
			(1 - gon) * kon,
			gon  * s0,  
			m * d0,
			m * s1,
			p * d1])

def sample_discrete_scipy(probs):
	
	return st.rv_discrete(values=(range(len(probs)), probs)).rvs()

def gillespie_draw(params, hazard, pop):

	# Compute propensities
	prob = hazard(params, pop)
	
	# Sum of propensities
	prob_sum = prob.sum()
	
	# Compute time
	tau = np.random.exponential(1.0 / prob_sum)
	
	# Compute discrete probabilities of each reaction
	rxn_probs = prob / prob_sum

	# Draw reaction from this distribution
	rxn = sample_discrete_scipy(rxn_probs)
	
	return rxn, tau

def gillespie_algorithm(params,hazard, S, T, pop_0,sample_point=200):

	i = 0
	cum_t = 0
	dt = float(T - 0.0)/float(sample_point)

	# starting population is the same for gene1 and gene2
	pop = pop_0.copy()

	r_t   = np.linspace(0, T, num = sample_point + 1)
	r_pop = np.empty( (S.shape[0], sample_point + 1), dtype=np.int)
	r_pop[:,0:] = pop.reshape((-1,1))

	while(cum_t <= T):
		# upstream gene
		j, tau = gillespie_draw(params, hazard, pop)

		cum_t += tau
		i = int(np.floor(cum_t/dt))
		pop += S[:,j]
		r_pop[:,i:] = pop.reshape((-1,1))

	return r_pop, r_t

def main():

	infile  = sys.argv[1]
	params_df = pd.read_csv(infile)
	for i,record in params_df.iterrows():

		params = record.values.round(decimals=3)
		T = 100
		n_sim = 1
		pop_0  = np.array([0,0,0])
		S = np.array([[-1,1,0,0,0,0],
				[0,0,1,-1,0,0],
				[0,0,0,0,1,-1]])
	
		sample_m_p = np.zeros((n_sim,2))
		for n in range(n_sim):
			pop,t = gillespie_algorithm(params,hazard,S,T,pop_0,sample_point=2000)
			sample_m_p[n,0] = pop[1,-1]		
			sample_m_p[n,1] = pop[2,-1]		

		sample_pb = sample(params,n=20000)
	
#		# histgram for mRNA and protein
#		fig, axs = plt.subplots(nrows=2,constrained_layout=True)
#		axs[0].hist(sample_m_p[:,0],density=True)
#		axs[1].hist(sample_m_p[:,1],density=True)
#		plt.show()

		fig, axs = plt.subplots(nrows=3,constrained_layout=True)
		axs[0].set_title('kon=%s,koff=%s,ksyn=%s'%(params[0],params[1],params[2]),loc='right')
		axs[0].plot(t, pop[0,:],'|',lw=0.1,c='k')
		axs[0].grid(False)
		axs[0].set_ylim([0.5, 1.5])
		axs[0].set_ylabel('gene status')
		axs[0].set_xticks([])
		axs[0].set_yticks([])
	
		axs[1].plot(t, pop[1,:], '-', lw=1,c='k')
		axs[1].grid(which='major', color='#CCCCCC', linestyle='--')
		axs[1].set_ylabel('mRNA')
		axs[1].xaxis.set_tick_params(which='major', size=10, width=2, direction='in')
		axs[1].yaxis.set_tick_params(which='major', size=10, width=2, direction='in')
		axs[0].get_shared_x_axes().join(axs[0], axs[1])
		axs[1].set_xlabel('Time(a.u.)')

		axs[2].hist(sample_pb,density=True,color='k')
		axs[2].set_xlabel('mRNA')
		axs[2].set_ylabel('probability density')
		axs[2].xaxis.set_tick_params(which='major', size=10, width=2, direction='in')
		axs[2].yaxis.set_tick_params(which='major', size=10, width=2, direction='in')

		plt.savefig('sge%s_full.tiff'%i,dpi=300,bbox_inches ='tight')
		plt.show()

if __name__ == "__main__":
	main()
