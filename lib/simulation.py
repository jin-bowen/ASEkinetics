import numpy as np
import scipy.stats as st
import matplotlib.pyplot as plt

def hazard(params, pop):

	kon, koff, s0, s1, d0, d1 = params
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

def gillespie_algorithm(params,hazard, S, T, pop_0, 
			sample_point=200):

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

	# mRNA expression with two state model
	# kon, koff, s0, s1, d0, d1 = params
#	params = np.array([0.94,2,100,10,1,0.1])
	params = np.array([1.86164,177.51201,262.44635,0.00525,1,0.94594])

	T = 2000
	n_sim = 1
	pop_0  = np.array([0,0,0])
	S = np.array([[-1,1,0,0,0,0],
			[0,0,1,-1,0,0],
			[0,0,0,0,1,-1]])

	sample_m_p = np.zeros((n_sim,2))
	for n in range(n_sim):
		pop,t = gillespie_algorithm(params,hazard,S,T,pop_0,sample_point=500)
		sample_m_p[n,0] = pop[1,-1]		
		sample_m_p[n,1] = pop[2,-1]		
#	np.savetxt('sge.txt',sample_m_p)

#	# histgram for mRNA and protein
#	fig, axs = plt.subplots(nrows=2,constrained_layout=True)
#	axs[0].hist(sample_m_p[:,0],density=True)
#	axs[1].hist(sample_m_p[:,1],density=True)
#	plt.show()

	fig, axs = plt.subplots(nrows=3,sharex=True,constrained_layout=True )
	fig.suptitle('gene expresison simulation')
	axs[0].plot(t, pop[0,:],'|',lw=0.1,c='k')
	axs[0].set_ylim([0.5, 1.5])
	axs[0].set_ylabel('gene status')

	axs[1].plot(t, pop[1,:], '-', lw=1,c='k')
	axs[1].grid()
	axs[1].grid(which='major', color='#CCCCCC', linestyle='--')
	axs[1].set_ylabel('mRNA')

	axs[2].plot(t, pop[2,:], '-', lw=1,c='k')
	axs[2].grid()
	axs[2].grid(which='major', color='#CCCCCC', linestyle='--')
	axs[2].set_ylabel('Protein')
	axs[2].set_xlabel('time/arbitrary unit')
	plt.subplots_adjust(hspace = .001)
	plt.savefig('sge_full.png')
	plt.show()

if __name__ == "__main__":
	main()
