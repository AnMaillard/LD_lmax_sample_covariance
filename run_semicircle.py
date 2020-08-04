import numpy as np
import time, pickle
from Classes.Monte_Carlo_Solver import Monte_Carlo_Solver
import multiprocessing as mp

#The run file for the MC runs with nu = semicircle

def run_instance(seed, t, NB_DIAGONALS, n, alpha, nu, NB_CYCLES, LENGTH_CYCLE, NB_WARMUP_CYCLES, verbosity, parameters, power_method):
    print("Starting t = ", t)
    np.random.seed(seed)
    seeds = np.random.randint(1e6, size = NB_DIAGONALS) 
    Solver = None
    means = np.zeros(NB_DIAGONALS) 
    stds = np.zeros(NB_DIAGONALS) 
    measurements = [None for k in range(NB_DIAGONALS)]

    for k in range(NB_DIAGONALS):
        print("Starting k = ", k+1, " / ", NB_DIAGONALS)
        seed = seeds[k]
        Solver = Monte_Carlo_Solver(t, n, alpha, nu, seed, NB_CYCLES, LENGTH_CYCLE, NB_WARMUP_CYCLES, verbosity, parameters, power_method)
        mean, std, measurement = Solver.run()
        means[k] = mean
        stds[k] = std
        measurements[k] = measurement

    #Saving 
    output = {'n':n, 'alpha':alpha, 'nu':nu, 'NB_DIAGONALS':NB_DIAGONALS, 'seeds':seeds, 't':t, 'means':means, 'stds':stds, 'measurements':measurements}
    filename = "Data/MonteCarlo/" + "n_"+str(n)+"_alpha_"+str(alpha)+"_nu_"+nu+"_t_"+str(t)+"_additional_points.pkl"
    outfile = open(filename,'wb')
    pickle.dump(output,outfile)
    outfile.close()
    return {'means':means, 'stds':stds, 'measurements':measurements}

if __name__== "__main__":

    nu = "semicircle"
    global_seed = int(time.time())
    np.random.seed(global_seed)
    NB_DIAGONALS = 5 #Nb of independent runs
    verbosity = 2
    n = 500
    NB_CYCLES = 100
    NB_WARMUP_CYCLES = 100
    LENGTH_CYCLE = 5000
    alpha = 1.
    #This is for alpha = 1.
    t_list = np.linspace(0.15,0.05,num = 10)
    #For the inverse power methods we need the expected value of x (from the analytical computation)
    x_expected = [5.21725822, 4.74634376, 4.37243719, 4.06226948, 3.81632135, 3.62947953, 3.48452541, 3.37115175, 3.27898215, 3.2089965 ]
    parameters = [{'x_expected':x_expected[i]} for (i,t) in enumerate(t_list)]
    partial_seeds = np.random.randint(1e6, size = len(t_list))
    power_method = True #Do we use power methods or eigh  #FIXME
    print("Starting nu =", nu, ", n =",n,", alpha =",alpha)
    pool = mp.Pool(processes=12) #The mp pool
    results = [pool.apply(run_instance, args=(partial_seeds[i_t], t, NB_DIAGONALS, n, alpha, nu, NB_CYCLES, LENGTH_CYCLE, NB_WARMUP_CYCLES, verbosity, parameters[i_t], power_method, )) for (i_t,t) in enumerate(t_list)]