"""
Analytical computation of the functions G and Gbar, for nu = MPastur(c = 1) and alpha = 2.
nu is the distribution of the diagonal d_mu 
"""
import numpy as np
import time, pickle
from scipy import optimize
import argparse
from Classes.Analytical_Solver import Analytical_Solver

if __name__== "__main__":

    verbosity = 1
    nu = "mpastur"
    alpha = 2.
    parameters = {'gamma':1.}
    t0 = time.time()
    Solver = Analytical_Solver(nu, alpha, verbosity, parameters=parameters, return_G_Gbar=True)
    t1 = time.time()
    print("Preprocessing done in ", t1-t0, " seconds")
    lambdamax = Solver.get_lambdamax()
    results = Solver.get_G_Gbar()

    output = {'alpha':alpha, 'nu':nu, 'lambdamax':lambdamax,'xcnu':results['xcnu'],'x_values':results['x_values'], 'G_values':results['G_values'], 'Gbar_values':results['Gbar_values']}
    filename = "Data/analytical/G_functions_" + "nu_"+nu+"_alpha_"+str(alpha)+".pkl"
    outfile = open(filename,'wb')
    pickle.dump(output,outfile)
    outfile.close()