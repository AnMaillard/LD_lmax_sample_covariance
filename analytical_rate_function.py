"""
Computation of the analytical rate function I(x), for
five distributions nu of the d_mu : (delta, doubledelta, mpastur, semicircle, beta) and the given value of alpha
"""
import numpy as np
import time, pickle
from scipy import optimize
from Classes.Analytical_Solver import Analytical_Solver

if __name__== "__main__":

    verbosity = 1

    nu_list = ["delta","doubledelta","semicircle","mpastur","uniform21"]
    alpha_list = np.linspace(1,10,num=10)
    for nu in nu_list:
        for alpha in alpha_list:
            if not (alpha == 1 and nu == "uniform21"): #This point is problematic
                print("Starting alpha = ", alpha, "for nu = ", nu)
                t0 = time.time()
                Solver = Analytical_Solver(nu, alpha, verbosity)
                t1 = time.time()
                print("Preprocessing done in ", t1-t0, " seconds")
                lambdamax = Solver.get_lambdamax()
                if nu == "uniform21": #In this case lambdamax < 0
                    xvalues = np.linspace(lambdamax, -1e-4, num = 10000)
                else: 
                    xvalues = np.linspace(lambdamax, 3*lambdamax, num = 10000)
                print("Computing the rate function..")
                t0 = time.time()
                Ivalues = Solver.compute_rate_function(xvalues)
                t1 = time.time()
                print("Computation of the rate function finished in ", t1-t0, " seconds")

                output = {'alpha':alpha, 'nu':nu, 'lambdamax':lambdamax,'xvalues':xvalues, 'Ivalues':Ivalues}
                filename = "Data/analytical/" + "nu_"+nu+"_alpha_"+str(alpha)+".pkl"
                outfile = open(filename,'wb')
                pickle.dump(output,outfile)
                outfile.close()