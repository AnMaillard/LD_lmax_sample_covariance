"""
A Monte-Carlo solver for computing x*(t) using the Metropolis-Hastings algorithm, for the paper
"Large deviations of extremal eigenvalues of generalized sample covariance matrices"

The only moves are rank-one changes of the matrix, corresponding to the replacement of a z_mu
The MC must be run for a fixed instance of the diagonal d_mu
- For dmax > 0, the Legendre transform F(t) = (1/n) ln E[exp(n*t*lambdamax(Hn))] is defined for 0 <= t < alpha/(2*dmax) 
- For dmax <= 0, the Legendre transform F(t) = (1/n) ln E[exp(n*t*lambdamax(Hn))] is defined for 0 <= t < +infty
"""

import numpy as np
import time
import scipy.linalg as lin
import scipy.special as sc
from scipy import optimize
from scipy import integrate
from .distributions import lambdamin, lambdamax, sc_density, mp_density, cdf_mpastur, cdf_beta, cdf_truncated_gaussian, pdf_beta, pdf_truncated_gaussian, cdf_sc

class Monte_Carlo_Solver():
    """ 
    The solver, for a fixed instance of the diagonal
    nu is the distribution of the diagonal elements, it can be of the following types:
        - "Delta" : a Delta peak in 1
        - "semicircle" : the Wigner semicircle distribution in (-2,2)
        - "mpastur" : the marchenko-pastur distribution with parameter gamma >= 1
        - "doubledelta" : a double Delta peak in +-1 
        - "uniform21" : uniform distribution in (-2,-1)
        - "beta" : The beta distribution with parameters (beta,gamma) and gamma > 1
    """
    def __init__(self, t, n, alpha, nu, seed, NB_CYCLES, LENGTH_CYCLE = 100, NB_WARMUP_CYCLES = 1000, verbosity = 0, parameters = {}, power_method = False):
        self.t = t #This is the t in E[exp(n t lambdamax(Hn))]
        assert t>=0., "ERROR : t must be positive !"
        self.n = n
        self.nu = nu
        self.alpha = alpha
        self.m = int(alpha*n)
        self.verbosity = verbosity
        self.NB_CYCLES = NB_CYCLES #The number of cycles (at the end of each cycle we take a measurement)
        self.NB_WARMUP_CYCLES = NB_WARMUP_CYCLES #Number of warmup cycles, before taking any measurements
        self.LENGTH_CYCLE = LENGTH_CYCLE #The length of a cycle (number of tried moves)
        np.random.seed(seed)
        assert self.nu in ["Delta","semicircle","mpastur","doubledelta","uniform21","beta"], "ERROR : Unknown density for the diagonal"
        if self.nu == "mpastur":
            self.gamma = parameters['gamma'] #Only for the MP case
        elif self.nu == "beta":
            #Only forthe beta case
            self.beta = parameters['beta']
            self.gamma = parameters['gamma']
        self.power_method = power_method #Do we use eigh of power method for computing lambdamax
        if self.power_method:
            #Only for the semicircle case, we use inverse power methods, to compute faster lambdamax
            if self.nu == "semicircle":
                print("We use inverse power methods instead of scipy.linalg.eigh !")
            else:
                print("We use power methods instead of scipy.linalg.eigh !")
        else:
            print("We use scipy.linalg.eigh !")

        #The value of dmax
        self.dmax = 1.
        if self.nu == "semicircle":
            self.dmax = 2.
        elif self.nu == "mpastur":
            self.dmax = lambdamax(self.gamma)
        elif self.nu == "uniform21":
            self.dmax = -1.
        assert self.dmax <= 0 or self.t < self.alpha / (2*self.dmax), "ERROR : t must be smaller than alpha/(2*dmax) !"
    
        #We compute the lambdamax of the asymptotic density mu
        self.compute_lambdamax_Gmax()
        if self.verbosity >= 1:
            print("lambdamax of mu = ", self.lambdamax_mu, " and Gmax = ", self.Gmax)
        
        self.Z = np.zeros((self.n,self.m))
        self.Hn = np.zeros((self.n,self.n))
        self.lambdamax = - np.inf
        self.measurements = -np.inf*np.ones(self.NB_CYCLES) 

        self.generate_diagonal() #Generates the diagonal that we will use
        self.initialize()
        if self.verbosity >=1:
            print("Diagonal generated and matrix initialized !")

        self.beta_d = 1. #The initial inverse temperature used for the moves. 
        self.Delta = 10.#The parameter Delta used in the MC
        #For some measures we empirically adapted the first value of (beta_d,Delta)
        if self.nu == "doubledelta":
            self.beta_d = 2.
        if self.nu == "uniform21":
            self.beta_d = 4.
            self.Delta = 4.

        if self.power_method:
            self.current_maxev = np.random.randn(self.n) #The current eigenvector associated to lambdamax, used in the power methods
            self.current_maxev /= lin.norm(self.current_maxev)
            if self.nu == "semicircle": #For the semicircle case and the inverse power method, we use the expected value of x*(t), obtained with analytical computations
                self.x_expected = parameters['x_expected']
                self.current_inverse = lin.inv(self.Hn - self.x_expected*np.eye(self.n)) #The matrix (H-x)^(-1)
    
    def compute_lambdamax_power(self, MAX_ITERATIONS = 1000, epsilon = 1e-4, MIN_ITERATIONS = 5):
        #Here I compute lambdamax(self.Hn) using a power method. This assumes that lambamax is the largest eigenvalue also in absolute value !
        converged = False
        k = 0
        current_lambdamax = self.lambdamax #Initial vector is self.current_maxev
        maxev = np.copy(self.current_maxev)
        while k < MIN_ITERATIONS or (not(converged) and k < MAX_ITERATIONS):
            maxev_new = np.dot(self.Hn,maxev)
            new_lambdamax = np.dot(maxev,maxev_new)
            converged = (np.abs(new_lambdamax - current_lambdamax)<= epsilon*np.abs(current_lambdamax))
            maxev_new /= lin.norm(maxev_new)
            current_lambdamax = new_lambdamax
            maxev = maxev_new
            k += 1
        if not(converged):
            assert False, "ERROR : Power iteration did not converge"
        return current_lambdamax, maxev
    
    def update_inverse(self, mu_index, zmu, epsilon):
        #I update self.current_inverse when Hn changes by epsilon*d_mu*z_mu z_mu^T / m, using Shermann-Morisson
        #This is only useful for the semicircle density, when power_method = True
        vector = np.dot(self.current_inverse, zmu)
        self.current_inverse += - (epsilon*self.d_list[mu_index]/self.m)*np.outer(vector,vector) / (1.+epsilon*self.d_list[mu_index]*np.dot(zmu,vector)/self.m)

    def compute_lambdamax_inverse(self, MAX_ITERATIONS = 10000, epsilon = 1e-4, MIN_ITERATIONS = 5):
        #Here I compute lambdamax(self.Hn) using a power method which is the inverse iteration
        #Only implemented for the semicircle
        assert self.nu == "semicircle", "ERROR : For now, only implemented for nu = semicircle"
        converged = False
        k = 0
        current_lambdamax = self.lambdamax #Initial vector is self.current_maxev
        maxev = np.copy(self.current_maxev)
        while k < MIN_ITERATIONS or (not(converged) and k < MAX_ITERATIONS):
            maxev_new = np.dot(self.current_inverse,maxev)
            new_lambdamax = self.x_expected + 1./np.dot(maxev,maxev_new) #We have lambda = x + 1/(mu((H -x )^{-1}))
            converged = (np.abs(new_lambdamax - current_lambdamax)<= epsilon*np.abs(current_lambdamax))
            maxev_new /= lin.norm(maxev_new)
            current_lambdamax = new_lambdamax
            maxev = maxev_new
            k += 1
        if not(converged):
            assert False, "ERROR : Power iteration did not converge"
        return current_lambdamax, maxev
        
    def generate_diagonal(self):
        #Generates the fixed diagonal used in the runs
        self.d_list = np.zeros(self.m)
        if self.nu == "delta":
            self.d_list = np.ones(self.m)
        elif self.nu == "doubledelta":
            self.d_list = 2.*np.random.randint(2, size=self.m) - 1
        elif self.nu == "uniform21":
            self.d_list = np.random.random_sample((self.m,)) - 2.
        elif self.nu == "semicircle":
            x = np.random.random_sample((self.m,)) #Uniform array in [0,1]
            #For each mu we find numerically y in [-2,2] s.t. cdf_sc(y) = x, this will be our diagonal
            for mu in range(self.m):
                def function(y):
                    return cdf_sc(y) - x[mu]
                sol = optimize.root_scalar(function, bracket=[-2,2],x0 = 0.)
                self.d_list[mu] = sol.root
        elif self.nu == "mpastur":
            x = np.random.random_sample((self.m,)) #Uniform array in [0,1]
            #For each mu we find numerically y in [lambdamin,lambdamax] s.t. cdf_mpastur(y) = x, this will be our diagonal
            for mu in range(self.m):
                def function(y):
                    return cdf_mpastur(y,self.gamma) - x[mu]
                sol = optimize.root_scalar(function, bracket=[lambdamin(self.gamma),lambdamax(self.gamma)],x0 = (lambdamin(self.gamma)+lambdamax(self.gamma))/2.)
                self.d_list[mu] = sol.root
        elif self.nu == "beta":
            x = np.random.random_sample((self.m,)) #Uniform array in [0,1]
            #For each mu we find numerically y in [lambdamin,lambdamax] s.t. cdf_beta(y) = x, this will be our diagonal
            for mu in range(self.m):
                def function(y):
                    return cdf_beta(y,self.beta,self.gamma) - x[mu]
                sol = optimize.root_scalar(function, bracket=[0.,1.],x0 = 1./2)
                self.d_list[mu] = sol.root
        else:
            assert False, "ERROR : nu is not of a known type"
    
    def initialize(self):
        self.Z = np.random.randn(self.n, self.m) #Generate the Gaussian iid Z matrix
        self.Hn = np.multiply(self.Z, self.d_list) @ np.transpose(self.Z) / self.m #The Hn matrix
        self.lambdamax = lin.eigh(self.Hn, eigvals_only = True, eigvals = ((self.n-1, self.n-1)))[0] #The largest eigenvalue
  
    def compute_lambdamax_Gmax(self):
        #Compute the value of lambdamax, solve the equation on Gmax = G_sigma(\lambda_max)
        if self.nu == "delta":
            def function_Gmax(H):
                return (self.alpha * H**2 / (self.alpha - H)**2) - 1., 2*H*self.alpha**2 / ((self.alpha - H)**3 )
            #We want the smallest positive solution to this equation. Since the function is strictly increasing we can start from very large values
            sol = optimize.root_scalar(function_Gmax, x0 = (self.alpha / self.dmax) * 0.9, fprime = True, method = 'newton')
            self.Gmax = sol.root

        elif self.nu == "doubledelta":
            def function_Gmax(H):
                return (self.alpha/2.) *( (H/ (self.alpha - H))**2 + (H/ (self.alpha + H))**2) - 1. ,  H*self.alpha**2 *(1 / ((self.alpha - H)**3 ) + 1 / ((self.alpha + H)**3 ))
            #We want the smallest positive solution to this equation
            sol = optimize.root_scalar(function_Gmax, x0 = (self.alpha / self.dmax) * 0.9, fprime = True, method = 'newton')
            self.Gmax = sol.root

        elif self.nu == "uniform21":
            def function_Gmax(H):
                return self.alpha*(2 + H*(1./(H+self.alpha) - 4./(2*H+self.alpha)) + (2*self.alpha/H)*np.log((H+self.alpha)/(2*H+self.alpha))) -1. 
            def function_Gmax_prime(H):
                return (self.alpha**2/H**2) *(2.*np.log((2*H+self.alpha)/(H+self.alpha)) - (self.alpha*H) *(8*H**2 + 9*H*self.alpha +2*self.alpha**2) / ((H+self.alpha)**2 * (2*H+self.alpha)**2) )
            #We want the smallest positive solution to this equation
            sol = optimize.root_scalar(function_Gmax, x0 = 1., fprime = function_Gmax_prime, method = 'newton')
            self.Gmax = sol.root

        elif self.nu == "semicircle":
            def function_Gmax(H):
                integral, _ = integrate.quad(lambda t:self.alpha*(H*t/(self.alpha-H*t))**2*sc_density(t), -2., 2.)
                integral_derivative, _ = integrate.quad(lambda t:2*self.alpha**2*H*(t**2/((self.alpha-H*t)**3))*sc_density(t), -2., 2.)
                return integral - 1.,  integral_derivative 
            #We want the smallest positive solution to this equation
            sol = optimize.root_scalar(function_Gmax, x0 = (self.alpha / self.dmax )* 0.9, fprime = True, method = 'newton')
            self.Gmax = sol.root

        elif self.nu == "mpastur":
            def function_Gmax(H):
                integral, _ = integrate.quad(lambda t:self.alpha*(H*t/(self.alpha-H*t))**2*mp_density(t,self.gamma), lambdamin(self.gamma), lambdamax(self.gamma))
                integral_derivative, _ = integrate.quad(lambda t:2*self.alpha**2*H*(t**2/((self.alpha-H*t)**3))*mp_density(t,self.gamma), lambdamin(self.gamma), lambdamax(self.gamma))
                return integral - 1., integral_derivative
            #We want the smallest positive solution to this equation
            sol = optimize.root_scalar(function_Gmax, x0 = (self.alpha / self.dmax )* 0.9, fprime = True, method = 'newton')
            self.Gmax = sol.root

        elif self.nu == "beta":
            def function_Gmax(H):
                integral, _ = integrate.quad(lambda t:self.alpha*(H*t/(self.alpha-H*t))**2*pdf_beta(t,self.beta,self.gamma), 0., 1.)
                integral_derivative, _ = integrate.quad(lambda t:2*self.alpha**2*H*(t**2/((self.alpha-H*t)**3))*pdf_beta(t,self.beta,self.gamma), 0., 1.)
                return integral - 1., integral_derivative
            #We want the smallest positive solution to this equation
            sol = optimize.root_scalar(function_Gmax, x0 = (self.alpha / self.dmax )* 0.9, fprime = True, method = 'newton')
            self.Gmax = sol.root
        else:
            assert False, "ERROR : Unknown nu"
        self.lambdamax_mu = self.inverseStieltjes(self.Gmax)

    def inverseStieltjes(self,H):
        #Returns the inverse Stieltjes transform
        if self.nu == "delta":
            return 1./H + self.alpha / (self.alpha - H) 
        elif self.nu == "doubledelta":
            return 1./H + (self.alpha / 2.) * (1. / (self.alpha - H) - 1./ (self.alpha + H) )
        elif self.nu == "uniform21":
            return (H - self.alpha*H + self.alpha**2 * np.log(1.+ (H/(H+self.alpha)) ) )/ H**2
        elif self.nu == "semicircle":
            def integrand(t):
                return self.alpha * (t / (self.alpha - t*H)) * sc_density(t)
            integral, _ = integrate.quad(integrand, -2., 2.)
            return 1./H + integral
        elif self.nu == "mpastur":
            def integrand(t):
                return self.alpha * (t / (self.alpha - t*H)) * mp_density(t,self.gamma)
            integral, _ = integrate.quad(integrand, lambdamin(self.gamma), lambdamax(self.gamma))
            return 1./H + integral
        elif self.nu == "beta":
            def integrand(t):
                return self.alpha * (t / (self.alpha - t*H)) * pdf_beta(t,self.beta,self.gamma)
            integral, _ = integrate.quad(integrand, 0., 1.)
            return 1./H + integral
        else:
            assert False, "ERROR : Unknown nu"
    
    def measure(self,i):
        #We measure only lambdamax
        self.measurements[i] = self.lambdamax 
    
    def run_warmup_cycle(self, min_acceptance = 0.2, max_acceptance = 0.3):
        accepted_list = [None for k in range(self.LENGTH_CYCLE)]
        for k in range(self.LENGTH_CYCLE):
            accepted_list[k] = self.move() #Do a simple move
        
        #Now we adapt Delta and beta_d to have an acceptance ratio between min_acceptance and max_acceptance
        acceptance_ratio = 1.0*np.sum(accepted_list) / self.LENGTH_CYCLE

        #Empirically, we find that tuning only Delta is sufficient
        factor = 0.8
        if self.nu == "semicircle" or self.nu == "mpastur":
            factor = 0.9
        if acceptance_ratio < min_acceptance:
            self.Delta *= factor
        else:
            self.Delta *= 1./factor

        return acceptance_ratio
    
    def run_cycle(self, i):
        accepted_list = [None for k in range(self.LENGTH_CYCLE)]
        for k in range(self.LENGTH_CYCLE):
            accepted_list[k] = self.move() #Do a simple move
        return 1.0*np.sum(accepted_list) / self.LENGTH_CYCLE

    def move(self):
        #Do a move in which we change a vector z_mu
        #We first pick a random mu, according to a probability proportional to exp[beta_d * d_mu], so that we chose the large d_mu more often
        mu_index = 0
        x = np.random.random_sample() #Uniform in [0,1]
        sum_partial = np.exp(self.beta_d*self.d_list[0])
        sum_tot = np.sum(np.exp(self.beta_d*self.d_list))
        while x > (sum_partial / sum_tot):
            mu_index += 1
            sum_partial += np.exp(self.beta_d*self.d_list[mu_index])

        #We generate zmu by putting more weight on the large norm eigenvectors (with a random direction)
        emu = np.random.randn(self.n)
        emu /= lin.norm(emu) #emu is the direction of z
        #Now we sample the norm of z from a truncated Gaussian probability distribution centered in 1, with a 
        # standard deviation Delta. Since n >> 1 the real distribution would be almost Gaussian with mean 1 and Delta = sqrt(2)/n 
        UPPER_BOUND = 1e3
        start_over = True
        while(start_over):
            u = np.random.random_sample() #Uniform in [0,1]
            def function(y):
                return cdf_truncated_gaussian(y,self.Delta) - u
            if function(UPPER_BOUND) > 0:
                sol = optimize.root_scalar(function, bracket=[0.,UPPER_BOUND],x0 = 1.)
                start_over = False
        norm2_z = sol.root #The SQUARED norm of z, norm2_z = ||z||^2 / n
        zmu = np.sqrt(self.n * norm2_z) * emu

        #Compute the new lambdamax under this change
        old_Hn = np.copy(self.Hn) #CAREFUL : MUST BE A COPY
        self.Hn += self.d_list[mu_index] * (np.outer(zmu,zmu) - np.outer(self.Z[:,mu_index],self.Z[:,mu_index])) / self.m
        if self.power_method:
            if self.nu != "semicircle":
                newlambdamax, new_maxev = self.compute_lambdamax_power()
            else:
                old_inverse = np.copy(self.current_inverse)
                #For semicircle and the power methods, we use the inverse iteration
                #First, we update the inverse using the Shermann-Morisson formula
                self.update_inverse(mu_index, zmu, epsilon = 1)
                self.update_inverse(mu_index, self.Z[:,mu_index], epsilon = -1)
                #Now we use the power method
                newlambdamax, new_maxev = self.compute_lambdamax_inverse()
        else:
            newlambdamax = lin.eigh(self.Hn, eigvals_only = True, eigvals = ((self.n-1, self.n-1)))[0]

        #Compute the acceptance probability, implementing the detailed balance.
        norm2_oldZ = lin.norm(self.Z[:,mu_index])**2/self.n
        p_acceptance = min(1., np.exp(self.n*self.t*(newlambdamax-self.lambdamax)  - self.n*(norm2_z - norm2_oldZ)/2. + ((self.n/2.) - 1)*(np.log(norm2_z) - np.log(norm2_oldZ)))*pdf_truncated_gaussian(norm2_oldZ,self.Delta)/pdf_truncated_gaussian(norm2_z,self.Delta))

        #Accept the move with this probability
        x = np.random.random_sample()
        if x <= p_acceptance: #Accept
            self.Z[:,mu_index] = zmu
            self.lambdamax = newlambdamax
            if self.power_method:
                self.current_maxev = new_maxev
        else:
            self.Hn = old_Hn #Reject
            #FIXME
            if self.power_method and self.nu == "semicircle":
                self.current_inverse = old_inverse

        #Returns True or False if the move is accepted or not
        return(x <= p_acceptance)
        
    def run(self):
        #Do the warmup cycles
        min_acceptance = 0.2
        max_acceptance = 0.3 #Optimal acceptance ratios
        t0 = time.time()
        acceptance_ratios = []
        step_show_warmup = (self.NB_WARMUP_CYCLES / 100)
        if self.NB_WARMUP_CYCLES < 100:
            step_show_warmup = (self.NB_WARMUP_CYCLES / 10)
        for i in range(self.NB_WARMUP_CYCLES):
            if i % step_show_warmup == 0 and self.verbosity >= 2 and i >= 1:
                t1 = time.time()
                print(100*i/self.NB_WARMUP_CYCLES, "% of warmup cycles done, this batch took", round(t1-t0,3), "s. Current lambdamax =",round(self.lambdamax,3), end='. ')
                print("Acceptance ratio in this batch :",round(100*np.mean(acceptance_ratios),3), "%. (beta_d, Delta) : (", round(self.beta_d,3), ",",round(self.Delta,3),")")
                t0 = time.time()
                acceptance_ratios = []
            acceptance_ratios.append(self.run_warmup_cycle(min_acceptance = min_acceptance, max_acceptance = max_acceptance))
        if self.verbosity >= 1:
            print("Warmup cycles completed !")
        #Do the cycles with measure at the end of each cycle
        t0 = time.time()
        acceptance_ratios = []
        step_show = (self.NB_CYCLES / 100)
        if self.NB_CYCLES < 100:
            step_show = (self.NB_CYCLES / 10)
        for i in range(self.NB_CYCLES):
            if i % step_show == 0 and self.verbosity >= 1 and i >= 2:
                t1 = time.time()
                print(100*i/self.NB_CYCLES, " % done, batch took", round(t1-t0,3),"s. ", round(100.*np.mean(acceptance_ratios),3), "% of accepted moves.", end = ' ')
                mean, std = self.compute_mean_std(self.measurements[:i])
                print("Current estimate of x*(t) =", round(mean,3), "+-", round(std,3), " and current lambdamax = ", round(self.lambdamax, 3))
                t0 = time.time()
                acceptance_ratios = []
            acceptance_ratios.append(self.run_cycle(i))
            self.measure(i) #Do a measurement
        
        if self.verbosity >= 2:
            mean, std = self.compute_mean_std(self.measurements)
            print("x*(t) = ", mean, " +- ", std)

        mean, std = self.compute_mean_std(self.measurements)
        return mean, std, self.measurements
    
    def compute_mean_std(self, measurements):
        #Compute the mean and std from the given measurements list
        return np.mean(measurements), np.std(measurements, ddof = 1)