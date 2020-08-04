import numpy as np
import time
from scipy import optimize
from scipy import integrate
from scipy import interpolate
import scipy.special as sc
from .distributions import lambdamin, lambdamax, sc_density, mp_density, cdf_mpastur, cdf_beta, cdf_truncated_gaussian, pdf_beta, pdf_truncated_gaussian, cdf_sc

class Analytical_Solver:
    """
    A class to compute the analytical rate function
    """
    def __init__(self, nu, alpha, verbosity = 0, parameters = {}, return_G_Gbar = False):
        """
        nu is the distribution of the diagonal elements, it can be of the following types:
        - "delta" : a delta peak in 1
        - "semicircle" : the semicircle distribution in (-2,2)
        - "mpastur" : the marchenko-pastur distribution with parameter gamma > 1
        - "doubledelta" : a double delta peak in +-1
        - "uniform21" : uniform distribution in (-2,-1)
        - "beta" : the beta distribution with parameters (beta,gamma)

        return_G_Gbar is True if we wish to get the explicit functions G(x) and Gbar(x)
        """
        self.nu = nu
        self.alpha = alpha
        self.verbosity = verbosity
        self.return_G_Gbar = return_G_Gbar
        assert self.alpha >= 1., "ERROR : alpha should always be greater than 1"
        assert self.nu in ["delta","semicircle","mpastur","doubledelta","uniform21","beta"], "ERROR : Unknown density for the diagonal"
        if self.nu == "mpastur":
            self.gamma = parameters['gamma'] 
            assert self.gamma >= 1, "ERROR : only gamma >= 1 for MP"
        if self.nu == "beta":
            self.beta = parameters['beta'] 
            self.gamma = parameters['gamma'] 
            assert self.gamma > 1, "ERROR : only gamma > 1 for Beta"

        #Initialize dmax, Gmaxnu and xc
        self.dmax = 1.
        self.Gmaxnu = np.inf
        self.xcnu = np.inf
        if self.nu == "semicircle":
            self.dmax = 2.
            self.Gmaxnu = 1.
            self.xcnu = self.dmax**2 * self.Gmaxnu + (1./self.alpha-1.)*self.dmax
        elif self.nu == "mpastur":
            self.dmax = lambdamax(self.gamma)
            self.Gmaxnu = self.gamma/(1.+np.sqrt(self.gamma))
            self.xcnu = self.dmax**2 * self.Gmaxnu + (1./self.alpha-1.)*self.dmax
        elif self.nu == "uniform21":
            self.dmax = -1.
        elif self.nu == "beta":
            self.Gmaxnu = sc.gamma(self.gamma-1)*sc.gamma(self.beta+self.gamma) / (sc.gamma(self.gamma)*sc.gamma(self.gamma+self.beta-1.))
            self.xcnu = self.dmax**2 * self.Gmaxnu + (1./self.alpha-1.)*self.dmax
        else:
            assert (nu == "delta" or nu == "doubledelta"), "ERROR : Unknown density nu"

        #Compute lambdamax and Gmax, which will be very useful later
        self.lambdamax = np.inf
        self.Gmax = np.inf
        self.compute_lambdamax_Gmax()
        if self.verbosity >= 1:
            print("lambdamax = ", self.lambdamax, " and Gmax = ", self.Gmax)
        self.preprocess_G_Gbar()    

    def compute_lambdamax_Gmax(self):
        #Compute the value of lambdamax, solve the equation on Gmax
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
            
        self.lambdamax = self.inverseStieltjes(self.Gmax)
    
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

    def preprocess_G_Gbar(self):
        #Compute the rate function from lambdamax to BOUND = 2 xc by default if xc finite, otherwise 10 lambdmax
        self.BOUND = min(2*self.xcnu ,10*self.lambdamax)
        if self.nu == "uniform21": #Negative matrix
            self.BOUND = -1e-4
        NB_POINTS = 100000
        #We take half of points close to lambdamax
        if self.nu == "uniform21":
            x_values = np.linspace(self.lambdamax, self.BOUND, NB_POINTS)
        else:
            x_values = np.linspace(self.lambdamax, 1.5*self.lambdamax, NB_POINTS/2, endpoint=False)
            x_values = np.concatenate((x_values,np.linspace(1.5*self.lambdamax, self.BOUND, NB_POINTS/2)))
        #First we compute G and Gbar
        G_values = self.Gmax*np.ones(NB_POINTS)
        Gbar_values = self.Gmax*np.ones(NB_POINTS)
        for (i,x) in enumerate(x_values):
            if i % (NB_POINTS/100) == 0 and self.verbosity >= 1:
                print(100*i/NB_POINTS, " % done")
            #We will solve the equation on the inverse Stieltjes transform with Newton's method, starting from the previous point. 
            #We have the constraint for G : G in [0,Gmax], and Gbar in [Gmax, alpha/dmax] if x <= xc(nu)
            #Solve for G
            G0 = self.Gmax/2.
            if i >=1 :
                if i>=2:
                    G0 = G_values.g[i-1] 
                sol_G = optimize.root_scalar(lambda t:self.inverseStieltjes(t)-x, bracket=[1e-10,self.Gmax], x0 = G0)
                G_values[i] = sol_G.root

            #Solve for Gbar
            if x >= self.xcnu: #For x >= xc this is just a constant
                Gbar_values[i] = self.alpha / self.dmax
            else:
                Gbar0 = (self.Gmax + self.alpha/self.dmax)/2.
                if i>=1:
                    if i>=2:
                        Gbar0 = Gbar_values[i-1] 
                    upper_bound = self.alpha / self.dmax
                    if self.nu in ["delta","doubledelta"]:
                        upper_bound *= (1-1e-5) #For these nus, Gmaxnu is infinite so we can not take the upper bound to be alpha/dmax 
                    elif self.nu == "uniform21":
                        upper_bound = 10**10
                        Gbar0 = 2*self.Gmax 
                    sol_Gbar = optimize.root_scalar(lambda t:self.inverseStieltjes(t)-x, bracket=[self.Gmax, upper_bound], x0 = Gbar0)
                    Gbar_values[i] = sol_Gbar.root
        
        #Now we construct H and HBar as interpolation objects
        print("Preprocessing of H and Gbar done !")
        self.G_function = interpolate.interp1d(x_values, G_values, kind='cubic', assume_sorted=True)
        self.Gbar_function = interpolate.interp1d(x_values, Gbar_values, kind='cubic', assume_sorted=True)

        if self.return_G_Gbar:
            self.x_values = x_values
            self.G_values = G_values
            self.Gbar_values = Gbar_values
        
    def get_G_Gbar(self):
        assert self.return_G_Gbar, "ERROR : Returning G and Gbar was not specified when creating the Solver"
        return {'x_values':self.x_values, 'G_values':self.G_values, 'Gbar_values':self.Gbar_values, 'xcnu':self.xcnu}

    def get_lambdamax(self):
        return self.lambdamax
        
    def compute_rate_function(self, xvalues):
        #Computes and outputs the rate function
        assert np.max(xvalues) <= self.BOUND, "ERROR : x is larger than the required BOUND"
        def rate_function(x):
            integral, _ = integrate.quad(lambda t:(self.Gbar_function(t) - self.G_function(t))/2., self.lambdamax, x)
            return integral
        result = []
        for (i,x) in enumerate(xvalues):
            if i % (len(xvalues)/100) == 0:
                print(i*100/len(xvalues), "% done for the rate function.")
            result.append(rate_function(x))
        return np.array(result)